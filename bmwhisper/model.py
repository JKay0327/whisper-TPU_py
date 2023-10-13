import os
import time
import base64
import gzip
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .decoding import decode as decode_function
from .decoding import detect_language as detect_language_function
from .transcribe import transcribe as transcribe_function
from .untool import Tool, make_np2c, data_type, data_type_map

from tpu_perf.infer import SGInfer, nptype

@dataclass
class ModelDimensions:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int


class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)


class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )


class Conv1d(nn.Conv1d):
    def _conv_forward(
        self, x: Tensor, weight: Tensor, bias: Optional[Tensor]
    ) -> Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )


def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
            # otherwise, perform key/value projections for self- or cross-attention as usual.
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            # for cross-attention, calculate keys and values once and reuse in subsequent calls.
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk

    def qkv_attention(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ):
        _, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        qk = q @ k
        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]
        qk = qk.float()

        w = F.softmax(qk, dim=-1).to(q.dtype)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()


class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)

        self.cross_attn = (
            MultiHeadAttention(n_state, n_head) if cross_attention else None
        )
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state)
        )
        self.mlp_ln = LayerNorm(n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        x = x + self.attn(self.attn_ln(x), mask=mask)[0]
        if self.cross_attn:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa)[0]
        x = x + self.mlp(self.mlp_ln(x))
        return x


class AudioEncoder(nn.Module):
    def __init__(
        self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        super().__init__()
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )
        self.ln_post = LayerNorm(n_state)

    def forward(self, x: Tensor):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)

        assert x.shape[ 1:] == self.positional_embedding.shape, "incorrect audio shape"
        x = (x + self.positional_embedding).to(x.dtype)

        for block in self.blocks:
            x = block(x)

        x = self.ln_post(x)
        return x


class TextDecoder(nn.Module):
    def __init__(
        self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [
                ResidualAttentionBlock(n_state, n_head, cross_attention=True)
                for _ in range(n_layer)
            ]
        )
        self.ln = LayerNorm(n_state)

        mask = torch.empty(n_ctx, n_ctx).fill_(-10000).triu_(1).flip(1)

        self.mask = mask
    
    def embedding(self, x: Tensor):
        return self.token_embedding(x)
    
    def attention(self, x: Tensor, xa: Tensor, kv_cache: Optional[dict] = None):
        for block in self.blocks:
            x = block(x, xa, mask=self.mask, kv_cache=kv_cache)
        x = self.ln(x)
        logits = (
            x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)
        ).float()

        return logits

    def forward(self, x: Tensor, xa: Tensor, kv_cache: Optional[dict] = None):
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : torch.Tensor, shape = (batch_size, n_mels, n_audio_ctx)
            the encoded audio features to be attended on
        """
        x_embedding = self.token_embedding(x)
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        x = (
            x_embedding
            + self.positional_embedding[offset : offset + x.shape[-1]]
        )
        # TODO: replace self.positon_embedding with a npz file
        x = x.to(xa.dtype)

        return self.attention(x, xa, kv_cache=kv_cache)

class Whisper(nn.Module):
    def __init__(self, dims: ModelDimensions, args):
        super().__init__()
        self.dims = dims
        self.model_name = args["model_name"]
        self.encoder = None
        self.decoder = None
        self.encoder_infer = None
        self.logits_decoder_infer = None
        self.decoder_main_infer = None
        self.decoder_post_infer = None
        self.decoder_loop_infer = None
        self.inference = args["inference"]
        self.export_onnx = args["export_onnx"]
        self.fp16 = args["fp16"]
        self.bmodel_dir = args["bmodel_dir"]
        self.beam_size = args["beam_size"]
        self.use_kvcache = args["use_kvcache"]
        self.split = args["split"]
        self.padding_size = args["padding_size"]
        self.quant = args["quant"]
        self.log = args["log"]
        self.runtime = args["runtime"]

        if not self.inference:
            self.encoder = AudioEncoder(
                self.dims.n_mels,
                self.dims.n_audio_ctx,
                self.dims.n_audio_state,
                self.dims.n_audio_head,
                self.dims.n_audio_layer,
            )
            self.decoder = TextDecoder(
                self.dims.n_vocab,
                self.dims.n_text_ctx,
                self.dims.n_text_state,
                self.dims.n_text_head,
                self.dims.n_text_layer,
            )

        # use the last half layers for alignment by default; see `set_alignment_heads()` below
        all_heads = torch.zeros(
            self.dims.n_text_layer, self.dims.n_text_head, dtype=torch.bool
        )
        all_heads[self.dims.n_text_layer // 2 :] = True
        self.alignment_heads = all_heads.to_sparse()

        # get positional embedding from npz file
        positional_embedding_path = os.path.join(os.path.dirname(__file__), "assets", f"positional_embedding_{self.model_name}.npz")
        assert os.path.exists(positional_embedding_path), f"{positional_embedding_path} not found"
        self.positional_embedding = torch.tensor(np.load(positional_embedding_path)["positional_embedding"])

        ############################
        ## BModel Loading
        ############################
        if self.inference:
            if self.fp16:
                dtype = "f16"
            else:
                dtype = "f32"

            quant_str = "all_quant"
            if self.quant:
                encoder_bmodel_path = f"{quant_str}_encoder_{self.model_name}_{self.beam_size}beam_{self.padding_size}pad_1684x_f16.bmodel"
                logits_decoder_bmodel_path = f"{quant_str}_logits_decoder_{self.model_name}_{self.beam_size}beam_{self.padding_size}pad_1684x_f16.bmodel"
            else:
                encoder_bmodel_path = f"encoder_{self.model_name}_{self.beam_size}beam_{self.padding_size}pad_1684x_f16.bmodel"
                logits_decoder_bmodel_path = f"logits_decoder_{self.model_name}_{self.beam_size}beam_{self.padding_size}pad_1684x_f16.bmodel"
            
            encoder_bmodel_path = os.path.join(self.bmodel_dir, encoder_bmodel_path)
            logits_decoder_bmodel_path = os.path.join(self.bmodel_dir, logits_decoder_bmodel_path)
            
            start_time = time.time()
            assert os.path.exists(encoder_bmodel_path), f"{encoder_bmodel_path} not found"
            self.encoder_infer = SGInfer(encoder_bmodel_path, 1, [0])
            assert os.path.exists(logits_decoder_bmodel_path), f"{logits_decoder_bmodel_path} not found"
            self.logits_decoder_infer = SGInfer(logits_decoder_bmodel_path, 1, [0])

            self.decoder_post_infer = None
            self.decoder_main_infer = None
            self.decoder_loop_infer = None

            ############################
            ## Using Untool
            ############################
            self.tool = Tool()
            self.handle = self.tool.bmhandle(0)
            self.bmrt1 = self.tool.bmrt(self.handle)
            self.bmrt2 = self.tool.bmrt(self.handle)
            self.bmrt3 = self.tool.bmrt(self.handle)
            self.bmrt4 = self.tool.bmrt(self.handle)
            self.bmrt5 = self.tool.bmrt(self.handle)
            encoder_bmodel_path        = f"{quant_str}_encoder_{self.model_name}_{self.beam_size}beam_{self.padding_size}pad_1684x_f16.bmodel"
            logits_decoder_bmodel_path = f"{quant_str}_logits_decoder_{self.model_name}_{self.beam_size}beam_{self.padding_size}pad_1684x_f16.bmodel"
            decoder_main_bmodel_path   = f"{quant_str}_decoder_main_with_kvcache_{self.model_name}_{self.beam_size}beam_{self.padding_size}pad_1684x_f16.bmodel"
            decoder_post_bmodel_path   = f"{quant_str}_decoder_post_{self.model_name}_{self.beam_size}beam_{self.padding_size}pad_1684x_f16.bmodel"
            decoder_loop_bmodel_path   = f"{quant_str}_decoder_loop_with_kvcache_{self.model_name}_{self.beam_size}beam_{self.padding_size}pad_1684x_f16.bmodel"
            encoder_bmodel_path        = os.path.join(self.bmodel_dir, encoder_bmodel_path)
            logits_decoder_bmodel_path = os.path.join(self.bmodel_dir, logits_decoder_bmodel_path)
            decoder_main_bmodel_path   = os.path.join(self.bmodel_dir, decoder_main_bmodel_path)
            decoder_post_bmodel_path   = os.path.join(self.bmodel_dir, decoder_post_bmodel_path)
            decoder_loop_bmodel_path   = os.path.join(self.bmodel_dir, decoder_loop_bmodel_path)
            self.encoder_handle        = self.tool.create_model(encoder_bmodel_path.encode("utf-8"), self.bmrt1)
            self.logits_decoder_handle = self.tool.create_model(logits_decoder_bmodel_path.encode("utf-8"), self.bmrt2)
            self.decoder_main_handle   = self.tool.create_model(decoder_main_bmodel_path.encode("utf-8"), self.bmrt3)
            self.decoder_main_handle   = self.tool.create_model(decoder_main_bmodel_path.encode("utf-8"), self.bmrt3)
            self.decoder_post_handle   = self.tool.create_model(decoder_post_bmodel_path.encode("utf-8"), self.bmrt4)
            self.decoder_loop_handle   = self.tool.create_model(decoder_loop_bmodel_path.encode("utf-8"), self.bmrt5)
            self.runtime1 = self.tool.create_un_runtime(self.handle)
            self.runtime2 = self.tool.create_un_runtime(self.handle)
            self.runtime3 = self.tool.create_un_runtime(self.handle)
            self.runtime4 = self.tool.create_un_runtime(self.handle)
            self.runtime5 = self.tool.create_un_runtime(self.handle)
            self.tool.set_bmodel_info(self.runtime1, self.encoder_handle)
            self.tool.set_bmodel_info(self.runtime2, self.logits_decoder_handle)
            self.tool.set_bmodel_info(self.runtime3, self.decoder_main_handle)
            self.tool.set_bmodel_info(self.runtime4, self.decoder_post_handle)
            self.tool.set_bmodel_info(self.runtime5, self.decoder_loop_handle)
            self.tool.set_stage(self.runtime1, 0)
            self.tool.set_stage(self.runtime2, 0)
            self.tool.set_stage(self.runtime3, 0)
            self.tool.set_stage(self.runtime4, 0)
            self.tool.set_stage(self.runtime5, 0)
            self.tool.init_all_tensors(self.runtime1)
            self.tool.init_all_tensors(self.runtime2)
            self.tool.init_all_tensors(self.runtime3)
            self.tool.init_all_tensors(self.runtime4)
            self.tool.init_all_tensors(self.runtime5)
            self.tool.malloc_device_address(self.runtime1)
            self.tool.malloc_device_address(self.runtime2)
            self.tool.malloc_device_address(self.runtime3)
            self.tool.malloc_device_address(self.runtime4)
            self.tool.malloc_device_address(self.runtime5)

            kvcache_rearrange_bmodel_path = f"{quant_str}_kvcache_rearrange_{self.model_name}_{self.beam_size}beam_{self.padding_size}pad_1684x_f16.bmodel"
            kvcache_rearrange_bmodel_path = os.path.join(self.bmodel_dir, kvcache_rearrange_bmodel_path)
            self.kvcache_rearrange_bmrt = []
            self.kvcache_rearrange_handle = []
            self.kvcache_rearrange_runtime = []
            for i in range(self.dims.n_text_layer * 2):
                bmrt = self.tool.bmrt(self.handle)
                handle = self.tool.create_model(kvcache_rearrange_bmodel_path.encode("utf-8"), bmrt)
                runtime = self.tool.create_un_runtime(self.handle)
                self.tool.set_bmodel_info(runtime, handle)
                self.tool.set_stage(runtime, 0)
                self.tool.init_all_tensors(runtime)
                self.tool.malloc_device_address(runtime)
                self.tool.set_input_tensor(runtime, 0, self.tool.get_output_tensor(self.runtime3, i + 1))
                self.tool.set_output_tensor(runtime, 0, self.tool.get_input_tensor(runtime, 0))

                self.kvcache_rearrange_bmrt.append(bmrt)
                self.kvcache_rearrange_handle.append(handle)
                self.kvcache_rearrange_runtime.append(runtime)

            for i in range(self.dims.n_text_layer * 4):
                self.tool.set_input_tensor(self.runtime5, i + 3, self.tool.get_output_tensor(self.runtime3, i + 1))
            for i in range(self.dims.n_text_layer * 2):
                self.tool.set_output_tensor(self.runtime5, i + 1, self.tool.get_input_tensor(self.runtime5, i + 3))
            kvcache_rearrange_runtime_base = self.kvcache_rearrange_runtime[0]
            for i in range(self.dims.n_text_layer * 2 - 1):
                self.tool.set_input_tensor(self.kvcache_rearrange_runtime[i + 1], 1, self.tool.get_input_tensor(kvcache_rearrange_runtime_base, 1))
            
            if self.model_name == "small":
                # self.paddings = [320, 384, 448]
                self.paddings = [448]
            else:
                self.paddings = [448]
            # self.paddings= [256]
            self.encoder_infer_zoo = {}
            self.logits_decoder_infer_zoo = {}
            self.decoder_post_infer_zoo = {}
            self.decoder_main_infer_zoo = {}
            self.decoder_loop_infer_zoo = {}

            for pad in self.paddings:
                if self.quant:
                    decoder_post_bmodel_path = f"{quant_str}_decoder_post_{self.model_name}_{self.beam_size}beam_{pad}pad_1684x_f16.bmodel"
                    if self.use_kvcache:
                        decoder_main_bmodel_path = f"{quant_str}_decoder_main_with_kvcache_{self.model_name}_{self.beam_size}beam_{pad}pad_1684x_f16.bmodel"
                        decoder_loop_bmodel_path = f"{quant_str}_decoder_loop_with_kvcache_{self.model_name}_{self.beam_size}beam_{pad}pad_1684x_f16.bmodel"
                    else:
                        decoder_main_bmodel_path = f"{quant_str}_decoder_main_{self.model_name}_{self.beam_size}beam_{pad}pad_1684x_f16.bmodel"
                        decoder_loop_bmodel_path = f"{quant_str}_decoder_loop_{self.model_name}_{self.beam_size}beam_{pad}pad_1684x_f16.bmodel"
                else:
                    decoder_post_bmodel_path = f"decoder_post_{self.model_name}_{self.beam_size}beam_{pad}pad_1684x_f16.bmodel"
                    if self.use_kvcache:
                        decoder_main_bmodel_path = f"decoder_main_with_kvcache_{self.model_name}_{self.beam_size}beam_{pad}pad_1684x_f16.bmodel"
                        decoder_loop_bmodel_path = f"decoder_loop_with_kvcache_{self.model_name}_{self.beam_size}beam_{pad}pad_1684x_f16.bmodel"
                    else:
                        decoder_main_bmodel_path = f"decoder_main_{self.model_name}_{self.beam_size}beam_{pad}pad_1684x_f16.bmodel"
                        decoder_loop_bmodel_path = f"decoder_loop_{self.model_name}_{self.beam_size}beam_{pad}pad_1684x_f16.bmodel"
                
                # encoder_bmodel_path = os.path.join(self.bmodel_dir, encoder_bmodel_path)
                # logits_decoder_bmodel_path = os.path.join(self.bmodel_dir, logits_decoder_bmodel_path)
                decoder_post_bmodel_path = os.path.join(self.bmodel_dir, decoder_post_bmodel_path)
                decoder_main_bmodel_path = os.path.join(self.bmodel_dir, decoder_main_bmodel_path)
                decoder_loop_bmodel_path = os.path.join(self.bmodel_dir, decoder_loop_bmodel_path)

                # assert os.path.exists(encoder_bmodel_path), f"{encoder_bmodel_path} not found"
                # self.encoder_infer_zoo[pad] = SGInfer(encoder_bmodel_path, 1, [0])
                # assert os.path.exists(logits_decoder_bmodel_path), f"{logits_decoder_bmodel_path} not found"
                # self.logits_decoder_infer_zoo[pad] = SGInfer(logits_decoder_bmodel_path, 1, [0])
                assert os.path.exists(decoder_post_bmodel_path), f"{decoder_post_bmodel_path} not found"
                self.decoder_post_infer_zoo[pad] = SGInfer(decoder_post_bmodel_path, 1, [0])
                assert os.path.exists(decoder_main_bmodel_path), f"{decoder_main_bmodel_path} not found"
                self.decoder_main_infer_zoo[pad] = SGInfer(decoder_main_bmodel_path, 1, [0])
                assert os.path.exists(decoder_loop_bmodel_path), f"{decoder_loop_bmodel_path} not found"
                self.decoder_loop_infer_zoo[pad] = SGInfer(decoder_loop_bmodel_path, 1, [0])

            # if self.use_kvcache:
            #     decoder_main_bmodel_path = os.path.join(self.bmodel_dir, f"decoder_main_with_kvcache_{self.model_name}_{self.beam_size}beam_{self.padding_size}pad_1684x_f16.bmodel")
            #     decoder_main_bmodel_path = os.path.join(self.bmodel_dir, f"quant_decoder_main_with_kvcache_{self.model_name}_{self.beam_size}beam_{self.padding_size}pad_1684x_f16.bmodel")
            #     self.decoder_main_infer = SGInfer(decoder_main_bmodel_path, 1, [0])
            #     decoder_loop_bmodel_path = os.path.join(self.bmodel_dir, f"decoder_loop_with_kvcache_{self.model_name}_{self.beam_size}beam_{self.padding_size}pad_1684x_f16.bmodel")
            #     decoder_loop_bmodel_path = os.path.join(self.bmodel_dir, f"quant_decoder_loop_with_kvcache_{self.model_name}_{self.beam_size}beam_{self.padding_size}pad_1684x_f16.bmodel")
            #     self.decoder_loop_infer = SGInfer(decoder_loop_bmodel_path, 1, [0])
            # else:
            #     decoder_main_bmodel_path = os.path.join(self.bmodel_dir, f"decoder_main_{self.model_name}_{self.beam_size}beam_{self.padding_size}pad_1684x_f16.bmodel")
            #     decoder_main_bmodel_path = os.path.join(self.bmodel_dir, f"quant_decoder_main_{self.model_name}_{self.beam_size}beam_{self.padding_size}pad_1684x_f16.bmodel")
            #     self.decoder_main_infer = SGInfer(decoder_main_bmodel_path, 1, [0])
            #     decoder_loop_bmodel_path = os.path.join(self.bmodel_dir, f"decoder_loop_{self.model_name}_{self.beam_size}beam_{self.padding_size}pad_1684x_f16.bmodel")
            #     decoder_loop_bmodel_path = os.path.join(self.bmodel_dir, f"quant_decoder_loop_{self.model_name}_{self.beam_size}beam_{self.padding_size}pad_1684x_f16.bmodel")
            #     self.decoder_loop_infer = SGInfer(decoder_loop_bmodel_path, 1, [0])
            print("--- %s seconds ---" % (time.time() - start_time))
        # else:
        #     self.paddings= [self.padding_size]
        self.time = 0
        self.main_loop_cnt = 0
        self.call_encoder = 0
        self.call_logits_decoder= 0
        self.call_decoder_loop= 0
        self.call_decoder_firstly= 0
        self.call_decoder_with_kvcache = 0
        self.max_ctx = 0

    def set_alignment_heads(self, dump: bytes):
        array = np.frombuffer(
            gzip.decompress(base64.b85decode(dump)), dtype=bool
        ).copy()
        mask = torch.from_numpy(array).reshape(
            self.dims.n_text_layer, self.dims.n_text_head
        )
        self.alignment_heads = mask.to_sparse()

    def embed_audio(self, mel: torch.Tensor):
        return self.encoder(mel)

    def logits(self, tokens: torch.Tensor, audio_features: torch.Tensor):
        print("{:=^80}".format(" model.logits "))
        # TODO: condition of multi-channel audio
        if self.inference:
            # import pdb; pdb.set_trace()
            # tokens = tokens.numpy().astype(np.int32)
            # audio_features = audio_features.numpy()
            # audio_features = audio_features.numpy().astype(nptype(self.logits_decoder_infer.get_input_info()["audio_features"]["dtype"]))
            start_time = time.time()



            logits_decoder_info = self.tool.model_info(self.logits_decoder_handle)
            tokens_input_dtype = data_type_map[logits_decoder_info['input_dtypes'][0]]
            audio_features_input_dtype = data_type_map[logits_decoder_info['input_dtypes'][1]]
            tokens = tokens.numpy().astype(tokens_input_dtype)
            audio_features = audio_features.numpy().astype(audio_features_input_dtype)
            tokens = tokens if tokens.flags.c_contiguous else np.ascontiguousarray(tokens)
            audio_features = audio_features if audio_features.flags.c_contiguous else np.ascontiguousarray(audio_features)

            self.tool.copy_data_from_numpy(self.tool.get_input_tensor(self.runtime2, 0), make_np2c(tokens), data_type[tokens_input_dtype])
            self.tool.copy_data_from_numpy(self.tool.get_input_tensor(self.runtime2, 1), make_np2c(audio_features), data_type[audio_features_input_dtype])
            self.tool.force_host_to_device(self.tool.get_input_tensor(self.runtime2, 0), self.handle)
            self.tool.force_host_to_device(self.tool.get_input_tensor(self.runtime2, 1), self.handle)

            logits = np.empty(logits_decoder_info[0]['output_shapes'][0], dtype=data_type_map[logits_decoder_info['output_dtypes'][0]])
            self.tool.copy_data_from_numpy(self.tool.get_output_tensor(self.runtime2, 0), make_np2c(logits), logits_decoder_info['output_dtypes'][0])
            # pdb.set_trace()
            self.tool.inference(self.runtime2)
            self.tool.copy_output_data_to_host(self.runtime2)
            # tool.print_output_data(self.model.runtime4)
            # import pdb; pdb.set_trace()

            logits = torch.from_numpy(logits)



            # _ = self.logits_decoder_infer.put(tokens, audio_features)
            # _, result, _ = self.logits_decoder_infer.get()
            # logits = torch.from_numpy(result[0])
            print(f"logits inference time: {time.time() - start_time} seconds")
            self.time += time.time() - start_time
            # import pdb; pdb.set_trace()
            # logits = torch.from_numpy(result[0].astype(np.float32))
        else:
            # import pdb; pdb.set_trace()
            if self.export_onnx:
                onnx_input_names = ["tokens", "audio_features"]
                onnx_output_names = ["logits",]
                onnx_input_dict = {"tokens":tokens, "audio_features":audio_features}
                model_name= f"logits_decoder_{self.model_name}_{self.beam_size}beam_{self.padding_size}pad"

                np.savez(model_name + "_inputs.npz", **onnx_input_dict)
                torch.onnx.export(
                    self.decoder,
                    (tokens, audio_features,),  # Pass the actual input data
                    model_name + ".onnx",
                    verbose=True,
                    input_names=onnx_input_names,  # Provide input names
                    output_names=onnx_output_names,  # Provide output names
                    opset_version=15,  # ONNX opset version to use
                )
            logits = self.decoder(tokens, audio_features)
        self.call_logits_decoder += 1
        return logits
        # return self.decoder(tokens, audio_features)

    def forward(
        self, mel: torch.Tensor, tokens: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        print("{:=^80}".format(" model.forward "))
        return self.decoder(tokens, self.encoder(mel))

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def is_multilingual(self):
        return self.dims.n_vocab == 51865

    def install_kv_cache_hooks(self, cache: Optional[dict] = None):
        """
        The `MultiHeadAttention` module optionally accepts `kv_cache` which stores the key and value
        tensors calculated for the previous positions. This method returns a dictionary that stores
        all caches, and the necessary hooks for the key and value projection modules that save the
        intermediate tensors to be reused during later calculations.

        Returns
        -------
        cache : Dict[nn.Module, torch.Tensor]
            A dictionary object mapping the key/value projection modules to its cache
        hooks : List[RemovableHandle]
            List of PyTorch RemovableHandle objects to stop the hooks to be called
        """
        cache = {**cache} if cache is not None else {}
        hooks = []
        c_num = []

        def save_to_cache(module, _, output):
            if module not in cache or output.shape[1] > self.dims.n_text_ctx:
                # save as-is, for the first token or cross attention
                cache[module] = output
            else:
                cache[module] = torch.cat([cache[module], output], dim=1).detach()
            return cache[module]

        def install_hooks(layer: nn.Module):
            if isinstance(layer, MultiHeadAttention):
                # import pdb; pdb.set_trace()
                hooks.append(layer.key.register_forward_hook(save_to_cache))
                hooks.append(layer.value.register_forward_hook(save_to_cache))

        def fn(module):
            if isinstance(module, MultiHeadAttention):
                c_num.append(1)
        self.decoder.apply(fn)
        print(f"decoder MultiHeadAttention num: {len(c_num)}") # 12

        self.decoder.apply(install_hooks)
        return cache, hooks

    detect_language = detect_language_function
    transcribe = transcribe_function
    decode = decode_function
