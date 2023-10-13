import os
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical

from .audio import CHUNK_LENGTH
from .tokenizer import Tokenizer, get_tokenizer
from .utils import compression_ratio
from .untool import Tool, make_np2c, data_type, data_type_map

from tpu_perf.infer import SGInfer, nptype

if TYPE_CHECKING:
    from .model import Whisper

@torch.no_grad()
def detect_language(
    model: "Whisper", mel: Tensor, tokenizer: Tokenizer = None
) -> Tuple[Tensor, List[dict]]:
    """
    Detect the spoken language in the audio, and return them as list of strings, along with the ids
    of the most probable language tokens and the probability distribution over all language tokens.
    This is performed outside the main decode loop in order to not interfere with kv-caching.

    Returns
    -------
    language_tokens : Tensor, shape = (n_audio,)
        ids of the most probable language tokens, which appears after the startoftranscript token.
    language_probs : List[Dict[str, float]], length = n_audio
        list of dictionaries containing the probability distribution over all languages.
    """
    if tokenizer is None:
        tokenizer = get_tokenizer(model.is_multilingual)
    if (
        tokenizer.language is None
        or tokenizer.language_token not in tokenizer.sot_sequence
    ):
        raise ValueError(
            "This model doesn't have language tokens so it can't perform lang id"
        )

    single = mel.ndim == 2
    if single:
        mel = mel.unsqueeze(0)

    start_time = time.time()
    # skip encoder forward pass if already-encoded audio features were given
    if mel.shape[-2:] != (model.dims.n_audio_ctx, model.dims.n_audio_state):
        # transform type from encoder inputs
        encoder_info = model.tool.model_info(model.encoder_handle)
        mel_input_dtype = data_type_map[encoder_info['input_dtypes'][0]]
        mel = mel.numpy().astype(mel_input_dtype)
        mel = mel if mel.flags.c_contiguous else np.ascontiguousarray(mel)

        # combine numpy addr with C addr & copy data from host to device for input data
        model.tool.copy_data_from_numpy(model.tool.get_input_tensor(model.runtime1, 0), make_np2c(mel), data_type[mel_input_dtype])
        model.tool.force_host_to_device(model.tool.get_input_tensor(model.runtime1, 0), model.handle)

        # combine numpy addr with C addr for output data need cpu infernece
        mel_out = np.empty(encoder_info[0]['output_shapes'][0], dtype=data_type_map[encoder_info['output_dtypes'][0]])
        model.tool.copy_data_from_numpy(model.tool.get_output_tensor(model.runtime1, 0), make_np2c(mel_out), encoder_info['output_dtypes'][0])

        # inference encoder on tpu
        model.tool.inference(model.runtime1)

        # copy data from device to host for output data
        model.tool.copy_output_data_to_host(model.runtime1)
        mel_out = torch.from_numpy(mel_out)

        model.time += time.time() - start_time
        model.call_encoder += 1
        # print(f"detect_language encoder time: {time.time() - start_time}")

    # forward pass using a single token, startoftranscript
    n_audio = mel_out.shape[0]
    x = torch.tensor([[tokenizer.sot]] * n_audio)  # [n_audio, 1]
    start_time = time.time()
    logits = model.logits(x, mel_out)[:, 0].float()
    # print(f"logits time: {time.time() - start_time}")

    # collect detected languages; suppress all non-language tokens
    mask = torch.ones(logits.shape[-1], dtype=torch.bool)
    mask[list(tokenizer.all_language_tokens)] = False
    logits[:, mask] = -np.inf
    language_tokens = logits.argmax(dim=-1)
    language_token_probs = logits.softmax(dim=-1).cpu()
    language_probs = [
        {
            c: language_token_probs[i, j].item()
            for j, c in zip(tokenizer.all_language_tokens, tokenizer.all_language_codes)
        }
        for i in range(n_audio)
    ]

    if single:
        language_tokens = language_tokens[0]
        language_probs = language_probs[0]

    return language_tokens, language_probs


@dataclass(frozen=True)
class DecodingOptions:
    # whether to perform X->X "transcribe" or X->English "translate"
    task: str = "transcribe"

    # language that the audio is in; uses detected language if None
    language: Optional[str] = None

    # sampling-related options
    temperature: float = 0.0
    sample_len: Optional[int] = None  # maximum number of tokens to sample
    best_of: Optional[int] = None  # number of independent sample trajectories, if t > 0
    beam_size: Optional[int] = None  # number of beams in beam search, if t == 0
    patience: Optional[float] = None  # patience in beam search (arxiv:2204.05424)

    # "alpha" in Google NMT, or None for length norm, when ranking generations
    # to select which to return among the beams or best-of-N samples
    length_penalty: Optional[float] = None

    # text or tokens to feed as the prompt or the prefix; for more info:
    # https://github.com/openai/whisper/discussions/117#discussioncomment-3727051
    prompt: Optional[Union[str, List[int]]] = None  # for the previous context
    prefix: Optional[Union[str, List[int]]] = None  # to prefix the current context

    # list of tokens ids (or comma-separated token ids) to suppress
    # "-1" will suppress a set of symbols as defined in `tokenizer.non_speech_tokens()`
    suppress_tokens: Optional[Union[str, Iterable[int]]] = "-1"
    suppress_blank: bool = True  # this will suppress blank outputs

    # timestamp sampling options
    without_timestamps: bool = False  # use <|notimestamps|> to sample text tokens only
    max_initial_timestamp: Optional[float] = 1.0

    # implementation details
    padding_size: int = 448 # max pre-allocation of key-value cache

@dataclass(frozen=True)
class DecodingResult:
    audio_features: Tensor
    language: str
    language_probs: Optional[Dict[str, float]] = None
    tokens: List[int] = field(default_factory=list)
    text: str = ""
    avg_logprob: float = np.nan
    no_speech_prob: float = np.nan
    temperature: float = np.nan
    compression_ratio: float = np.nan


class Inference:
    def logits(self, tokens: Tensor, audio_features: Tensor) -> Tensor:
        """Perform a forward pass on the decoder and return per-token logits"""
        raise NotImplementedError

    def rearrange_kv_cache(
            self, 
            source_indices, 
            self_attention_kcache: Tensor, 
            self_attention_vcache: Tensor, 
            cross_attention_kcache: Tensor, 
            cross_attention_vcache: Tensor
        ) -> None:
        """Update the key-value cache according to the updated beams"""
        raise NotImplementedError

    def cleanup_caching(self) -> None:
        """Clean up any resources or hooks after decoding is finished"""
        pass


class PyTorchInference(Inference):
    def __init__(self, model: "Whisper", initial_token_length: int):
        self.model: "Whisper" = model

    def rearrange_kv_cache(
            self, 
            source_indices, 
            self_attention_kcache: Tuple[Tensor] = None, 
            self_attention_vcache: Tuple[Tensor] = None, 
        ):
        if source_indices != list(range(len(source_indices))):
            start_time = time.time()
            indices = np.array(source_indices, dtype=np.int32)
            indices = indices if indices.flags.contiguous else indices.copy()
            # combine numpy addr with C addr & copy data from host to device for input data
            self.model.tool.copy_data_from_numpy(self.model.tool.get_input_tensor(self.model.kvcache_rearrange_runtime[0], 1), make_np2c(indices), 6)
            self.model.tool.force_host_to_device(self.model.tool.get_input_tensor(self.model.kvcache_rearrange_runtime[0], 1), self.model.handle)
            for i in range(2 * self.model.dims.n_text_layer):
                # inference decoder_main on tpu
                self.model.tool.inference(self.model.kvcache_rearrange_runtime[i])
            self.model.time += time.time() - start_time
            self.model.call_kvcache_rearrange += 2 * self.model.dims.n_text_layer
            return

class SequenceRanker:
    def rank(
        self, tokens: List[List[Tensor]], sum_logprobs: List[List[float]]
    ) -> List[int]:
        """
        Given a list of groups of samples and their cumulative log probabilities,
        return the indices of the samples in each group to select as the final result
        """
        raise NotImplementedError


class MaximumLikelihoodRanker(SequenceRanker):
    """
    Select the sample with the highest log probabilities, penalized using either
    a simple length normalization or Google NMT paper's length penalty
    """

    def __init__(self, length_penalty: Optional[float]):
        self.length_penalty = length_penalty

    def rank(self, tokens: List[List[Tensor]], sum_logprobs: List[List[float]]):
        def scores(logprobs, lengths):
            result = []
            for logprob, length in zip(logprobs, lengths):
                if self.length_penalty is None:
                    penalty = length
                else:
                    # from the Google NMT paper
                    penalty = ((5 + length) / 6) ** self.length_penalty
                result.append(logprob / penalty)
            return result

        # get the sequence with the highest score
        lengths = [[len(t) for t in s] for s in tokens]
        return [np.argmax(scores(p, l)) for p, l in zip(sum_logprobs, lengths)]


class TokenDecoder:
    def reset(self):
        """Initialize any stateful variables for decoding a new sequence"""

    def update(
        self, 
        tokens: Tensor, 
        logits: Tensor, 
        sum_logprobs: Tensor, 
    ) -> Tuple[Tensor, bool]:
        """Specify how to select the next token, based on the current trace and logits

        Parameters
        ----------
        tokens : Tensor, shape = (n_batch, current_sequence_length)
            all tokens in the context so far, including the prefix and sot_sequence tokens

        logits : Tensor, shape = (n_batch, vocab_size)
            per-token logits of the probability distribution at the current step

        sum_logprobs : Tensor, shape = (n_batch)
            cumulative log probabilities for each sequence

        Returns
        -------
        tokens : Tensor, shape = (n_batch, current_sequence_length + 1)
            the tokens, appended with the selected next token

        completed : bool
            True if all sequences has reached the end of text

        """
        raise NotImplementedError

    def finalize(
        self, tokens: Tensor, sum_logprobs: Tensor
    ) -> Tuple[Sequence[Sequence[Tensor]], List[List[float]]]:
        """Finalize search and return the final candidate sequences

        Parameters
        ----------
        tokens : Tensor, shape = (n_audio, n_group, current_sequence_length)
            all tokens in the context so far, including the prefix and sot_sequence

        sum_logprobs : Tensor, shape = (n_audio, n_group)
            cumulative log probabilities for each sequence

        Returns
        -------
        tokens : Sequence[Sequence[Tensor]], length = n_audio
            sequence of Tensors containing candidate token sequences, for each audio input

        sum_logprobs : List[List[float]], length = n_audio
            sequence of cumulative log probabilities corresponding to the above

        """
        raise NotImplementedError


class GreedyDecoder(TokenDecoder):
    def __init__(self, temperature: float, eot: int):
        self.temperature = temperature
        self.eot = eot

    def update(
        self, 
        tokens: Tensor, 
        logits: Tensor, 
        sum_logprobs: Tensor, 
        self_attention_kcache: Tensor = None, 
        self_attention_vcache: Tensor = None, 
    ) -> Tuple[Tensor, bool]:
        if self.temperature == 0:
            next_tokens = logits.argmax(dim=-1)
        else:
            next_tokens = Categorical(logits=logits / self.temperature).sample()

        logprobs = F.log_softmax(logits.float(), dim=-1)
        current_logprobs = logprobs[torch.arange(logprobs.shape[0]), next_tokens]
        sum_logprobs += current_logprobs * (tokens[:, -1] != self.eot)

        next_tokens[tokens[:, -1] == self.eot] = self.eot
        tokens = torch.cat([tokens, next_tokens[:, None]], dim=-1)

        completed = (tokens[:, -1] == self.eot).all()
        return tokens, completed

    def finalize(self, tokens: Tensor, sum_logprobs: Tensor):
        # make sure each sequence has at least one EOT token at the end
        tokens = F.pad(tokens, (0, 1), value=self.eot)
        return tokens, sum_logprobs.tolist()


class BeamSearchDecoder(TokenDecoder):
    def __init__(
        self,
        beam_size: int,
        eot: int,
        inference: Inference,
        patience: Optional[float] = None,
    ):
        self.beam_size = beam_size
        self.eot = eot
        self.inference = inference
        self.patience = patience or 1.0
        self.max_candidates: int = round(beam_size * self.patience)
        self.finished_sequences = None

        assert (
            self.max_candidates > 0
        ), f"Invalid beam size ({beam_size}) or patience ({patience})"

    def reset(self):
        self.finished_sequences = None

    def update(
        self, 
        tokens: Tensor, 
        logits: Tensor, 
        sum_logprobs: Tensor, 
        self_attention_kcache: Tensor = None, 
        self_attention_vcache: Tensor = None, 
    ) -> Tuple[Tensor, bool]:
        if tokens.shape[0] % self.beam_size != 0:
            raise ValueError(f"{tokens.shape}[0] % {self.beam_size} != 0")

        n_audio = tokens.shape[0] // self.beam_size
        if self.finished_sequences is None:  # for the first update
            self.finished_sequences = [{} for _ in range(n_audio)]

        logprobs = F.log_softmax(logits.float(), dim=-1)
        next_tokens, source_indices, finished_sequences = [], [], []
        for i in range(n_audio):
            scores, sources, finished = {}, {}, {}

            # STEP 1: calculate the cumulative log probabilities for possible candidates
            for j in range(self.beam_size):
                idx = i * self.beam_size + j
                prefix = tokens[idx].tolist()
                for logprob, token in zip(*logprobs[idx].topk(self.beam_size + 1)):
                    new_logprob = (sum_logprobs[idx] + logprob).item()
                    sequence = tuple(prefix + [token.item()])
                    scores[sequence] = new_logprob
                    sources[sequence] = idx

            # STEP 2: rank the candidates and keep the top beam_size sequences for each audio
            saved = 0
            for sequence in sorted(scores, key=scores.get, reverse=True):
                if sequence[-1] == self.eot:
                    finished[sequence] = scores[sequence]
                else:
                    sum_logprobs[len(next_tokens)] = scores[sequence]
                    next_tokens.append(sequence)
                    source_indices.append(sources[sequence])

                    saved += 1
                    if saved == self.beam_size:
                        break

            finished_sequences.append(finished)

        tokens = torch.tensor(next_tokens, device=tokens.device)
        # import pdb; pdb.set_trace()

        if self_attention_kcache:
            self.inference.rearrange_kv_cache(
                source_indices, 
                self_attention_kcache, 
                self_attention_vcache, 
            )
        else:
            self.inference.rearrange_kv_cache(source_indices)

        # add newly finished sequences to self.finished_sequences
        assert len(self.finished_sequences) == len(finished_sequences)
        for previously_finished, newly_finished in zip(
            self.finished_sequences, finished_sequences
        ):
            for seq in sorted(newly_finished, key=newly_finished.get, reverse=True):
                if len(previously_finished) >= self.max_candidates:
                    break  # the candidate list is full
                previously_finished[seq] = newly_finished[seq]

        # mark as completed if all audio has enough number of samples
        completed = all(
            len(sequences) >= self.max_candidates
            for sequences in self.finished_sequences
        )
        return tokens, completed

    def finalize(self, preceding_tokens: Tensor, sum_logprobs: Tensor):
        # collect all finished sequences, including patience, and add unfinished ones if not enough
        sum_logprobs = sum_logprobs.cpu()
        for i, sequences in enumerate(self.finished_sequences):
            if (
                len(sequences) < self.beam_size
            ):  # when not enough sequences are finished
                for j in list(np.argsort(sum_logprobs[i]))[::-1]:
                    sequence = preceding_tokens[i, j].tolist() + [self.eot]
                    sequences[tuple(sequence)] = sum_logprobs[i][j].item()
                    if len(sequences) >= self.beam_size:
                        break

        tokens: List[List[Tensor]] = [
            [torch.tensor(seq) for seq in sequences.keys()]
            for sequences in self.finished_sequences
        ]
        sum_logprobs: List[List[float]] = [
            list(sequences.values()) for sequences in self.finished_sequences
        ]
        return tokens, sum_logprobs


class LogitFilter:
    def apply(self, logits: Tensor, tokens: Tensor) -> None:
        """Apply any filtering or masking to logits in-place

        Parameters
        ----------
        logits : Tensor, shape = (n_batch, vocab_size)
            per-token logits of the probability distribution at the current step

        tokens : Tensor, shape = (n_batch, current_sequence_length)
            all tokens in the context so far, including the prefix and sot_sequence tokens

        """
        raise NotImplementedError


class SuppressBlank(LogitFilter):
    def __init__(self, tokenizer: Tokenizer, sample_begin: int):
        self.tokenizer = tokenizer
        self.sample_begin = sample_begin

    def apply(self, logits: Tensor, tokens: Tensor):
        if tokens.shape[1] == self.sample_begin:
            logits[:, self.tokenizer.encode(" ") + [self.tokenizer.eot]] = -np.inf


class SuppressTokens(LogitFilter):
    def __init__(self, suppress_tokens: Sequence[int]):
        self.suppress_tokens = list(suppress_tokens)

    def apply(self, logits: Tensor, tokens: Tensor):
        logits[:, self.suppress_tokens] = -np.inf


class ApplyTimestampRules(LogitFilter):
    def __init__(
        self,
        tokenizer: Tokenizer,
        sample_begin: int,
        max_initial_timestamp_index: Optional[int],
    ):
        self.tokenizer = tokenizer
        self.sample_begin = sample_begin
        self.max_initial_timestamp_index = max_initial_timestamp_index

    def apply(self, logits: Tensor, tokens: Tensor):
        # suppress <|notimestamps|> which is handled by without_timestamps
        if self.tokenizer.no_timestamps is not None:
            logits[:, self.tokenizer.no_timestamps] = -np.inf

        # timestamps have to appear in pairs, except directly before EOT; mask logits accordingly
        for k in range(tokens.shape[0]):
            sampled_tokens = tokens[k, self.sample_begin :]
            seq = [t for t in sampled_tokens.tolist()]
            last_was_timestamp = (
                len(seq) >= 1 and seq[-1] >= self.tokenizer.timestamp_begin
            )
            penultimate_was_timestamp = (
                len(seq) < 2 or seq[-2] >= self.tokenizer.timestamp_begin
            )

            if last_was_timestamp:
                if penultimate_was_timestamp:  # has to be non-timestamp
                    logits[k, self.tokenizer.timestamp_begin :] = -np.inf
                else:  # cannot be normal text tokens
                    logits[k, : self.tokenizer.eot] = -np.inf

            timestamps = sampled_tokens[
                sampled_tokens.ge(self.tokenizer.timestamp_begin)
            ]
            if timestamps.numel() > 0:
                # timestamps shouldn't decrease; forbid timestamp tokens smaller than the last
                # also force each segment to have a nonzero length, to prevent infinite looping
                if last_was_timestamp and not penultimate_was_timestamp:
                    timestamp_last = timestamps[-1]
                else:
                    timestamp_last = timestamps[-1] + 1
                logits[k, self.tokenizer.timestamp_begin : timestamp_last] = -np.inf

        if tokens.shape[1] == self.sample_begin:
            # suppress generating non-timestamp tokens at the beginning
            logits[:, : self.tokenizer.timestamp_begin] = -np.inf

            # apply the `max_initial_timestamp` option
            if self.max_initial_timestamp_index is not None:
                last_allowed = (
                    self.tokenizer.timestamp_begin + self.max_initial_timestamp_index
                )
                logits[:, last_allowed + 1 :] = -np.inf

        # if sum of probability over timestamps is above any other token, sample timestamp
        logprobs = F.log_softmax(logits.float(), dim=-1)
        for k in range(tokens.shape[0]):
            timestamp_logprob = logprobs[k, self.tokenizer.timestamp_begin :].logsumexp(
                dim=-1
            )
            max_text_token_logprob = logprobs[k, : self.tokenizer.timestamp_begin].max()
            if timestamp_logprob > max_text_token_logprob:
                logits[k, : self.tokenizer.timestamp_begin] = -np.inf


class DecodingTask:
    inference: Inference
    sequence_ranker: SequenceRanker
    decoder: TokenDecoder
    logit_filters: List[LogitFilter]

    def __init__(self, model: "Whisper", options: DecodingOptions):
        self.model = model
        
        language = options.language or "en"
        tokenizer = get_tokenizer(
            model.is_multilingual, language=language, task=options.task
        )
        self.tokenizer: Tokenizer = tokenizer
        self.options: DecodingOptions = self._verify_options(options)

        self.n_group: int = options.beam_size or options.best_of or 1
        self.n_ctx: int = model.dims.n_text_ctx
        self.sample_len: int = options.sample_len or model.dims.n_text_ctx // 2
        self.n_text_head = self.model.dims.n_text_head
        self.n_text_layer = self.model.dims.n_text_layer
        self.padding_size = options.padding_size

        self.sot_sequence: Tuple[int] = tokenizer.sot_sequence
        if self.options.without_timestamps:
            self.sot_sequence = tokenizer.sot_sequence_including_notimestamps

        self.initial_tokens: Tuple[int] = self._get_initial_tokens()
        self.sample_begin: int = len(self.initial_tokens)
        self.sot_index: int = self.initial_tokens.index(tokenizer.sot)

        # inference: implements the forward pass through the decoder, including kv caching
        self.inference = PyTorchInference(model, len(self.initial_tokens))

        # sequence ranker: implements how to rank a group of sampled sequences
        self.sequence_ranker = MaximumLikelihoodRanker(options.length_penalty)

        # decoder: implements how to select the next tokens, given the autoregressive distribution
        if options.beam_size is not None:
            self.decoder = BeamSearchDecoder(
                options.beam_size, tokenizer.eot, self.inference, options.patience
            )
        else:
            self.decoder = GreedyDecoder(options.temperature, tokenizer.eot)

        # logit filters: applies various rules to suppress or penalize certain tokens
        self.logit_filters = []
        if self.options.suppress_blank:
            self.logit_filters.append(SuppressBlank(self.tokenizer, self.sample_begin))
        if self.options.suppress_tokens:
            self.logit_filters.append(SuppressTokens(self._get_suppress_tokens()))
        if not options.without_timestamps:
            precision = CHUNK_LENGTH / model.dims.n_audio_ctx  # usually 0.02 seconds
            max_initial_timestamp_index = None
            if options.max_initial_timestamp:
                max_initial_timestamp_index = round(
                    self.options.max_initial_timestamp / precision
                )
            self.logit_filters.append(
                ApplyTimestampRules(
                    tokenizer, self.sample_begin, max_initial_timestamp_index
                )
            )

    def _verify_options(self, options: DecodingOptions) -> DecodingOptions:
        if options.beam_size is not None and options.best_of is not None:
            raise ValueError("beam_size and best_of can't be given together")
        if options.temperature == 0:
            if options.best_of is not None:
                raise ValueError("best_of with greedy sampling (T=0) is not compatible")
        if options.patience is not None and options.beam_size is None:
            raise ValueError("patience requires beam_size to be given")
        if options.length_penalty is not None and not (
            0 <= options.length_penalty <= 1
        ):
            raise ValueError("length_penalty (alpha) should be a value between 0 and 1")

        return options

    def _get_initial_tokens(self) -> Tuple[int]:
        tokens = list(self.sot_sequence)

        if prefix := self.options.prefix:
            prefix_tokens = (
                self.tokenizer.encode(" " + prefix.strip())
                if isinstance(prefix, str)
                else prefix
            )
            if self.sample_len is not None:
                max_prefix_len = self.n_ctx // 2 - self.sample_len
                prefix_tokens = prefix_tokens[-max_prefix_len:]
            tokens = tokens + prefix_tokens

        if prompt := self.options.prompt:
            prompt_tokens = (
                self.tokenizer.encode(" " + prompt.strip())
                if isinstance(prompt, str)
                else prompt
            )
            tokens = (
                [self.tokenizer.sot_prev]
                + prompt_tokens[-(self.n_ctx // 2 - 1) :]
                + tokens
            )

        return tuple(tokens)

    def _get_suppress_tokens(self) -> Tuple[int]:
        suppress_tokens = self.options.suppress_tokens

        if isinstance(suppress_tokens, str):
            suppress_tokens = [int(t) for t in suppress_tokens.split(",")]

        if -1 in suppress_tokens:
            suppress_tokens = [t for t in suppress_tokens if t >= 0]
            suppress_tokens.extend(self.tokenizer.non_speech_tokens)
        elif suppress_tokens is None or len(suppress_tokens) == 0:
            suppress_tokens = []  # interpret empty string as an empty list
        else:
            assert isinstance(suppress_tokens, list), "suppress_tokens must be a list"

        suppress_tokens.extend(
            [
                self.tokenizer.transcribe,
                self.tokenizer.translate,
                self.tokenizer.sot,
                self.tokenizer.sot_prev,
                self.tokenizer.sot_lm,
            ]
        )
        if self.tokenizer.no_speech is not None:
            # no-speech probability is collected separately
            suppress_tokens.append(self.tokenizer.no_speech)

        return tuple(sorted(set(suppress_tokens)))

    def _get_audio_features(self, mel: Tensor):
        mel = mel.half()

        if mel.shape[-2:] == (
            self.model.dims.n_audio_ctx,
            self.model.dims.n_audio_state,
        ):
            # encoded audio features are given; skip audio encoding
            audio_features = mel
        else:
            start_time = time.time()
            # type transform for decoder_main inputs
            encoder_info = self.model.tool.model_info(self.model.encoder_handle)
            mel_input_dtype = data_type_map[encoder_info['input_dtypes'][0]]
            mel = mel.numpy().astype(mel_input_dtype)

            mel = mel if mel.flags.c_contiguous else np.ascontiguousarray(mel)

            # combine numpy addr with C addr & copy data from host to device for input data
            self.model.tool.copy_data_from_numpy(self.model.tool.get_input_tensor(self.model.runtime1, 0), make_np2c(mel), data_type[mel_input_dtype])
            self.model.tool.force_host_to_device(self.model.tool.get_input_tensor(self.model.runtime1, 0), self.model.handle)

            # combine numpy addr with C addr for output data need cpu infernece
            mel_out = np.empty(encoder_info[0]['output_shapes'][0], dtype=data_type_map[encoder_info['output_dtypes'][0]])
            self.model.tool.copy_data_from_numpy(self.model.tool.get_output_tensor(self.model.runtime1, 0), make_np2c(mel_out), encoder_info['output_dtypes'][0])

            # inference encoder on tpu
            self.model.tool.inference(self.model.runtime1)

            # copy data from device to host for output data
            self.model.tool.copy_output_data_to_host(self.model.runtime1)
            audio_features = torch.from_numpy(mel_out)

            self.model.call_encoder +=1
            self.model.time += time.time() - start_time
            # print(f"_get_audio_features encoder time: {time.time() - start_time}")

        return audio_features

    def _detect_language(self, audio_features: Tensor, tokens: Tensor):
        languages = [self.options.language] * audio_features.shape[0]
        lang_probs = None

        if self.options.language is None or self.options.task == "lang_id":
            lang_tokens, lang_probs = self.model.detect_language(
                audio_features, self.tokenizer
            )
            languages = [max(probs, key=probs.get) for probs in lang_probs]
            if self.options.language is None:
                tokens[:, self.sot_index + 1] = lang_tokens  # write language tokens

        return languages, lang_probs

    def _main_loop_untool(self, audio_features: Tensor, tokens: Tensor):
        # print("{:=^100}".format(f" start main_loop {self.model.main_loop_cnt} "))
        self.model.main_loop_cnt += 1
        n_batch = tokens.shape[0]
        sum_logprobs: Tensor = torch.zeros(n_batch, device=audio_features.device)
        no_speech_probs = [np.nan] * n_batch
        initial_tokens_length = len(self.initial_tokens)
        padding_num = self.padding_size
    
        attention_mask_firstly = torch.empty(padding_num, padding_num).fill_(-10000).triu_(1)
        attention_mask_with_kvcache_max = torch.empty(448, 448).fill_(-10000).triu_(1)
        attention_mask_with_kvcache = attention_mask_with_kvcache_max[-padding_num:, -padding_num:]
        loop_start_time = time.time()
        tool = self.model.tool

        try:
            for i in range(self.sample_len):
                if i == 0:
                    tokens_input = F.pad(tokens, (padding_num - tokens.shape[-1], 0, 0, 0), value=0)
                    positional_embedding_input = F.pad(self.model.positional_embedding[:i+initial_tokens_length], (0, 0, padding_num - initial_tokens_length - i, 0), value=0)
                    mask = F.pad(attention_mask_firstly[:tokens.shape[-1], :tokens.shape[-1]], (padding_num - tokens.shape[-1], 0, 0, 0), value=-10000)
                    mask = F.pad(mask, (0, 0, padding_num - tokens.shape[-1], 0), value=0)
                    mask = mask.reshape(1, 1, *mask.shape).repeat(n_batch, self.n_text_head, 1, 1).permute(0, 2, 1, 3).contiguous()
                else:
                    tokens_input = tokens[:, -1:]
                    offset = i + initial_tokens_length - 1
                    positional_embedding_input = self.model.positional_embedding[offset:offset+1]
                    mask = attention_mask_with_kvcache[offset:offset+1].flip(1)
                    mask = mask.reshape(1, 1, *mask.shape).repeat(n_batch, self.n_text_head, 1, 1).permute(0, 2, 1, 3).contiguous()
                # import pdb; pdb.set_trace()

                if i == 0:
                    start_time = time.time()
                    # type transform for decoder_main inputs
                    decoder_main_info = tool.model_info(self.model.decoder_main_handle)
                    tokens_input = tokens_input.numpy().astype(np.int32)
                    audio_features_dtype = data_type_map[decoder_main_info['input_dtypes'][1]]
                    audio_features = audio_features.numpy().astype(audio_features_dtype)
                    positional_embedding_input_dtype = data_type_map[decoder_main_info['input_dtypes'][2]]
                    positional_embedding_input = positional_embedding_input.numpy().astype(positional_embedding_input_dtype)
                    mask_dtype = data_type_map[decoder_main_info['input_dtypes'][3]]
                    mask = mask.numpy().astype(mask_dtype)

                    tokens_input = tokens_input if tokens_input.flags.c_contiguous else np.ascontiguousarray(tokens_input)
                    audio_features = audio_features if audio_features.flags.c_contiguous else np.ascontiguousarray(audio_features)
                    positional_embedding_input = positional_embedding_input if positional_embedding_input.flags.c_contiguous else np.ascontiguousarray(positional_embedding_input)
                    mask = mask if mask.flags.c_contiguous else np.ascontiguousarray(mask)

                    # combine numpy addr with C addr & copy data from host to device for input data
                    tool.copy_data_from_numpy(tool.get_input_tensor(self.model.runtime3, 0), make_np2c(tokens_input), data_type[np.int32])
                    tool.copy_data_from_numpy(tool.get_input_tensor(self.model.runtime3, 1), make_np2c(audio_features), data_type[audio_features_dtype])
                    tool.copy_data_from_numpy(tool.get_input_tensor(self.model.runtime3, 2), make_np2c(positional_embedding_input), data_type[positional_embedding_input_dtype])
                    tool.copy_data_from_numpy(tool.get_input_tensor(self.model.runtime3, 3), make_np2c(mask), data_type[mask_dtype])
                    tool.force_host_to_device(tool.get_input_tensor(self.model.runtime3, 0), self.model.handle)
                    tool.force_host_to_device(tool.get_input_tensor(self.model.runtime3, 1), self.model.handle)
                    tool.force_host_to_device(tool.get_input_tensor(self.model.runtime3, 2), self.model.handle)
                    tool.force_host_to_device(tool.get_input_tensor(self.model.runtime3, 3), self.model.handle)

                    # combine numpy addr with C addr for output data need cpu infernece
                    x = np.empty(decoder_main_info[0]['output_shapes'][0], dtype=data_type_map[decoder_main_info['output_dtypes'][0]])
                    tool.copy_data_from_numpy(tool.get_output_tensor(self.model.runtime3, 0), make_np2c(x), decoder_main_info['output_dtypes'][0])

                    # inference decoder_main on tpu
                    tool.inference(self.model.runtime3)

                    # copy data from device to host for output data
                    tool.device_to_host(tool.get_output_tensor(self.model.runtime3, 0), self.model.handle)

                    # get input data for decoder_post
                    # this process is dynamic
                    x_sot = x[:, padding_num - initial_tokens_length + self.sot_index:padding_num - initial_tokens_length + self.sot_index + 1].copy()
                    x_last = x[:, -1:].copy()

                    # combine numpy addr with C addr & copy data from host to device for input data
                    decoder_post_info = tool.model_info(self.model.decoder_post_handle)
                    tool.copy_data_from_numpy(tool.get_input_tensor(self.model.runtime4, 0), make_np2c(x_sot), data_type[x_sot.dtype])
                    tool.copy_data_from_numpy(tool.get_input_tensor(self.model.runtime4, 1), make_np2c(x_last), data_type[x_last.dtype])
                    tool.force_host_to_device(tool.get_input_tensor(self.model.runtime4, 0), self.model.handle)
                    tool.force_host_to_device(tool.get_input_tensor(self.model.runtime4, 1), self.model.handle)
                    
                    # combine numpy addr with C addr for output data need cpu infernece
                    logits = np.empty(decoder_post_info[0]['output_shapes'][0], dtype=data_type_map[decoder_post_info['output_dtypes'][0]])
                    no_speech_probs = np.empty(decoder_post_info[0]['output_shapes'][1], dtype=data_type_map[decoder_post_info['output_dtypes'][1]])
                    tool.copy_data_from_numpy(tool.get_output_tensor(self.model.runtime4, 0), make_np2c(logits), decoder_post_info['output_dtypes'][0])
                    tool.copy_data_from_numpy(tool.get_output_tensor(self.model.runtime4, 1), make_np2c(no_speech_probs), decoder_post_info['output_dtypes'][1])

                    # inference decoder_post on tpu
                    tool.inference(self.model.runtime4)

                    # copy data from device to host for output data
                    tool.copy_output_data_to_host(self.model.runtime4)

                    logits = torch.from_numpy(logits)
                    no_speech_probs = no_speech_probs.tolist()
                    
                    self.model.call_decoder_firstly += 1
                    self.model.time += time.time() - start_time

                else:
                    start_time = time.time()
                    # type transform for decoder_loop inputs
                    decoder_loop_info = tool.model_info(self.model.decoder_loop_handle)

                    tokens_input = tokens_input.numpy().astype(np.int32)
                    positional_embedding_input_dtype = data_type_map[decoder_loop_info['input_dtypes'][1]]
                    positional_embedding_input = positional_embedding_input.numpy().astype(positional_embedding_input_dtype)
                    mask_dtype = data_type_map[decoder_loop_info['input_dtypes'][2]]
                    mask = mask.numpy().astype(mask_dtype)

                    tokens_input = tokens_input if tokens_input.flags.contiguous else np.ascontiguousarray(tokens_input)
                    positional_embedding_input = positional_embedding_input if positional_embedding_input.flags.contiguous else np.ascontiguousarray(positional_embedding_input)
                    mask = mask if mask.flags.contiguous else np.ascontiguousarray(mask)

                    # combine numpy addr with C addr & copy data from host to device for input data
                    tool.copy_data_from_numpy(tool.get_input_tensor(self.model.runtime5, 0), make_np2c(tokens_input), data_type[np.int32])
                    tool.copy_data_from_numpy(tool.get_input_tensor(self.model.runtime5, 1), make_np2c(positional_embedding_input), data_type[positional_embedding_input_dtype])
                    tool.copy_data_from_numpy(tool.get_input_tensor(self.model.runtime5, 2), make_np2c(mask), data_type[mask_dtype])
                    tool.force_host_to_device(tool.get_input_tensor(self.model.runtime5, 0), self.model.handle)
                    tool.force_host_to_device(tool.get_input_tensor(self.model.runtime5, 1), self.model.handle)
                    tool.force_host_to_device(tool.get_input_tensor(self.model.runtime5, 2), self.model.handle)

                    # combine numpy addr with C addr for output data need cpu infernece in first loop
                    if i == 1:
                        tool.copy_data_from_numpy(tool.get_output_tensor(self.model.runtime5, 0), make_np2c(logits.numpy()), decoder_loop_info['output_dtypes'][0])

                    # tool.malloc_device_address(self.model.runtime5)
                    # inference decoder_post on tpu
                    tool.inference(self.model.runtime5)

                    # copy data from device to host for output data
                    tool.device_to_host(tool.get_output_tensor(self.model.runtime5, 0), self.model.handle)

                    self.model.call_decoder_loop += 1
                    self.model.time += time.time() - start_time

                # apply the logit filters, e.g. for suppressing or applying penalty to
                for logit_filter in self.logit_filters:
                    logit_filter.apply(logits, tokens)

                # expand the tokens tensor with the selected next tokens
                tokens, completed = self.decoder.update(tokens, 
                                                        logits.float(), 
                                                        sum_logprobs, 
                                                    )

                if completed or tokens.shape[-1] > self.n_ctx:
                    break
        finally:
            pass
        # print(f'loop cost time: {time.time() - loop_start_time}')
        return tokens, sum_logprobs, no_speech_probs

    @torch.no_grad()
    def run(self, mel: Tensor) -> List[DecodingResult]:
        self.decoder.reset()
        tokenizer: Tokenizer = self.tokenizer
        n_audio: int = mel.shape[0]

        audio_features: Tensor = self._get_audio_features(mel)  # encoder forward pass
        tokens: Tensor = torch.tensor([self.initial_tokens]).repeat(n_audio, 1)

        # detect language if requested, overwriting the language token
        languages, language_probs = self._detect_language(audio_features, tokens) # encoder forward pass

        if self.options.task == "lang_id":
            return [
                DecodingResult(
                    audio_features=features, language=language, language_probs=probs
                )
                for features, language, probs in zip(
                    audio_features, languages, language_probs
                )
            ]

        # repeat text tensors by the group size, for beam search or best-of-n sampling
        tokens = tokens.repeat_interleave(self.n_group, dim=0).to(torch.int32)

        # call the main sampling loop
        tokens, sum_logprobs, no_speech_probs = self._main_loop_untool(audio_features, tokens) # decoder forward pass

        # reshape the tensors to have (n_audio, n_group) as the first two dimensions
        audio_features = audio_features[:: self.n_group]
        no_speech_probs = no_speech_probs[:: self.n_group]
        assert audio_features.shape[0] == len(no_speech_probs) == n_audio

        tokens = tokens.reshape(n_audio, self.n_group, -1)
        sum_logprobs = sum_logprobs.reshape(n_audio, self.n_group)

        # get the final candidates for each group, and slice between the first sampled token and EOT
        tokens, sum_logprobs = self.decoder.finalize(tokens, sum_logprobs)
        tokens: List[List[Tensor]] = [
            [t[self.sample_begin : (t == tokenizer.eot).nonzero()[0, 0]] for t in s]
            for s in tokens
        ]

        # select the top-ranked sample in each group
        selected = self.sequence_ranker.rank(tokens, sum_logprobs)
        tokens: List[List[int]] = [t[i].tolist() for i, t in zip(selected, tokens)]
        texts: List[str] = [tokenizer.decode(t).strip() for t in tokens]

        sum_logprobs: List[float] = [lp[i] for i, lp in zip(selected, sum_logprobs)]
        avg_logprobs: List[float] = [
            lp / (len(t) + 1) for t, lp in zip(tokens, sum_logprobs)
        ]

        fields = (
            texts,
            languages,
            tokens,
            audio_features,
            avg_logprobs,
            no_speech_probs,
        )
        if len(set(map(len, fields))) != 1:
            raise RuntimeError(f"inconsistent result lengths: {list(map(len, fields))}")

        return [
            DecodingResult(
                audio_features=features,
                language=language,
                tokens=tokens,
                text=text,
                avg_logprob=avg_logprob,
                no_speech_prob=no_speech_prob,
                temperature=self.options.temperature,
                compression_ratio=compression_ratio(text),
            )
            for text, language, tokens, features, avg_logprob, no_speech_prob in zip(
                *fields
            )
        ]


@torch.no_grad()
def decode(
    model: "Whisper",
    mel: Tensor,
    options: DecodingOptions = DecodingOptions(),
    **kwargs,
) -> Union[DecodingResult, List[DecodingResult]]:
    """
    Performs decoding of 30-second audio segment(s), provided as Mel spectrogram(s).

    Parameters
    ----------
    model: Whisper
        the Whisper model instance

    mel: torch.Tensor, shape = (80, 3000) or (*, 80, 3000)
        A tensor containing the Mel spectrogram(s)

    options: DecodingOptions
        A dataclass that contains all necessary options for decoding 30-second segments

    Returns
    -------
    result: Union[DecodingResult, List[DecodingResult]]
        The result(s) of decoding contained in `DecodingResult` dataclass instance(s)
    """
    # import pdb; pdb.set_trace()
    if single := mel.ndim == 2:
        mel = mel.unsqueeze(0)

    if kwargs:
        options = replace(options, **kwargs)

    result = DecodingTask(model, options).run(mel)

    return result[0] if single else result
