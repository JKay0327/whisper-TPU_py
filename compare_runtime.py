import os
import ctypes 
import numpy as np 
import time 
import torch

from tpu_perf.infer import SGInfer, nptype
from bmwhisper.untool import Tool, make_np2c, data_type, data_type_map

def compare():
    encoder_bmodel_path = f"all_quant_encoder_small_5beam_448pad_1684x_f16.bmodel"
    bmodel_dir = "./bmodel"
    encoder_bmodel_path = os.path.join(bmodel_dir, encoder_bmodel_path)
    assert os.path.exists(encoder_bmodel_path), f"{encoder_bmodel_path} not found"
    mel = np.random.randn(1, 80, 3000).astype(np.float16)
    print(mel)
    start_time = time.time()
    encoder_sg_infer = SGInfer(encoder_bmodel_path, 1, [0])
    encoder_sg_infer.put(mel)
    _, result, _ = encoder_sg_infer.get()
    output_sg = torch.from_numpy(result[0])
    print("{:=^80}".format(f" sg_infer "))
    print(output_sg)
    print("sg time: ", time.time() - start_time)

    print(mel)
    start_time = time.time()
    tool = Tool()
    handle = tool.bmhandle(0)
    bmrt = tool.bmrt(handle)
    encoder_handle = tool.create_model(encoder_bmodel_path.encode("utf-8"), bmrt)
    runtime = tool.create_un_runtime(handle)
    tool.set_bmodel_info(runtime, encoder_handle)
    tool.set_stage(runtime, 0)
    tool.init_all_tensors(runtime)
    tool.malloc_device_address(runtime)
    encoder_info = tool.model_info(encoder_handle)
    tool.copy_data_from_numpy(tool.get_input_tensor(runtime, 0), make_np2c(mel), data_type[np.float16])

    output_untool = np.empty(encoder_info[0]['output_shapes'][0], dtype=data_type_map[encoder_info['output_dtypes'][0]])
    print(output_untool)
    tool.copy_data_from_numpy(tool.get_output_tensor(runtime, 0), make_np2c(output_untool), encoder_info['output_dtypes'][0])
    tool.copy_input_data_to_device(runtime)
    tool.inference(runtime)
    tool.copy_output_data_to_host(runtime)
    tool.print_output_data(runtime)
    # print(output_untool)
    # tool.device_to_host(tool.get_output_tensor(runtime, 0))
    output_untool = torch.from_numpy(output_untool)
    print("{:=^80}".format(f" untool_infer "))
    print(output_untool)
    print("untool time: ", time.time() - start_time)

    tool.destroy_un_runtime(runtime)
    tool.destroy_model(encoder_handle)
    tool.free_bmrt(bmrt)
    tool.free_bmhandle(handle)

    #########################################################################################
    decoder_bmodel_path = f"all_quant_decoder_main_with_kvcache_small_5beam_448pad_1684x_f16.bmodel"
    bmodel_dir = "./bmodel"
    decoder_bmodel_path = os.path.join(bmodel_dir, decoder_bmodel_path)
    inputs = np.load("./tmp/onnx_model/decoder_main_with_kvcache_small_5beam_448pad_inputs.npz")
    inputs = np.load("./test_input.npz")
    tokens_input = inputs["tokens_input"].astype(np.int32)
    audio_features = inputs["audio_features"].astype(np.float16)
    positional_embedding_input = inputs["positional_embedding_input"].astype(np.float16)
    mask = inputs["mask"].astype(np.float16)

    start_time = time.time()
    decoder_sg_infer = SGInfer(decoder_bmodel_path, 1, [0])
    decoder_sg_infer.put(tokens_input, audio_features, positional_embedding_input, mask)
    _, result, _ = decoder_sg_infer.get()
    output_sg = torch.from_numpy(result[0])
    print("{:=^80}".format(f" sg_infer "))
    print(output_sg)
    print("sg time: ", time.time() - start_time)

    start_time = time.time()
    tool1 = Tool()
    bmrt1 = tool1.bmrt(handle)
    decoder_handle = tool1.create_model(decoder_bmodel_path.encode("utf-8"), bmrt1)
    runtime1       = tool1.create_un_runtime(handle)
    print(runtime1)
    tool1.set_bmodel_info(runtime1, decoder_handle)
    print("!!!!")
    tool1.set_stage(runtime1, 0)
    tool1.init_all_tensors(runtime1)
    tool1.malloc_device_address(runtime1)
    decoder_info = tool1.model_info(decoder_handle)
    tool1.copy_data_from_numpy(tool1.get_input_tensor(runtime1, 0), make_np2c(tokens_input), data_type[np.int32])
    tool1.copy_data_from_numpy(tool1.get_input_tensor(runtime1, 1), make_np2c(audio_features), data_type[np.float16])
    tool1.copy_data_from_numpy(tool1.get_input_tensor(runtime1, 2), make_np2c(positional_embedding_input), data_type[np.float16])
    tool1.copy_data_from_numpy(tool1.get_input_tensor(runtime1, 3), make_np2c(mask), data_type[np.float16])

    output_untool = np.empty(decoder_info[0]['output_shapes'][0], dtype=data_type_map[decoder_info['output_dtypes'][0]])
    print(output_untool)
    tool1.copy_data_from_numpy(tool1.get_output_tensor(runtime1, 0), make_np2c(output_untool), decoder_info['output_dtypes'][0])
    tool1.copy_input_data_to_device(runtime1)
    tool1.inference(runtime1)
    tool1.copy_output_data_to_host(runtime1)
    tool1.print_output_data(runtime1)
    # print(output_untool)
    # tool1.device_to_host(tool1.get_output_tensor(runtime1, 0))
    output_untool = torch.from_numpy(output_untool)
    print("{:=^80}".format(f" untool_infer "))
    print(output_untool)
    print("untool time: ", time.time() - start_time)

    tool1.destroy_un_runtime(runtime1)
    tool1.destroy_model(decoder_handle)
    tool1.free_bmrt(bmrt1)
    # tool1.free_bmhandle(handle)

if __name__ == "__main__":
    compare()