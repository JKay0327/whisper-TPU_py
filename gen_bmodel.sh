#!/bin/bash

# Default values for parameters
model="large-v2"
beam_size=5
padding_size=448
compare=true
use_kvcache=true
quant=true
process=""

bmodel_dir="bmodel"
if [ ! -d "$bmodel_dir" ]; then
    mkdir "$bmodel_dir"
    echo "[Cmd] mkdir $bmodel_dir"
fi

work_dir="tmp"
if [ ! -d "$work_dir" ]; then
    echo "[Err] "$work_dir" directory not found, please run gen_onnx.sh first"
    exit 1
fi
pushd "$work_dir"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --model)
            model="$2"
            shift 2
            ;;
        --beam_size)
            beam_size="$2"
            shift 2
            ;;
        --padding_size)
            padding_size="$2"
            shift 2
            ;;
        --process)
            process="$2"
            shift 2
            ;;
        --compare)
            compare=true
            shift
            ;;
        --use_kvcache)
            use_kvcache=true
            shift
            ;;
        --quant)
            quant=true
            shift
            ;;
        *)
            # Unknown option
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# padding_size_1=$((padding_size - 1))
padding_size_1=$((padding_size))
n_mels=80
n_audio_ctx=1500

if [ "$model" == "tiny" ]; then
	n_text_state=384
	n_text_head=6
	n_text_layer=4
elif [ "$model" == "base" ]; then
	n_text_state=512
	n_text_head=8
	n_text_layer=6
elif [ "$model" == "small" ]; then
	n_text_state=768
	n_text_head=12
	n_text_layer=12
elif [ "$model" == "medium" ]; then
	n_text_state=1024
	n_text_head=16
	n_text_layer=24
elif [ "$model" == "large-v2" ]; then
	n_text_state=1280
	n_text_head=20
	n_text_layer=32
elif [ "$model" == "large-v3" ]; then
    n_mels=128
	n_text_state=1280
	n_text_head=20
	n_text_layer=32
else
	echo "model must be one of tiny, base, small, medium, large"
	exit 1
fi

function gen_bmodel() {
    echo "Transforming $process_name ..."
    case $process_name in 
        encoder)
            input_shapes="[[1,$n_mels,3000]]"
            ;;
        logits_decoder)
            input_shapes="[[1,1],[1,${n_audio_ctx},${n_text_state}]]"
            ;;
        decoder_post)
            input_shapes="[[$beam_size,1,$n_text_state],[$beam_size,1,$n_text_state]]"
            ;;
        decoder_main)
            input_shapes="[[$beam_size,$padding_size],[1,$n_audio_ctx,$n_text_state],[$padding_size,$n_text_state],[$beam_size,$padding_size,$n_text_head,$padding_size]]"
            ;;
        decoder_main_with_kvcache)
            input_shapes="[[$beam_size,$padding_size],[1,$n_audio_ctx,$n_text_state],[$padding_size,$n_text_state],[$beam_size,$padding_size,$n_text_head,$padding_size]]"
            ;;
        decoder_loop)
            input_shapes="[[$beam_size,$padding_size],[1,$n_audio_ctx,$n_text_state],[$padding_size,$n_text_state],[$beam_size,$padding_size,$n_text_head,$padding_size]]"
            ;;
        decoder_loop_with_kvcache)
            input_shapes="[[$beam_size,1],[1,$n_text_state],[$beam_size,1,$n_text_head,$padding_size]"
            for ((i=0; i<$((n_text_layer * 2)); i++)); do
                input_shapes="${input_shapes},[$beam_size,$padding_size_1,$n_text_state]"
            done
            for ((i=0; i<$((n_text_layer * 2)); i++)); do
                input_shapes="${input_shapes},[1,$n_audio_ctx,$n_text_state]"
            done
            input_shapes="${input_shapes}]"
            ;;
        *)
            # Unknown option
            echo "Unknown option: $process_name"
            exit 1
            ;;
    esac
    
    onnx_file="${model_name}.onnx"
    test_input="${model_name}_inputs.npz"
    test_result="${model_name}_top_outputs.npz"
    model_transform_cmd="model_transform.py --model_name $model_name \
        --model_def ../${onnx_file} \
        --input_shapes $input_shapes \
        --mlir transformed.mlir"

    model_deploy_cmd="model_deploy.py --mlir transformed.mlir \
        --quantize F16 \
        --chip bm1684x \
        --model $bmodel_file"

    if [ "$compare" = true ]; then
        model_transform_cmd="$model_transform_cmd --test_input ../${test_input} \
        --test_result $test_result \
        --debug"

        model_deploy_cmd="$model_deploy_cmd --test_reference $test_result \
        --test_input ../${test_input} \
        --compare_all \
        --debug"
    fi

    # if [ "$quant" = true ]; then
    #     if [ "$process_name" == "decoder_post" ]; then
    #         model_deploy_cmd="$model_deploy_cmd --quant_input"
    #     elif [ "$process_name" == "decoder_main_with_kvcache" ]; then
    #         model_deploy_cmd="$model_deploy_cmd --quant_output"
    #     else
    #         model_deploy_cmd="$model_deploy_cmd --quant_input --quant_output"
    #     fi
    # fi

    if [ "$quant" = true ]; then
        model_deploy_cmd="$model_deploy_cmd --quant_input --quant_output"
    fi

    echo "[Cmd] Running command: $model_transform_cmd"
    eval "$model_transform_cmd"

    echo "[Cmd] Running command: $model_deploy_cmd"
    eval "$model_deploy_cmd"

    echo "[Msg] Bmodel generate done!"
}

if [ -z "$process" ]; then
    process_list=("encoder" "logits_decoder" "decoder_post")
    # process_list=("logits_decoder")
    # process_list=("decoder_loop_with_kvcache")
    echo "Generating process list ..."
    if [ "$use_kvcache" = true ]; then
        process_list+=("decoder_main_with_kvcache" "decoder_loop_with_kvcache")
    else
        process_list+=("decoder_main" "decoder_loop")
    fi
    echo "process list: ${process_list[@]}"

    for process_name in "${process_list[@]}"; do
        model_name="${process_name}_${model}_${beam_size}beam_${padding_size}pad"
        bmodel_file="${model_name}_1684x_f16.bmodel"
        if [ "$quant" = true ]; then
            bmodel_file="all_quant_${bmodel_file}"
        fi
        if [ -e "./bmodel/$bmodel_file" ]; then
            echo "[Msg] ./bmodel/$bmodel_file already exists, skip this process"
            continue
        fi

        if [ ! -d "$process_name" ]; then
            mkdir "$process_name"
            echo "[Cmd] mkdir $process_name"
        fi
        if [ "$model" == "large-v3" ] || [ "$model" == "large-v2" ]; then
            cp onnx_model/$process_name/* $process_name/
        else
            cp onnx_model/$process_name*.* $process_name/
        fi
        pushd "$process_name"
        if [ ! -d "$model_name" ]; then
            mkdir "$model_name"
            echo "[Cmd] mkdir $model_name"
        fi
        pushd "$model_name"

        # ##############################
        # echo "Transforming $process_name ..."
        # case $process_name in 
        #     encoder)
        #         input_shapes="[[1,80,3000]]"
        #         ;;
        #     logits_decoder)
        #         input_shapes="[[1,1],[1,${n_audio_ctx},${n_text_state}]]"
        #         ;;
        #     decoder_post)
        #         input_shapes="[[$beam_size,1,$n_text_state],[$beam_size,1,$n_text_state]]"
        #         ;;
        #     decoder_main)
        #         input_shapes="[[$beam_size,$padding_size],[1,$n_audio_ctx,$n_text_state],[$padding_size,$n_text_state],[$beam_size,$padding_size,$n_text_head,$padding_size]]"
        #         ;;
        #     decoder_main_with_kvcache)
        #         input_shapes="[[$beam_size,$padding_size],[1,$n_audio_ctx,$n_text_state],[$padding_size,$n_text_state],[$beam_size,$padding_size,$n_text_head,$padding_size]]"
        #         ;;
        #     decoder_loop)
        #         input_shapes="[[$beam_size,$padding_size],[1,$n_audio_ctx,$n_text_state],[$padding_size,$n_text_state],[$beam_size,$padding_size,$n_text_head,$padding_size]]"
        #         ;;
        #     decoder_loop_with_kvcache)
        #         input_shapes="[[$beam_size,1],[1,$n_text_state],[$beam_size,1,$n_text_head,$padding_size]"
        #         for ((i=0; i<$((n_text_layer * 2)); i++)); do
        #             input_shapes="${input_shapes},[$beam_size,$padding_size_1,$n_text_state]"
        #         done
        #         for ((i=0; i<$((n_text_layer * 2)); i++)); do
        #             input_shapes="${input_shapes},[1,$n_audio_ctx,$n_text_state]"
        #         done
        #         input_shapes="${input_shapes}]"
        #         ;;
        #     *)
        #         # Unknown option
        #         echo "Unknown option: $process_name"
        #         exit 1
        #         ;;
        # esac

        # onnx_file="${model_name}.onnx"
        # test_input="${model_name}_inputs.npz"
        # test_result="${model_name}_top_outputs.npz"
        # model_transform_cmd="model_transform.py --model_name $model_name \
        #     --model_def ../${onnx_file} \
        #     --input_shapes $input_shapes \
        #     --mlir transformed.mlir"

        # model_deploy_cmd="model_deploy.py --mlir transformed.mlir \
        #     --quantize F16 \
        #     --chip bm1684x \
        #     --model $bmodel_file"
        # if [ "$compare" = true ]; then
        #     model_transform_cmd="$model_transform_cmd --test_input ../${test_input} \
        #     --test_result $test_result \
        #     --debug"

        #     model_deploy_cmd="$model_deploy_cmd --test_reference $test_result \
        #     --test_input ../${test_input} \
        #     --compare_all \
        #     --debug"
        # fi
        # if [ "$quant" = true ]; then
        #     if [ "$process_name" == "decoder_post" ]; then
        #         model_deploy_cmd="$model_deploy_cmd --quant_input"
        #     else
        #         model_deploy_cmd="$model_deploy_cmd --quant_input --quant_output"
        #     fi
        # fi
        # echo "[Cmd] Running command: $model_transform_cmd"
        # eval "$model_transform_cmd"
        # echo "[Cmd] Running command: $model_deploy_cmd"
        # eval "$model_deploy_cmd"

        gen_bmodel

        echo "[Cmd] cp $bmodel_file ../../../bmodel/"
        cp $bmodel_file ../../../bmodel/
        popd
        rm -rf $bmodel_file
        popd
    done
else 
    process_name="$process"
    model_name="${process_name}_${model}_${beam_size}beam_${padding_size}pad"
    echo "Generating $model_name ..."
    bmodel_file="${model_name}_1684x_f16.bmodel"
    echo "bmodel_file: $bmodel_file"
    if [ "$quant" = true ]; then
        bmodel_file="quant_${bmodel_file}"
    fi

    if [ ! -d "$process_name" ]; then
        mkdir "$process_name"
        echo "[Cmd] mkdir $process_name"
    fi
    cp onnx_model/$process_name*.* $process_name/
    pushd "$process_name"
    if [ ! -d "$model_name" ]; then
        mkdir "$model_name"
        echo "[Cmd] mkdir $model_name"
    fi
    pushd "$model_name"
    # exit 1
    gen_bmodel
    popd
    popd
fi

popd
chown -R 1000:1000 .
