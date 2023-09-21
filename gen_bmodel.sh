#!/bin/bash

# Default values for parameters
model="small"
beam_size=5
padding_size=448
compare=false
use_kvcache=false
bmodel_dir="bmodel"

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
        --compare)
            compare=true
            shift
            ;;
        --use_kvcache)
            use_kvcache=true
            shift
            ;;
        *)
            # Unknown option
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

padding_size_1=$((padding_size - 1))
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
elif [ "$model" == "large" ]; then
	n_text_state=1280
	n_text_head=20
	n_text_layer=32
else
	echo "model must be one of tiny, base, small, medium, large"
	exit 1
fi

process_list=("encoder" "logits_decoder" "decoder_post")
# process_list=("encoder")
echo "Generating process list ..."
if [ "$use_kvcache" = true ]; then
    process_list+=("decoder_main_with_kvcache" "decoder_loop_with_kvcache")
else
    process_list+=("decoder_main" "decoder_loop")
fi
echo "process list: ${process_list[@]}"

if [ ! -d "$bmodel_dir" ]; then
    mkdir "$bmodel_dir"
    echo "[Cmd] mkdir $bmodel_dir"
fi

for process_name in "${process_list[@]}"; do
    echo "Transforming $process_name ..."
    model_name="${process_name}_${model}_${beam_size}beam_${padding_size}pad"
    bmodel_file="${model_name}_1684x_f16.bmodel"

    if [ -e "./bmodel/$bmodel_file" ]; then
        echo "[Msg] ./bmodel/$bmodel_file already exists, skip this process"
        continue
    fi

    case $process_name in 
        encoder)
            input_shapes="[[1,80,3000]]"
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
    echo "[Cmd] Running command: $model_transform_cmd"
    eval "$model_transform_cmd"

    echo "[Cmd] Running command: $model_deploy_cmd"
    eval "$model_deploy_cmd"

    echo "Moving bmodel to ../../bmodel/"

    echo "[Cmd] mv $bmodel_file ../../bmodel/"
    mv $bmodel_file ../../bmodel/
    popd
    popd
    rm -rf $process_name
done

