#!/bin/bash

# Default values for parameters
model="small"
beam_size=5
padding_size=448
use_kvcache=false

python setup.py install

work_dir="tmp"
if [ ! -d "$work_dir" ]; then
    mkdir "$work_dir"
    echo "[Cmd] mkdir $work_dir"
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

onnx_export_cmd="bmwhisper ../test/demo.wav --model $model --beam_size $beam_size --padding_size $padding_size --export_onnx"

echo "Generating onnx models ..."
if [ "$use_kvcache" = true ]; then
    onnx_export_cmd="$onnx_export_cmd --use_kvcache"
fi

echo "[Cmd] Running command: $onnx_export_cmd"
eval "$onnx_export_cmd"

if [ ! -d "onnx_model" ]; then
    mkdir "onnx_model"
    echo "[Cmd] mkdir onnx_model"
fi

mv *.onnx *.npz onnx_model

popd