#!/bin/bash

# Default values for parameters
model="small"
beam_size=5
padding_size=448
use_kvcache=false

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

echo "Generating onnx models ..."
if [ "$use_kvcache" = true ]; then
    python run.py demo.wav --model $model --beam_size $beam_size --padding_size $padding_size --export_onnx  --use_kvcache
else
    python run.py demo.wav --model $model --beam_size $beam_size --padding_size $padding_size --export_onnx
fi

if [ ! -d onnx_model ]; then
    mkdir "onnx_model"
    echo "[Cmd] mkdir onnx_model"
fi

mv *.onnx *.npz onnx_model