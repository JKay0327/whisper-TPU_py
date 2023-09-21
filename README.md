# whisper-TPU python

## Environment
The codebase is expected to be compatible with Python 3.8-3.11 and recent PyTorch versions. The codebase also depends on a few Python packages, most notably OpenAI's tiktoken for their fast tokenizer implementation. You can setup the environment with the following command:
```bash
pip install requirements.txt
```
It also requires the command-line tool `ffmpeg` to be installed on your system, which is available from most package managers:
```bash
# if you use a conda environment
conda install ffmpeg
 
# on Ubuntu or Debian
sudo apt update && sudo apt install ffmpeg 
```
You can install bmwhisper as follow:
```bash
python setup.py install
```

## Command-line usage
### CPU mode
The following command will transcribe speech in audio files using cpu, using the 'base' model:
```bash
bmwhisper demo.wav --model base
```
### TPU mode
To use TPU, firstly, you need to generate the onnx model:
```bash
./gen_onnx.sh --model base

# if you want to use kvcache
./gen_onnx.sh --model base --use_kvcache
```
Then, transform onnx model to bmodel:
```bash
./gen_bmodel.sh --model base

# if you want to use kvcache
./gen_bmodel.sh --model base --use_kvcache

# if you want to compare the data when transforming and deploying
./gen_bmodel.sh --model base --compare
```
Use `--inference` button to allow TPU inference mode:
```bash
bmwhisper demo.wav --model base --inference
```