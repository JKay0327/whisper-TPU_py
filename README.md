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
### TPU mode
Please disable debug info first:
```bash
export LOG_LEVEL=-1
```
Default model is `small`, start using whisper-TPU with `bmwhisper` in this catalogue `whisper-TPU_py`:
```bash
bmwhisper demo.wav
```
Or you can set the absolute path of bmodel dir like this `--bmodel_dir [bmodel_dir]`, and `bmwhisper` can be used anywhere:
```bash
bmwhisper demo.wav --bmodel_dir /your/path/to/bmodel_dir
```
You can change the model by adding `--model [model_name]`:
```bash
bmwhisper demo.wav --model medium
```
Model available now:
* base
* small
* medium
You can change the chip mode by adding `--chip_mode soc`, default is `pcie`:
```bash
bmwhisper demo.wav --chip_mode soc
```