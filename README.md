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

## Http api usage

### Start api service
`python api.py` 
    
    This service depends on the whisper command. Please install whisper successfully before using this service.

### upload task
- url `/upload_task`
- method `post`
- form parameter `{'file': audio file}`
- response `{'task_id': task_id}`
- description `upload an audio file in form, the key is 'file'. The system will add the requested file to the task queue and reutrn a task_id at the same time.`

### download result file
- url `check_task`
- method `post`
- json parameter `{'task_id': task_id}`
- response `{"state": task state}` or `.zip result file`
- description `check if the task with task_id is completed. If completed, return the result file; if not, return the task status. `