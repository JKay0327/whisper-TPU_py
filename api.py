import os
from flask import Flask, request, jsonify, send_file
import queue
import datetime
import subprocess
import zipfile
import threading

"""
url: /upload_task 
desc: upload an audio file in form, the key is 'file'. The system will add the requested file to the task queue and reutrn a task_id at the same time.

url: /check_task
desc: check if the task with task_id is completed. If completed, return the result file; if not, return the task status. 
"""

app = Flask(__name__)
task_queue = queue.Queue()

# label task number in this setup.
app.config['TASK_COUNT'] = 0

base_dir = "./api_cache"
if not os.path.exists(base_dir):
    os.makedirs(base_dir)


def execute_task(task):
    out_dir = os.path.dirname(task) + "/result"
    try:
        console_result = subprocess.run(
            ['bmwhisper', task, "--model", "small", "--output_dir", out_dir, "--bmodel_dir",
             "/data/whisper-TPU_py/bmodel", "--chip_mode", "soc", "--verbose", "False"], capture_output=True, text=True)
        # print(console_result)
        with zipfile.ZipFile(os.path.dirname(task) + "/result.zip", 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(out_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, os.path.relpath(file_path, out_dir))
        print(f"{task.split('/')[-2]} task complete. {task_queue.qsize()} remaining tasks.")
    except Exception as e:
        with open(os.path.dirname(task) + "/error.txt", 'w') as file:
            file.write(str(e))
        print(f"{task.split('/')[-2]} task error. {task_queue.qsize()} remaining tasks.")


def background_task_worker():
    while True:
        if not task_queue.empty():
            task = task_queue.get()
            execute_task(task)


@app.route('/upload_task', methods=['POST'])
def upload_task():
    if 'file' not in request.files:
        return 'No file part in the request', 400

    audio = request.files['file']

    if audio.filename == '':
        return 'No selected file', 400
    else:
        task_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S") + str(app.config['TASK_COUNT'])
        app.config['TASK_COUNT'] += 1

        if not os.path.exists(base_dir + "/" + task_id):
            os.makedirs(base_dir + "/" + task_id)
        task_url = base_dir + "/" + task_id + "/" + audio.filename
        audio.save(task_url)
        task_queue.put(task_url)

    return jsonify({"task_id": task_id})


@app.route('/check_task', methods=['POST'])
def check_task():
    task_id = str(request.json['task_id'])
    if not os.path.exists(base_dir + "/" + task_id + "/result.zip"):
        if os.path.exists(base_dir + "/" + task_id + "/error.txt"):
            with open('error.txt', 'r') as file:
                content = file.read()
            return jsonify({"state": "task error.", "message": content})
        else:
            return jsonify({"state": "in progress."})
    else:
        return send_file(base_dir + "/" + task_id + "/result.zip")


if __name__ == '__main__':
    background_worker = threading.Thread(target=background_task_worker)
    background_worker.start()
    app.run(host='0.0.0.0', port=5000)
