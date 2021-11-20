import jaxnerf.train as train
import jax
import os
import requests
from flask import Flask, jsonify, request

MODELS_PATH = '/mnt/nerf/models/'
CHECKPNT_PATH = '/mnt/nerf/checkpoints/'
CONFIG_PATH = '/mnt/nerf/configs/'
TPU_DRIVER_MODE = 1
##path = os.listdir('/mnt/nerf') ## /mnt/nerf

app = Flask(__name__)

@app.route('/test/',methods=['POST'])
async def train_model():
    path = os.getenv('KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS')
    path = path.split(':')
    new_flags={
        "data_dir": MODELS_PATH + request.json['data_dir'],
        "train_dir": CHECKPNT_PATH + request.json['train_dir'],
        "config": CONFIG_PATH + request.json['config']
    }

    try:
       print(new_flags)
       reqq = requests.post('http://'+path[1][2:]+':8475/requestversion/tpu_driver_nightly')
       #await train.run_train(new_flags)
       #cambiar el estado del proceso de pending o progress bar a terminado
       print("finish")
    except ValueError:
        print(ValueError)
        #cambiar el estado a error
        #get last step
    return  jsonify({"status":"200",
                     "message": "succes",
                     "requests": reqq,
                     "jax_her":jax.devices()})


@app.route('/train/',methods=['POST'])
async def basic_train():
    new_flags={
        "data_dir": MODELS_PATH + request.json['data_dir'],
        "train_dir": CHECKPNT_PATH + request.json['train_dir'],
        "config": CONFIG_PATH + request.json['config']
    }
    try:
       print(new_flags)
       await train.run_train(new_flags)
       return  jsonify({"status":"200",
                     "message": "succes"})
       #cambiar el estado del proceso de pending o progress bar a terminado
       print("finish")
    except ValueError:
        print(ValueError)
        return  jsonify({"status":"500",
                     "message": "erro"})
        #cambiar el estado a error
        #get last step
    

if __name__ == '__main__':
    app.debug = True
    app.run(host = '0.0.0.0',port= 3000)