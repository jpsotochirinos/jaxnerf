#import jaxnerf.train as train
from flask import config
from flask.config import Config
from jaxnerf.db.db import Model,Eval, Profile,Render,Train,db,app
from jaxnerf.db.init import init
from flask import Flask, jsonify, request
import subprocess
import psutil
import time
import os
from jaxnerf.nd import dataset
from jaxnerf.nd import utils

##path = os.listdir('/mnt/nerf') ## /mnt/nerf

#app = Flask(__name__)

@app.route('/create/',methods=['POST'])
async def create():
    _model_ = Model(
    model = request.json['model'],
    description = request.json['description'],
    bucket = request.json['bucket'],
    type = request.json['type'],
    status = request.json['status'],
    process = request.json['process'],
    checkpoint = request.json['checkpoint'],
    last_test = request.json['last_test'],
    last_step = request.json['last_step'],
    max_step = request.json['max_step'],
    path_render = request.json['path_render'],
    time_train = request.json['time_train'],
    time_render = request.json['time_render'],
    config = request.json['config']
    ) 
    _profile_=Profile(
        place=request.json['place'],
        history=request.json['history'],
        images=request.json['images'],
        video=request.json['video'],
        model_3d=request.json['model_3d'],
        render_path=request.json['render_path'],
    )
    try:
        _model_.profiles.append(_profile_)   
        db.session.add(_model_)
        db.session.commit()
    except ValueError:
        print(ValueError)
        return  jsonify({"status":"505",
                     "message": "error"})
    return  jsonify({"status":"200",
                     "message": "succes"})
                     
@app.route('/check/',methods=['POST'])
async def check():
    model = request.json['model']
    check = request.json['check']
    _model = Model.query.filter_by(model=model).first()
    #if(_model is None):
    #    return jsonify({"status":"404",
    #                 "message": "model no found"})
    cpu= utils.checkCPU()
    mem= utils.checkMEM()
    files = utils.checkModelFile(model,check)

    if(cpu>80 and 
       mem>20 and 
       files):
        _model.status = "ready2train"
        db.session.merge(_model)
        db.session.commit()
        return  jsonify({
                    "cpu":cpu,
                    "memory": mem,
                    "files": files,
                    "model_status":"ready2train",
                    "status":"200",
                     "message": "succes"})
    else:
        return  jsonify({"status":"503",
                     "message": "not enough resources"})

@app.route('/train/',methods=['POST'])
async def basic_train():
    model = request.json['model']
    _model = Model.query.filter_by(model=model).first()
    #if(_model is None):
    #    return jsonify({"status":"404",
    #                 "message": "model no found"})
    if(_model.status != "ready2train"):
         return jsonify({"status":"400",
         "message": "no ready to train yet"})

    try:
        print(model)
        #comando en segundo plano
        subprocess.Popen(["python","-m" ,"jaxnef.train",
        "--data_dir ",dataset.DATA_DIR,
        "--train_dir ",dataset.TRAIN_DIR,
        "--config ",dataset.CONFIG])
        _model.status = "training"
        db.session.merge(_model)
        db.session.commit()
        return  jsonify({"status":"200",
                        "message": "succes"})
    except ValueError:
        print(ValueError)
        return  jsonify({"status":"500",
                     "message": "erro"})
        #cambiar el estado a error
        #get last step
@app.route('/',methods=['POST'])
async def basic_train():
    model = request.json['process']
    try:
        print(model)
        #comando en segundo plano
        subprocess.Popen(["python","-m", "jaxnerf.ia" ])
        return  jsonify({"status":"200",
                        "message": "succes"})
    except ValueError:
        print(ValueError)
        return  jsonify({"status":"500",
                     "message": "erro"})

@app.route('/model/',methods=['POST'])
def model_status():
    try:
       db.session.commit()
       h = Model.query.filter_by(rg_model=request.json['model']).first()
       heritage = {
           "model":h.rg_model,
           "max_step":h.rg_max_step,
           "last_step":h.rg_last_step,
           "checkpoint":h.rg_checkpoint,
           "config":h.rg_config,
           "last_train":h.rg_train,
           "status":h.rg_status,
           "render_path":h.rg_path_render
       }
       return  jsonify({"status":"200",
                     "message": "succes",
                     "heritage":heritage})
    except ValueError:
        print(ValueError)
        return  jsonify({"status":"500",
                     "message": "erro"})

@app.route('/model/train/',methods=['POST'])
def model_train():
    db.session.commit()
    model = request.json['model']
    h = Model.query.filter_by(rg_model=model).first()
    if not h:
        return  jsonify({"status":"404",
                    "message": "no found"})
    else: 
        t = h.trains
        trains =[]
        for train in t:
            aux = {
                "model_step":train.rg_model,
                "step":train.rg_last_step,
                "i_loss":train.rg_i_loss,
                "avg_loss":train.rg_avg_loss,
                "weight_l2":train.rg_weight_l2,
                "lr":train.rg_lr,
                "ray_per_sec":train.rg_rays_per_sec,
            }
            trains.append(aux)

        return  jsonify({"status":"200",
                        "message": "succes",
                        model:trains})


@app.route('/status/',methods=['POST']) 
async def status():
    command_name = request.json['process']
    process = []
    if utils.checkIfProcessRunning(command_name):
        procObjList = [procObj for procObj in psutil.process_iter() if command_name in procObj.name().lower() ]
        for elem in procObjList:
            process_pid = elem.pid
            process_name = elem.name()
            process_status = elem.status()
            process_create_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(elem.create_time()))
            if(process_status=="running"):
                print('here')
            process.append ({
                "pid": str(process_pid),
                "name": process_name,
                "status": process_status,
                "create_time":process_create_time
            })

        return jsonify({
            "process":process,
            "status":200,
            "message":"succes"
        })
                    
    else:
        return jsonify({
            "status":404,
            "message":"no found process"
        })

@app.route('/stop/',methods=['POST']) 
async def stop():
    command_name = request.json['process']
    process = []
    if utils.checkIfProcessRunning(command_name):
        procObjList = [procObj for procObj in psutil.process_iter() if command_name in procObj.name().lower() ]
        for elem in procObjList:
            process_pid = elem.pid
            process_name = elem.name()
            process_status = elem.status()
            process_create_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(elem.create_time()))
            if(process_status=="running"):
                elem.terminate()
                process.append ({
                    "pid": str(process_pid),
                    "name": process_name,
                    "status": "stopped",
                    "create_time":process_create_time
                })

        return jsonify({
            "process":process,
            "status":202,
            "message":"accepted"
        })
                    
    else:
        db.session.commit()
        h = Model.query.all()
        h[len(h)-1].rg_status = 'stopped'
        db.session.commit()
        return jsonify({
            "status":404,
            "message":"no found process"
        })
        
if __name__ == '__main__':
    if(os.path.exists('jaxnerf/db/datatrain.db')):
        print(" * DataBase already created")
    else:
        print(" * DataBase created")
        init()
    app.debug = True
    app.run(host = '0.0.0.0',port= 3000)