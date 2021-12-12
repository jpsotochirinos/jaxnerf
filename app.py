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
    model = request.json['model']
    _exist = Model.query.filter_by(model=model).first()
    if(_exist is not None):
        return  jsonify({"status":"400",
                     "message": "all ready exists"})
    if(utils.check_models(model) is False):
        return  jsonify({"status":"404",
                     "message": "model storage no found"})
    _model = Model(
    model = request.json['model']
    ) 
    _profile_=Profile()
    def check_keys():
        try:
            _model.description=request.json['description'] 
        except KeyError:
            print(KeyError)
        try:
            _model.bucket = request.json['bucket'] 
        except KeyError:
            print(KeyError)
        try:
            _model.type = request.json['type'] 
        except KeyError:
            print(KeyError)
        try:
            _model.status = request.json['status'] 
        except KeyError:
            print(KeyError)
        try:
            _model.process = request.json['process'] 
        except KeyError:
            print(KeyError)
        try:
            _model.checkpoint = request.json['checkpoint'] 
        except KeyError:
            print(KeyError)
        try:
            _model.last_test = request.json['last_test'] 
        except KeyError:
            print(KeyError)
        try:
            _model.last_step = request.json['last_step'] 
        except KeyError:
            print(KeyError)
        try:
            _model.max_step = request.json['max_step'] 
        except KeyError:
            print(KeyError)
        try:
            _model.time_train = request.json['time_train']
        except KeyError:
            print(KeyError)
        try:
            _model.time_render = request.json['time_render'] 
        except KeyError:
            print(KeyError)
        try:
            _model.config = request.json['config'] 
        except KeyError:
            print(KeyError)
        try:
            _model.files_checker = request.json['files_checker'] 
        except KeyError:
            print(KeyError)
        
        try:
            _profile_.place=request.json['place'] 
        except KeyError:
            print(KeyError)
        try:
            _profile_.history=request.json['history'] 
        except KeyError:
            print(KeyError)
        try:
            _profile_.images=request.json['images'] 
        except KeyError:
            print(KeyError)
        try:
            _profile_.video=request.json['video'] 
        except KeyError:
            print(KeyError)
        try:
            _profile_.model_3d=request.json['model_3d'] 
        except KeyError:
            print(KeyError)
    check_keys()
    _model.profiles.append(_profile_)
    try:   
        utils.get_models(_model.model) 
        u_str= ""
        _status, f_chk = utils.checkModelFile(_model.model,False)
        _model.files_checker = u_str.join(f_chk)
        if(_status):
            _model.status="ready2train"
        else:
            _model.status="missfiles"
        db.session.add(_model)
        db.session.commit()
    except ValueError:
        print(ValueError)
        return  jsonify({"status":"505",
                     "message": "error"})
    return  jsonify({
            "status":"200",
            "model":_model.model,
            "file_checker":_model.files_checker,
            "message": "succes"})
                     
@app.route('/check/',methods=['POST'])
async def check():
    model = request.json['model']
    _model = Model.query.filter_by(model=model).first()
    if(_model is None):
        return jsonify({"status":"404",
                     "message": "model no found"})
    cpu= utils.checkCPU()
    mem= utils.checkMEM()
    if(_model.status!="ready2train"):
        _status,files = utils.checkModelFile(model,True)
    else:
        _status = True

    if(cpu>80 and 
       mem>20 and 
       _status):
        _model.status = "ready2train"
        db.session.merge(_model)
        db.session.commit()
        return  jsonify({
                    "cpu":cpu,
                    "memory": mem,
                    "files": _model.files_checker,
                    "model_status":"ready2train",
                    "status":"200",
                     "message": "succes"})
    else:
        return  jsonify({
                    "files_checker": _model.files_checker,
                    "status":"503",
                    "message": "not enough resources"})

@app.route('/train/',methods=['POST'])
async def basic_train():
    model = request.json['model']
    _model = Model.query.filter_by(model=model).first()
    if(_model is None):
        return jsonify({"status":"404",
                     "message": "model no found"})
    ##if(_model.status != "ready2train"):
    ##     return jsonify({"status":"400",
    ##     "message": "no ready to train yet"})

    try:
        print(model)
        #comando en segundo plano
        __proce = subprocess.Popen(["python","-m" ,"jaxnerf.process_test.test"])
        _model.process = __proce.pid
        #"--data_dir ",dataset.DATA_DIR,
        #"--train_dir ",dataset.TRAIN_DIR,
        #"--config ",dataset.CONFIG])
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
async def home():
    try:
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
       model = request.json['model']
       _model = Model.query.filter_by(model=model).first()
       
       if(_model is None):
           return  jsonify({"status":"404",
                     "message": "no found"})
       model_obj = {
            "model": _model.model,
            "description": _model.description,
            "bucket" : _model.bucket,
            "type" : _model.type,
            "status" : _model.status,
            "process" : _model.process,
            "checkpoint": _model.checkpoint,
            "last_test": _model.last_test,
            "last_step" : _model.last_step,
            "max_step" : _model.max_step,
            "time_train" : _model.time_train,
            "time_render" : _model.time_render,
            "config" : _model.config,
            "files_checker": _model.files_checker
       }
       _profile = _model.profiles[0]
       profile_obj = {
            "place":_profile.place ,
            "history":_profile.history ,
            "images":_profile.images ,
            "video":_profile.video ,
            "model_3d":_profile.model_3d 
       }
       return  jsonify({"status":"200",
                     "message": "succes",
                     "model":model_obj,
                     "profile": profile_obj})
    except ValueError:
        print(ValueError)
        return  jsonify({"status":"500",
                     "message": "erro"})

@app.route('/model/train/',methods=['POST'])
def model_train():
    db.session.commit()
    model = request.json['model']
    _model = Model.query.filter_by(model=model).first()
    if(_model is None):
        return  jsonify({"status":"404",
                    "message": "no found"})
    else: 
        _trains = _model.trains
        trains =[]
        for _train in _trains:
            aux = {
                "model":_model.model,
                "step":_train.last_step,
                "i_loss":_train.i_loss,
                "avg_loss":_train.avg_loss,
                "weight_l2":_train.weight_l2,
                "lr":_train.lr,
                "ray_per_sec":_train.rays_per_sec,
                "cpu_percent":_train.cpu_percent,
                "mem_percent":_train.mem_percent,
                "type_step":_train.type_step
            }
            trains.append(aux)

        return  jsonify({"status":"200",
                        "message": "succes",
                        model:trains})

@app.route('/status/',methods=['POST']) 
async def status():
    model = request.json['model']
    _model = Model.query.filter_by(model=model).first()
    _pid = _model.process
    process = []
    procObjList = [procObj for procObj in psutil.process_iter() if int(_pid) == procObj.pid ]
    print(_pid)
    print(procObjList)
    if (len(procObjList)>0):
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
    model = request.json['model']
    _model = Model.query.filter_by(model=model).first()
    _pid = _model.process
    process =[]
    procObjList = [procObj for procObj in psutil.process_iter() if int(_pid) == procObj.pid ]
    if (len(procObjList)>0):
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
        _model.status = 'stopped'
        db.session.merge(_model)
        db.session.commit()
        return jsonify({
            "status":404,
            "message":"no found process"
        })
        
if __name__ == '__main__':
    if(os.path.exists('tmp') and os.path.exists('tmp/models')):
        print(" * Folder tmp already created")
    else:
        tmp_path = os.path.join(dataset.ROOT_DIR,'tmp')
        models_path = os.path.join(dataset.ROOT_DIR+'/tmp','models')
        os.mkdir(tmp_path)
        os.mkdir(models_path)
        print(" * Folder tmp created")
    if(os.path.exists('jaxnerf/db/datatrain.db')):
        print(" * DataBase already created")
    else:
        print(" * DataBase created")
        init()
    app.debug = True
    app.run(host = '0.0.0.0',port= 3000)