#import jaxnerf.train as train
from flask import config
from flask.config import Config
from jaxnerf.nd.dataset import DATA_DIR, TRAIN_DIR
from jaxnerf.db.db import Model,Eval, Profile,Render,Train,Tpu,Performance,db,app
from jaxnerf.db.init import init
from flask import Flask, jsonify, request
import subprocess
import psutil
import time
import os
import requests
from jaxnerf.nd import dataset
from jaxnerf.nd import utils

##path = os.listdir('/mnt/nerf') ## /mnt/nerf
TRAINING = False
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
            _model.config = request.json['config'] 
        except KeyError:
            print(KeyError)
        try:
            _model.factor = request.json['factor'] 
        except KeyError:
            print(KeyError)
        try:
            _model.factor_guess = request.json['factor_guess'] 
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
    fck= ""
    _model = Model.query.filter_by(model=model).first()
    if(_model is None):
        return jsonify({"status":"404",
                     "message": "model no found"})
    _model.factor=str(utils.check_img_size(_model.model))
    if(_model.factor!=_model.factor_guess):
        _model.status = "checkresize"
        db.session.merge(_model)
        db.session.commit()
        return jsonify({"status":"505",
                        "factor_ok":_model.factor,
                        "factor_guess":_model.factor_guess,
                        "model_status":_model.status,
                        "message": "check factor and resize"})
    cpu= utils.checkCPU()
    mem= utils.checkMEM()
    if(_model.status!="ready2train"):
        _status,files = utils.checkModelFile(model,True)
        _model.files_checker = fck.join(files)
    else:
        _status = True
        _model.files_checker = "11111111"
    if(cpu>80 and 
       mem>20 and 
       _status and
       _model.factor==_model.factor_guess 
       ):
        _model.status = "ready2train"
        db.session.merge(_model)
        db.session.commit()
        return  jsonify({
                    "cpu":cpu,
                    "memory": mem,
                    "files": _model.files_checker,
                    "factor": _model.factor,
                    "model_status":_model.status,
                    "status":"200",
                     "message": "succes"})
    else:
        return  jsonify({
                    "files_checker": _model.files_checker,
                    "files_factor": _model.factor,
                    "tpu": reqq.status_code,
                    "status":"503",
                    "message": "not enough resources"})

@app.route('/resize_auth/',methods=['POST'])
async def resize_auth():
    model = request.json['model']
    _model = Model.query.filter_by(model=model).first()
    if(_model is None):
        return jsonify({"status":"404",
                     "message": "model no found"})

    if(_model.files_checker !="11111111"):
        return jsonify({"status":"404",
                     "message": "something wrong with files"})

    if(_model.factor>_model.factor_guess):
        r = utils.minify(_model.model,int(_model.factor))
        _model.factor_guess = _model.factor
        db.session.merge(_model)
        db.session.commit()
        return jsonify({"status":"200",
                        "factor":_model.factor,
                        "message": "resize in "+r+"%"})
    else:
        return  jsonify({
                    "files_factor": _model.factor,
                    "status":"200",
                    "message": "unchanged"})

@app.route('/resize/',methods=['POST'])
async def resize():
    model = request.json['model']
    factor = request.json['factor']
    _model = Model.query.filter_by(model=model).first()
    if(_model is None):
        return jsonify({"status":"404",
                     "message": "model no found"})

    if(_model.files_checker !="1111111"):
        return jsonify({"status":"404",
                     "message": "something wrong with files"})
    
    r = utils.minify(_model.model,factors=[int(factor)])
    _model.factor = factor
    _model.factor_guess = factor
    db.session.merge(_model)
    db.session.commit()
    return jsonify({"status":"200",
                    "factor":_model.factor,
                    "message": "resize in "+r})


@app.route('/train/',methods=['POST'])
async def basic_train():
    _tpu = Tpu.query.filter_by(acelerator="v3-8").first()
    model = request.json['model']
    _model = Model.query.filter_by(model=model).first()
    if(_model is None):
        return jsonify({"status":"404",
                     "message": "model no found"})

    if(_tpu.status):
        return jsonify({"status":"500",
        "message": "training"})

    if(_model.status != "ready2train"):
        return jsonify({"status":"400",
        "message": "no ready to train yet"})
    
    try:
        _dataDir= DATA_DIR+_model.model
        _trainDir= TRAIN_DIR+_model.model
        _config = "configs/"+_model.config
        #comando en segundo plano
        _test = ["python","-m","jaxnerf.process_test.test"]
        _perf = ["python","-m","jaxnerf.performance"]
        _train = ["python","-m" ,"jaxnerf.train",
                  "--data_dir="+_dataDir,
                  "--train_dir="+_trainDir,
                  "--config="+_config,
                  "--factor="+_model.factor]
        #_catch = subprocess.Popen(_test)
        __proce = subprocess.Popen(_train)       
        _model.process = __proce.pid
        _model.status = "training"
        _tpu.status = True
        _tpu.model = _model.model
        _tpu.pid_model = __proce.pid
        db.session.merge(_model)
        db.session.merge(_tpu)
        db.session.commit()
        __perf = subprocess.Popen(_perf)
        print(__perf.pid)
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

@app.route('/model/',methods=['GET'])
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

@app.route('/model/train/',methods=['GET'])
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
    if(_model is None):
        return  jsonify({"status":"404",
                    "message": "no found"})
    _pid = _model.process
    process = []
    procObjList = [procObj for procObj in psutil.process_iter() if int(_pid) == procObj.pid ]
    if (len(procObjList)>0):
        for elem in procObjList:
            process_pid = elem.pid
            process_name = elem.name()
            process_status = elem.status()
            process_create_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(elem.create_time()))
            if(process_status!="zombie"):
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
    if(_model is None):
        return  jsonify({"status":"404",
                    "message": "no found"})
    _tpu = Tpu.query.filter_by(acelerator="v3-8").first()
    _pid = _model.process
    process =[]
    procObjList = [procObj for procObj in psutil.process_iter() if int(_pid) == procObj.pid ]
    if (len(procObjList)>0):
        for elem in procObjList:
            process_pid = elem.pid
            process_name = elem.name()
            process_status = elem.status()
            process_create_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(elem.create_time()))
            if(process_status!="zombie"):
                elem.terminate()
                process.append ({
                    "pid": str(process_pid),
                    "name": process_name,
                    "status": "stopped",
                    "create_time":process_create_time
                })
        _tpu.status = False
        _tpu.model = ""
        _tpu.type_step = ""
        _tpu.pid_model = ""
        _model.status = 'stopped'
        db.session.merge(_tpu)
        db.session.merge(_model)
        db.session.commit()
        return jsonify({
            "process":process,
            "status":202,
            "message":"accepted"
        })
                    
    else:
        _tpu.status = False
        _tpu.model = ""
        _tpu.type_step = ""
        _tpu.pid_model = ""
        _model.status = 'stopped'
        db.session.merge(_tpu)
        db.session.merge(_model)
        db.session.commit()
        return jsonify({
            "status":404,
            "message":"no found process"
        })

@app.route('/performance/model/',methods=['POST']) 
async def performance_model():
    model = request.json['model']
    _model = Model.query.filter_by(model=model).first()
    if(_model is None):
        return  jsonify({"status":"404",
                    "message": "no found"})
    _cpu,_men = utils.median_cpu_men_by_model(_model.model)
    return jsonify({"status":"200",
                    "cpu":_cpu,
                    "mem":_men,
                    "message": "median performance of "+_model.model+" training"})

@app.route('/performance/',methods=['POST']) 
async def performance():
    _cpu,_men = utils.median_cpu_men()
    return jsonify({"status":"200",
                    "cpu":_cpu,
                    "mem":_men,
                    "message": "median performance"})  
                    
@app.route('/performancefind/',methods=['POST']) 
async def performance_by():
    _type_step = request.json['type_step']
    _cpu,_men = utils.median_cpu_men_by_type(_type_step)
    return jsonify({"status":"200",
                    "cpu":_cpu,
                    "mem":_men,
                    "message": "median performance"})  


if __name__ == '__main__':
    if(os.path.exists('tmp_2') and os.path.exists('tmp_2/models')):
        print(" * Folder tmp already created")
    else:
        tmp_path = os.path.join(dataset.ROOT_DIR,'tmp_2')
        models_path = os.path.join(dataset.ROOT_DIR+'/tmp_2','models')
        os.mkdir(tmp_path)
        os.mkdir(models_path)
        print(" * Folder tmp created")
    if(os.path.exists('jaxnerf/db/datatrain.db')):
        print(" * DataBase already created")
    else:
        print(" * DataBase created")
        init()
    _tpu = Tpu.query.filter_by(acelerator="v3-8").first()
    if(_tpu is None):
        #try:
        path = os.getenv('KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS')
        if(path):
            path = path.split(':')
            url = 'http://'+path[1][2:]+':8475/requestversion/tpu_driver_nightly'
            reqq = requests.post(url)
        if(reqq.status_code == 200):
            print(" * TPU node conected " +_tpu.type+" "+_tpu.acelerator)
        else:
            print(" * TPU node disconected " +_tpu.type+" "+_tpu.acelerator)
        accelerator_type ="v3-8"
        #accelerator_type =requests.get('http://metadata.google.internal/computeMetadata/v1/instance/attributes/accelerator-type',headers={'Metadata-Flavor': 'Google'}).text
        _tpu  = Tpu(
                    type="VM",
                    acelerator=accelerator_type,
                    cores="8",
                    tpu_mem="128",
                    mem="365",
                    cpu="96",
                    status =False
            )
        db.session.add(_tpu)
        db.session.commit()
        print(" * TPU profile created " +_tpu.type+" "+_tpu.acelerator)
        # except requests.exceptions.ConnectionError:
        #     accelerator_type = "node"
        #     _tpu  = Tpu(
        #             type="Node",
        #             acelerator="v3-8",
        #             cores="8",
        #             tpu_mem="128",
        #             mem="40",
        #             cpu="10",
        #             status =False
        #         )
        #     db.session.add(_tpu)
        #     db.session.commit()
        #     print(" * TPU profile created " +_tpu.type+" "+_tpu.acelerator)
    else:
        print(" * TPU profile already created " +_tpu.type+" "+_tpu.acelerator)
    
    app.debug = True
    app.run(host = '0.0.0.0',port= 3000)