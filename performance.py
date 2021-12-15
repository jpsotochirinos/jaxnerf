from jaxnerf.db.db import Model, Performance,Tpu, db
import psutil

db.session.commit()
_tpu = Tpu.query.filter_by(acelerator="v3-8").first()
while _tpu.status:
    db.session.commit()
    procObjList = [procObj for procObj in psutil.process_iter() if int(_tpu.pid_model) == procObj.pid ]
    print(procObjList)
    if (len(procObjList)>0):
        obj = [elem for elem in procObjList]
        if(obj[0].status()!="zombie"):
            try:
                _performance = Performance(
                    model =_tpu.model,
                    cpu_percent=psutil.cpu_percent(4),
                    mem_percent=psutil.virtual_memory()[2],
                    type_step = _tpu.type_step
                )
                print(_performance)
                db.session.add(_performance)
                db.session.commit()
            except psutil.NoSuchProcess:
                print("except")
                db.session.commit()
                _model = Model.query.filter_by(model=_tpu.model).first()
                if _model.status != 'stopped':
                    _model.status = "interrupted"
                    _tpu.status = False
                    _tpu.model = ""
                    _tpu.type_step = ""
                    _tpu.pid_model = ""
                    db.session.merge(_model)
                    db.session.merge(_tpu)
                    db.session.commit()
        else:
            print("second else")
            db.session.commit()
            _model = Model.query.filter_by(model=_tpu.model).first()
            if _model.status != 'stopped':
                _model.status = "interrupted"
                _tpu.status = False
                _tpu.model = ""
                _tpu.type_step = ""
                _tpu.pid_model = ""
                db.session.merge(_model)
                db.session.merge(_tpu)
                db.session.commit()
    else:
        print("first else")
        break