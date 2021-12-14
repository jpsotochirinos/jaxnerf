from jaxnerf.db.db import Model, Performance,Tpu, db
import psutil
import numpy as np
_tpu = Tpu.query.filter_by(acelerator="v3-8").first()

_perf = Performance.query.filter_by(model="mode_02_01").all()


cpu_list = np.array([perf.mem_percent for perf in _perf]).astype(np.float64)
#cpu_list = np.shape(cpu_list)
median = np.median(cpu_list) 
print(cpu_list)
print(median)
