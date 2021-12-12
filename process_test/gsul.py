
import psutil
import subprocess
import time
import os
import numpy as np

model="mode_02_01"
models="mode_02_02asdf"
gs_dir = 'gs://nerf-models/models/'+model+'/*'
gs_status = [
    'gsutil','-q','stat',
    gs_dir]
echo = ['echo','$?']

r = subprocess.call(gs_status)
print(r)
# try:
#     (subprocess.check_output(gs_status, universal_newlines=True))
#     print("yeiii")
# except subprocess.CalledProcessError as e:
#     raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
