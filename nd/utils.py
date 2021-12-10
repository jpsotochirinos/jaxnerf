
import psutil
import subprocess
import time
import os
import numpy as np
from jaxnerf.nd.dataset import *

def checkIfProcessRunning(processName):
    for proc in psutil.process_iter():
        try:
            if processName.lower() in proc.name().lower():
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return False
def checkCPU():
    return 100-psutil.cpu_percent(4)

def checkMEM():
    return 100-psutil.virtual_memory()[2]

def checkModelFile(model,only):
    check_folder = os.path.exists(DATA_DIR+model)
    check_database = os.path.exists(DATA_DIR+model+'/database.db')
    check_poses_bounds = os.path.exists(DATA_DIR+model+'/poses_bounds.npy')
    check_sparce_folder = os.path.exists(DATA_DIR+model+'/sparse/')
    check_sparce_0_folder = os.path.exists(DATA_DIR+model+'/sparse/0/')
    check_sparce_0_files = False
    check_images_folder = os.path.exists(DATA_DIR+model+'/images/')
    check_images_files = False
    
    if(check_sparce_0_folder):
        bin_arr = ['cameras.bin','images.bin','points3D.bin','project.ini']
        sparce_bin_files = os.listdir(DATA_DIR+model+'/sparse/0/')
        check_sparce_0_files = (len(sparce_bin_files)==len(bin_arr))
    
    if(check_images_folder and check_sparce_0_files):
        images_arr = np.loadtxt(DATA_DIR+model+'/images.dat', dtype=str)
        images_files = os.listdir(DATA_DIR+model+'/images/')
        check_images_files = (len(images_arr) == len(images_files)) 
    if(
        check_folder and
        check_database and
        check_poses_bounds and
        check_sparce_folder and
        check_sparce_0_folder and 
        check_sparce_0_files and 
        check_images_folder and
        check_images_files
    ):
        return True
    elif only:
        gstuil =[
            'gsutil',
            '-m','cp','-r',
            'gs://nerf-models/models/'+model+'/',
            DATA_DIR
        ]
        (subprocess.check_output(gstuil, universal_newlines=True))
        return(checkModelFile(model,False))
    else:
        return False