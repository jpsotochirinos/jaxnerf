
import psutil
import subprocess
import time
import os
from os import path
import numpy as np
from jaxnerf.nd.dataset import *
from jaxnerf.db.db import Performance,Model,db

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
    with open(path.join(DATA_DIR+model, "poses_bounds.npy"),"rb") as fp:
      poses_arr = np.load(fp)
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])

    if(check_sparce_0_folder):
        bin_arr = ['cameras.bin','images.bin','points3D.bin','project.ini']
        sparce_bin_files = os.listdir(DATA_DIR+model+'/sparse/0/')
        check_sparce_0_files = (len(sparce_bin_files)==len(bin_arr))
    
    if(check_images_folder and check_sparce_0_files):
        images_arr = poses.shape[-1]
        images_files = os.listdir(DATA_DIR+model+'/images/')
        check_images_files = (images_arr == len(images_files)) 
    check_arr = [
        str(int(check_folder)),
        str(int(check_database)),
        str(int(check_poses_bounds)),
        str(int(check_sparce_folder)),
        str(int(check_sparce_0_folder)), 
        str(int(check_sparce_0_files)),
        str(int(check_images_folder)),
        str(int(check_images_files))
    ]
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
        return True,check_arr
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
        return False,check_arr

def get_models(model):
    gstuil =[
            'gsutil',
            '-m','cp','-r',
            'gs://nerf-models/models/'+model+'/',
            DATA_DIR
        ]
    subprocess.check_output(gstuil, universal_newlines=True)
    checkModelFile(model,False)
    return True

def check_models(model):
    gs_dir = 'gs://nerf-models/models/'+model+'/*'
    gs_status = [
        'gsutil','-q','stat',
        gs_dir]
    r = subprocess.call(gs_status)
    if r!=0:
        return False
    return True
def check_img_size(model):
    list_img = DATA_DIR+model+'/images/'
    images_files = os.listdir(list_img)
    avg_size = 0
    for img in images_files:
        size = os.path.getsize(list_img+img)/1024.0**2
        avg_size += size / len(images_files)
    if(np.floor(avg_size/2.7) == 1 or np.floor(avg_size/2.7) ==0):
        return 0
    else:
        return np.floor(avg_size/2.7)

def minify(model, factors=[], resolutions=[]):
    basedir = DATA_DIR + model + '/'
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return
    
    from shutil import copy
    from subprocess import check_output
    
    imgdir = os.path.join(basedir, 'images')
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    imgdir_orig = imgdir
    
    wd = os.getcwd()
    resizearg = 0
    for r in factors + resolutions:
        if isinstance(r, int):
            name = 'images_{}'.format(r)
            resizearg = '{}%'.format(int(100./r))
        else:
            name = 'images_{}x{}'.format(r[1], r[0])
            resizearg = '{}x{}'.format(r[1], r[0])
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue
            
        print('Minifying', r, basedir)
        
        os.makedirs(imgdir)
        check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)
        
        ext = imgs[0].split('.')[-1]
        args = ' '.join(['mogrify', '-resize', resizearg, '-format', 'png', '*.{}'.format(ext)])
        print(args)
        os.chdir(imgdir)
        check_output(args, shell=True)
        os.chdir(wd)
        
        if ext != 'png':
            check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
            print('Removed duplicates')
        print('Done')
    return resizearg

def median_cpu_men_by_model(model):
    _perf = Performance.query.filter_by(model=model).all()
    if(_perf is None):
        return  0,0
    cpu_list = np.array([perf.cpu_percent for perf in _perf]).astype(np.float64)
    men_list = np.array([perf.mem_percent for perf in _perf]).astype(np.float64)
    median_cpu = np.median(cpu_list) 
    median_men = np.median(men_list) 
    return median_cpu,median_men

def median_cpu_men():
    _perf = Performance.query.all()
    if(_perf is None):
        return  0,0
    cpu_list = np.array([perf.cpu_percent for perf in _perf]).astype(np.float64)
    men_list = np.array([perf.mem_percent for perf in _perf]).astype(np.float64)
    median_cpu = np.median(cpu_list) 
    median_men = np.median(men_list) 
    return median_cpu,median_men

def median_cpu_men_by_type(type_step):
    _perf = Performance.query.filter_by(type_step=type_step).all()
    if(_perf is None):
        return  0,0
    cpu_list = np.array([perf.cpu_percent for perf in _perf]).astype(np.float64)
    men_list = np.array([perf.mem_percent for perf in _perf]).astype(np.float64)
    median_cpu = np.median(cpu_list) 
    median_men = np.median(men_list) 
    return median_cpu,median_men