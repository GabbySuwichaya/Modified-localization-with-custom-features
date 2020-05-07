# I have not yet tested this python script.... 

import argparse

import numpy as np

import os, pdb

import shutil

import subprocess

import sqlite3

import torch

import types

from tqdm import tqdm

from matchers import mutual_nn_matcher

from camera import Camera

from utils import quaternion_to_rotation_matrix, camera_center_to_translation

import sys
IS_PYTHON3 = sys.version_info[0] >= 3

def array_to_blob(array):
    if IS_PYTHON3:
        return array.tostring()
    else:
        return np.getbuffer(array)

def recover_database_images_and_ids(paths, args):
    # Connect to the database.
    connection = sqlite3.connect(paths.database_path)
    cursor = connection.cursor()

    # Recover database images and ids.
    images = {}
    cameras = {}
    cursor.execute("SELECT name, image_id, camera_id FROM images;")
    for row in cursor:
        images[row[0]] = row[1]
        cameras[row[0]] = row[2]

    # Close the connection to the database.
    cursor.close()
    connection.close()

    return images, cameras


def preprocess_reference_model(paths, args):
    print('Preprocessing the reference model...')
    
    # Recover intrinsics.
    with open(os.path.join(paths.reference_model_path, 'database_intrinsics.txt')) as f:
        raw_intrinsics = f.readlines()
    
    camera_parameters = {}

    for intrinsics in raw_intrinsics:
        intrinsics = intrinsics.strip('\n').split(' ')
        
        image_name = intrinsics[0]
        
        camera_model = intrinsics[1]

        intrinsics = [float(param) for param in intrinsics[2 :]]

        camera = Camera()
        camera.set_intrinsics(camera_model=camera_model, intrinsics=intrinsics)

        camera_parameters[image_name] = camera

    with open(os.path.join(paths.reference_model_query_path, 'day_time_queries_with_intrinsics.txt')) as f_day_query:
        raw_day_intrinsics_query = f_day_query.readlines() 

    for intrinsics in raw_day_intrinsics_query:
        intrinsics = intrinsics.strip('\n').split(' ')
        
        image_name = intrinsics[0]
        
        camera_model = intrinsics[1]

        intrinsics = [float(param) for param in intrinsics[2 :]]

        camera = Camera()
        camera.set_intrinsics(camera_model=camera_model, intrinsics=intrinsics)

        camera_parameters[image_name] = camera
 
    with open(os.path.join(paths.reference_model_query_path, 'night_time_queries_with_intrinsics.txt')) as f_night_query:
        raw_night_intrinsics_query = f_night_query.readlines() 

    for intrinsics in raw_night_intrinsics_query:
        intrinsics = intrinsics.strip('\n').split(' ')
        
        image_name = intrinsics[0]
        
        camera_model = intrinsics[1]

        intrinsics = [float(param) for param in intrinsics[2 :]]

        camera = Camera()
        camera.set_intrinsics(camera_model=camera_model, intrinsics=intrinsics)

        camera_parameters[image_name] = camera


    
    # Recover poses.
    with open(os.path.join(paths.reference_model_path, 'aachen_cvpr2018_db.nvm')) as f:
        raw_extrinsics = f.readlines()

    # Skip the header.
    n_cameras = int(raw_extrinsics[2])
    raw_extrinsics = raw_extrinsics[3 : 3 + n_cameras]

    for extrinsics in raw_extrinsics:
        extrinsics = extrinsics.strip('\n').split(' ')

        image_name = extrinsics[0]

        # Skip the focal length. Skip the distortion and terminal 0.
        qw, qx, qy, qz, cx, cy, cz = [float(param) for param in extrinsics[2 : -2]]

        qvec = np.array([qw, qx, qy, qz])
        c = np.array([cx, cy, cz])
        
        # NVM -> COLMAP.
        t = camera_center_to_translation(c, qvec)

        camera_parameters[image_name].set_pose(qvec=qvec, t=t)
    
    return camera_parameters

    
def Update_camera_params( images, cameras, camera_parameters, paths, args):
    # Connect to the database.
    connection = sqlite3.connect(paths.database_path)
    cursor = connection.cursor() 
    cursor.execute("DELETE FROM cameras;")
    connection.commit()
    
    # Recover database images and ids. 
    
    for key in images.keys():  
        
        if camera_parameters.get(key, -1) == -1:
            print("Skip camera intrinsics parameters setting for %s" % key)
            continue
        camera_size = camera_parameters[key].camera_model[:2]
        param_array = np.asarray(camera_parameters[key].camera_model[2:-1], np.float64) 
            
        
        model = 2
        prior_focal_length = 0
        
        cursor.execute("INSERT INTO cameras(camera_id, model, width, height, params, prior_focal_length) VALUES(?, ?, ?, ?, ?, ?);", (key, model, camera_size[0], camera_size[1], array_to_blob(param_array), prior_focal_length))
  
        connection.commit()
         
    # Close the connection to the database.
    cursor.close()
    connection.close()   


def Update_database_images_and_ids( images, cameras, camera_parameters, paths, args):
    
    print("Update_database_images_and_ids ..... ")
    
    # Connect to the database.
    connection = sqlite3.connect(paths.database_path)
    cursor = connection.cursor() 
    cursor.execute("DELETE FROM images;")
    connection.commit()
    new_cameras_ids = []
    new_image_names = []
    # Recover database images and ids. 
    for key, value in images.items():


        images_name_path = key  
        images_id = value 
        camera_id = cameras[key]

        if not("db/" in key):
               
            if camera_parameters.get(key, -1) == -1:
                print("[Warning] Skip camera extrinsic parameters setting for %s" % key)
                continue
            
            print("Query images : %s insert 0" % (key))
            prior_q = np.zeros(4)
            prior_t = np.zeros(3)
 
        else: 
            prior_q   = camera_parameters[key].qvec
            prior_t   = camera_parameters[key].t  
            
        new_cameras_ids.append(camera_id)
        new_image_names.append(images_name_path)
         
        cursor.execute("INSERT INTO images(image_id, name, camera_id, prior_qw, prior_qx, prior_qy, prior_qz, prior_tx, prior_ty, prior_tz ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?);", (images_id, images_name_path, camera_id, prior_q[0], prior_q[1], prior_q[2],
             prior_q[3], prior_t[0], prior_t[1], prior_t[2]))  
        connection.commit()
         
    # Close the connection to the database.
    cursor.close()
    connection.close() 
    
    return new_image_names, new_cameras_ids

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', required=True, help='Path to the dataset')
    parser.add_argument('--colmap_path', required=True, help='Path to the COLMAP executable folder')
    parser.add_argument('--method_name', required=True, help='Name of the method')
    args = parser.parse_args()

    # Torch settings for the matcher.
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # Create the extra paths.
    paths = types.SimpleNamespace()
    paths.dummy_database_path = os.path.join(args.dataset_path, 'database.db')
    paths.database_path = os.path.join(args.dataset_path, args.method_name + '.db')
    paths.image_path    = os.path.join(args.dataset_path, 'images', 'images_upright')
    paths.features_path = os.path.join(args.dataset_path, args.method_name)
    paths.reference_model_path       = os.path.join(args.dataset_path, '3D-models')
    paths.reference_model_query_path = os.path.join(args.dataset_path, 'queries')
 
    paths.empty_model_path = os.path.join(args.dataset_path, 'sparse-%s-empty' % args.method_name) 
     
    # Create a copy of the dummy database.
    if os.path.exists(paths.database_path):
        raise FileExistsError('The database file already exists for method %s.' % args.method_name)
    shutil.copyfile(paths.dummy_database_path, paths.database_path)
    
    # Reconstruction pipeline.
    camera_parameters = preprocess_reference_model(paths, args)
    images, cameras   = recover_database_images_and_ids(paths, args)
    Update_camera_params(images, cameras, camera_parameters, paths, args) 
    Update_database_images_and_ids(images, cameras, camera_parameters, paths, args) 