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

def blob_to_array(blob):
    return np.frombuffer(blob, dtype=np.float64)

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


def get_camera_intrinsics_from_databasefile(paths, args ):
    # Connect to the database.
    connection = sqlite3.connect(paths.database_path)
    cursor = connection.cursor()

    # Recover database images and ids. 
    cameras_database = {}
    cursor.execute("SELECT camera_id, model, width, height, params, prior_focal_length FROM cameras;")
    for row in cursor: 
        cameras_database_temp = {}
        cameras_database_temp['camera_id'] = row[0]
        cameras_database_temp['model']     = row[1]
        cameras_database_temp['size']      = np.array([row[2],row[3]]) 
        cameras_database_temp['params']    = blob_to_array(row[4]) 
        cameras_database_temp['prior_focal_length']  = row[5]
        cameras_database[row[0]] = cameras_database_temp

    # Close the connection to the database.
    cursor.close()
    connection.close()

    return cameras_database
 

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

def Check_camera_intrinsics(cameras, camera_parameters, cameras_database):
    query_not_the_same = {}
    db_not_the_same    = {}
    num_db = 0
    num_query = 0
    for image_name in images.keys(): 
        camera_id        = cameras[image_name] 
        try:
            camera_param_txt = camera_parameters[image_name].intrinsics
        except:
            if "db" in image_name:
                print("Database image name: %s not exist in database_intrinsics.txt" % image_name)
            elif "query" in image_name:
                print("Query image name   : %s not exist in day-night_time_queries_with_intrinsics.txt" % image_name)
        
        try:
            camera_param_database = cameras_database[camera_id]['params']
        except:
            print("Image: %s not exist in database.db" % image_name)

        if "db" in image_name : 
            if not((camera_param_txt[2:] == camera_param_database).all()):   
                print("Database IMG: %s with different INTRINSIC PARAM" % image_name)
                db_ = {} 
                db_["param_txt"] = camera_param_txt
                db_["param_database"] = np.concatenate((cameras_database[camera_id]['size'],cameras_database[camera_id]['params']))
                db_not_the_same[image_name] = db_ 
                num_db = num_db +1

        elif "query" in image_name : 
            if not((camera_param_txt[2:] == camera_param_database).all()):   
                print("Query IMG: %s with different INTRINSIC PARAM" % image_name)
                query_ = {} 
                query_["param_txt"] = camera_param_txt
                query_["param_database"] = np.concatenate((cameras_database[camera_id]['size'],cameras_database[camera_id]['params']))
                query_not_the_same[image_name] = query_ 
                num_query = num_query +1
    print("Summary....")
    print("Database IMG:  #%5d images with different INTRINSIC PARAM from database_intrinsics.txt" %  num_db)
    print("Query IMG:     #%5d images with different INTRINSIC PARAM from day/night_time_queries_with_intrinsics.txt" %  num_query)
    return query_not_the_same

 

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
    paths.features_path = os.path.join(args.dataset_path, args.method_name)
    paths.reference_model_path       = os.path.join(args.dataset_path, '3D-models')
    paths.reference_model_query_path = os.path.join(args.dataset_path, 'queries')
 
    paths.empty_model_path = os.path.join(args.dataset_path, 'sparse-%s-empty' % args.method_name) 
     
    # Create a copy of the dummy database. 
    
    # Reconstruction pipeline. 
    images, cameras   = recover_database_images_and_ids(paths, args)
    camera_parameters = preprocess_reference_model(paths, args)
    cameras_database  = get_camera_intrinsics_from_databasefile(paths, args )
    query_not_the_same = Check_camera_intrinsics(cameras, camera_parameters, cameras_database) 