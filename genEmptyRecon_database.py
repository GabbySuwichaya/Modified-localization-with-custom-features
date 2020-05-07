# Generate empty reconstruction (only) for database images .... 

import argparse

import numpy as np

import os, pdb

import shutil

import subprocess

import sqlite3

import torch

import types

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
    connection = sqlite3.connect(paths.dummy_database_path)
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


def generate_empty_reconstruction(images, cameras, camera_parameters, paths, args):
    print('Generating the empty reconstruction...')

    if not os.path.exists(paths.empty_model_path):
        os.mkdir(paths.empty_model_path)
    
    with open(os.path.join(paths.empty_model_path, 'cameras.txt'), 'w') as f:
        for image_name in images:
            image_id = images[image_name]
            camera_id = cameras[image_name]
            try:
                camera = camera_parameters[image_name]
            except:
                continue
            f.write('%d %s %s\n' % (
                camera_id, 
                camera.camera_model, 
                ' '.join(map(str, camera.intrinsics))
            ))

    with open(os.path.join(paths.empty_model_path, 'images.txt'), 'w') as f:
        for image_name in images:
            image_id = images[image_name]
            camera_id = cameras[image_name]
            try:
                camera = camera_parameters[image_name]
            except:
                continue
            f.write('%d %s %s %d %s\n\n' % (
                image_id, 
                ' '.join(map(str, camera.qvec)), 
                ' '.join(map(str, camera.t)), 
                camera_id,
                image_name
            ))

    with open(os.path.join(paths.empty_model_path, 'points3D.txt'), 'w') as f:
        pass



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
    paths.dummy_database_path = os.path.join(args.dataset_path, 'db.db')
     
    paths.image_path = os.path.join(args.dataset_path, 'images', 'images_upright') 
    paths.reference_model_path = os.path.join(args.dataset_path, '3D-models')
    paths.match_list_path = os.path.join(args.dataset_path, 'image_pairs_to_match.txt')
    paths.empty_model_path = os.path.join(args.dataset_path, 'sparse-%s-empty' % args.method_name)  
    
    # Create empty reconstruction...  
    camera_parameters = preprocess_reference_model(paths, args)
    images, cameras = recover_database_images_and_ids(paths, args)
    generate_empty_reconstruction(images, cameras, camera_parameters, paths, args) 
