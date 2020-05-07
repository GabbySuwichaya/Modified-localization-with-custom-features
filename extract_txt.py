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

def recover_query_poses(paths, args):
    print('Recovering query poses...')

    if not os.path.isdir(paths.final_txt_model_path):
        os.mkdir(paths.final_txt_model_path)

    # Convert the model to TXT.
    subprocess.call([os.path.join(args.colmap_path, 'colmap'), 'model_converter',
                     '--input_path', paths.final_model_path,
                     '--output_path', paths.final_txt_model_path,
                     '--output_type', 'TXT'])
    
    # Recover query names.

    query_image_list_path_day = os.path.join(args.dataset_path, 'queries/day_time_queries_with_intrinsics.txt')
    query_image_list_path_night = os.path.join(args.dataset_path, 'queries/night_time_queries_with_intrinsics.txt')
    
    with open(query_image_list_path_day) as f_day:
        raw_queries_day = f_day.readlines()

    with open(query_image_list_path_night) as f_night:
        raw_queries_night = f_night.readlines()
    
    raw_queries = raw_queries_day + raw_queries_night
 
    query_names = set()
    for raw_query in raw_queries:
        raw_query = raw_query.strip('\n').split(' ')
        query_name = raw_query[0]
        query_names.add(query_name)

    with open(os.path.join(paths.final_txt_model_path, 'images.txt')) as f:
        raw_extrinsics = f.readlines()

    f = open(paths.prediction_path, 'w')

    # Skip the header.
    for extrinsics in raw_extrinsics[4 :: 2]:
        extrinsics = extrinsics.strip('\n').split(' ')

        image_name = extrinsics[-1]

        if image_name in query_names:
            # Skip the IMAGE_ID ([0]), CAMERA_ID ([-2]), and IMAGE_NAME ([-1]).
            f.write('%s %s\n' % (image_name.split('/')[-1], ' '.join(extrinsics[1 : -2])))

    f.close()


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
    paths.final_model_path = os.path.join(args.dataset_path, 'sparse-%s-final' % args.method_name)
    paths.final_txt_model_path = os.path.join(args.dataset_path, 'sparse-%s-final-txt' % args.method_name)
    paths.prediction_path = os.path.join(args.dataset_path, 'Aachen_eval_[%s].txt' % args.method_name) 
    
    recover_query_poses(paths, args)
