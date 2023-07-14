import argparse
import os
import json
import signal
import shutil
import subprocess
import sys
import torch
import time
import numpy as np
import multiprocessing as mp
from  pytorch_fid.fid_score import calculate_fid_given_paths

processes = []

def signal_handler(sig, frame):
    for p in processes:
        p.kill()
        p.wait()

signal.signal(signal.SIGINT, signal_handler)

def get_co3d_list():
    co3d_list_rel_path = "dataloader/co3d_lists/co3d_list.json"
    with open(co3d_list_rel_path) as fp:
        co3d_list = json.load(fp)
    return co3d_list

def generate_scenes_for_categories(categories, flags, devices=[1,2,3,4,5,6,7]):
    co3d_lists = get_co3d_list()
    scenes = []
    for scene, scene_category in co3d_lists.items():
        if scene_category in categories or "all" in categories:
            scenes.append(scene)
    num_devices = len(devices)
    scenes_per_device = len(scenes) // num_devices
    scene_dist = [scenes[i*scenes_per_device:(i+1)*scenes_per_device] for i in range(num_devices)]
    for i in range(len(scenes) % num_devices):
        idx = num_devices * scenes_per_device + i
        scene_dist[i].append(scenes[idx])

    for i in range(num_devices):
        cmd = ["python", __file__, "--gpu_id", str(devices[i])] + scene_dist[i]
        cmd += flags
        process = subprocess.Popen(cmd)
        processes.append(process)

    for p in processes:
        while p.poll() == None:
            time.sleep(0.5)

def process_scenes(scenes, rank, path_prefix):
    co3d_lists = get_co3d_list()
    batch_size = 50
    dims = 2048

    for idx, scene in enumerate(scenes):
        print(f"Process {rank}, scene {idx}/{len(scenes)} ({100*idx/len(scenes):.2f}%)")
        category = co3d_lists[scene]
        scene_path = os.path.join(path_prefix, category, scene)
        dirs = list(os.scandir(scene_path))
        image_dir = "images" if "images" in dirs else "fgbg"
        image_path = os.path.join(scene_path, image_dir)
        output_path = os.path.join(scene_path, f"{category}_{scene}_fidstats.npz")
        module = "pytorch_fid"

        cmd = ["python", "-m", module, image_path, output_path]
        flags = ["--device", f"cuda:{rank}", "--num-workers", "4", "--save-stats"]
        cmd += flags
        _ = subprocess.check_output(cmd).decode("utf-8")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("scenes", nargs="*")
    parser.add_argument("-g", "--gpu_id", type=int, default=0)
    parser.add_argument("-c", "--categories", type=str, default=None)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("-v", "--verbose", dest="verbose", action="store_true", default=False)

    args = parser.parse_args()

    scenes = args.scenes

    if args.categories is not None:
        remove = args.scenes + ["-c", args.categories]
        flags = [flag for flag in sys.argv[1:] if flag not in remove]
        categories = args.categories.split(',')
        generate_scenes_for_categories(categories, flags)
        exit()

    process_scenes(scenes, args.gpu_id, args.data_path)
