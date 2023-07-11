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

def clear_scene_render_dir(scene, render_path="render"):
    co3d_list = get_co3d_list()
    cls = co3d_list[scene]
    scene_render_dir = os.path.join(render_path, cls, scene)
    process = subprocess.run(['rm', '-r', scene_render_dir])
    return process.returncode

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

def process_scenes(scenes, rank):
    co3d_lists = get_co3d_list()
    path_prefix = "/media/data_cifs/projects/prj_video_imagenet/PeRFception"
    linear_path_prefix = os.path.join(path_prefix, "render_canonical_linear")
    planar_path_prefix = os.path.join(path_prefix, "render_canonical_planar")
    interp_path_prefix = os.path.join(path_prefix, "render_interp")
    output_path_prefix = os.path.join(path_prefix, "render_canonical_merged")
    batch_size = 50
    dims = 2048

    for idx, scene in enumerate(scenes):
        print(f"Process {rank}, scene {idx}/{len(scenes)} ({100*idx/len(scenes):.2f}%)")
        category = co3d_lists[scene]
        linear_path = os.path.join(linear_path_prefix, category, scene)
        linear_path_fgbg = os.path.join(linear_path, "fgbg")
        planar_path = os.path.join(planar_path_prefix, category, scene)
        planar_path_fgbg = os.path.join(planar_path, "fgbg")
        interp_path = os.path.join(interp_path_prefix, category, scene)
        interp_path_fgbg = os.path.join(interp_path, "fgbg")
        module = "pytorch_fid"
        cmd = ["python", "-m", module, interp_path_fgbg, linear_path_fgbg]
        flags = ["--device", f"cuda:{rank}", "--num-workers", "4"]
        cmd += flags
        output = subprocess.check_output(cmd).decode("utf-8")
        fid_linear = float(output.split()[-1])
        cmd = ["python", "-m", module, interp_path_fgbg, planar_path_fgbg]
        flags = ["--device", f"cuda:{rank}", "--num-workers", "4"]
        cmd += flags
        output = subprocess.check_output(cmd).decode("utf-8")
        fid_planar = float(output.split()[-1])
        print(f"FID values for {category}/{scene}: linear: {fid_linear:.2f}, planar: {fid_planar:.2f}")
        fid_is_planar = fid_planar < fid_linear
        src_dir = planar_path if fid_is_planar else linear_path
        dest_dir = os.path.join(output_path_prefix, category, scene)
        os.makedirs(dest_dir, exist_ok=True)
        _ = subprocess.run(["rm", "-rf", dest_dir])
        _ = subprocess.run(["cp", "-r", src_dir, dest_dir])
        npz_filename = os.path.join(dest_dir, f"{category}_{scene}.npz")
        with np.load(npz_filename) as data:
            d = {}
            d.update(data)
            d['fid_is_planar'] = np.array(fid_is_planar)
            d['fid_score_planar'] = np.array(fid_planar)
            d['fid_score_linear'] = np.array(fid_linear)
            np.savez_compressed(npz_filename, **d)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("scenes", nargs="*")
    parser.add_argument("-n", "--no-generate", dest="generate", action="store_false")
    parser.add_argument("-g", "--gpu_id", type=int, default=0)
    parser.add_argument("-c", "--categories", type=str, default=None)
    parser.add_argument("--render_path", type=str, default="render")
    parser.add_argument("-v", "--verbose", dest="verbose", action="store_true", default=False)

    args = parser.parse_args()

    scenes = args.scenes

    if args.categories is not None:
        remove = args.scenes + ["-c", args.categories]
        flags = [flag for flag in sys.argv[1:] if flag not in remove]
        categories = args.categories.split(',')
        generate_scenes_for_categories(categories, flags)
        exit()

    process_scenes(scenes, args.gpu_id)
