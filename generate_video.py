import argparse
import os
import json
import signal
import subprocess
import sys
import time
import imageio
from PIL import Image
import numpy as np


processes = []

def signal_handler(sig, frame):
    for p in processes:
        p.kill()
        p.wait()

signal.signal(signal.SIGINT, signal_handler)

def get_co3d_list():
    co3d_list_rel_path = "dataloader/co3d_lists/co3d_list_good.json"
    with open(co3d_list_rel_path) as fp:
        co3d_list = json.load(fp)
    return co3d_list

def clear_scene_render_dir(scene, render_path="render"):
    co3d_list = get_co3d_list()
    cls = co3d_list[scene]
    scene_render_dir = os.path.join(render_path, cls, scene)
    process = subprocess.run(['rm', '-r', scene_render_dir])
    return process.returncode

def generate_images(scene, gpu_id, render_path, render_strategy, do_render):
    cmd_str = f"python3 -m run --gpu_id {gpu_id} --ginc configs/PeRFception-v1-1.gin --scene_name {scene} --render_path {render_path} --render_strategy {render_strategy}"
    cmd = cmd_str.split()
    cmd += ['--no-render'] if not do_render else []
    print(f"Generating images for scene {scene}...")
    process = subprocess.run(cmd, stdout=subprocess.DEVNULL)
    print(f"Scene {scene} complete")
    return process.returncode

def generate_video(scene, render_path="render"):
    asset = "fgbg"

    co3d_lists = get_co3d_list()

    cls_list = []
    if scene not in co3d_lists:
        print(f"SCENE {scene} NOT IN CO3D_LISTS - ABORTING")
        return
    cls = co3d_lists[scene]
    print(f"Processing {cls}, {scene}")
    scene_path = os.path.join(render_path, cls, scene)
    asset_path = os.path.join(scene_path, asset)
    img_path_list = os.listdir(asset_path)
    img_path_list = sorted(img_path_list)
    img_list = []
    for img in img_path_list:
        if img.endswith(".mp4"):
            continue
        imgpath = os.path.join(asset_path, img)
        img_list.append(np.asarray(Image.open(imgpath)))
    imageio.mimwrite(os.path.join(asset_path, f"{scene}_{asset}.mp4"), img_list)
    print(f"Completed processing {cls}, {scene}")
    return

def generate_all_scenes(devices=[1,2,3,4,5,6,7]):
    co3d_lists = get_co3d_list()
    scenes = list(co3d_lists.keys())
    num_devices = len(devices)
    scenes_per_device = len(scenes) // num_devices
    scene_dist = [scenes[i*scenes_per_device:(i+1)*scenes_per_device] for i in range(num_devices)]
    for i in range(len(scenes) % num_devices):
        idx = num_devices * scenes_per_device + i
        scene_dist[i].append(scenes[idx])

    for i in range(num_devices):
        cmd = ["python", "generate_video.py", "--gpu_id", str(devices[i])] + scene_dist[i]
        process = subprocess.Popen(cmd)
        processes.append(process)

    for p in processes:
        while p.poll() == None:
            time.sleep(0.5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("scenes", nargs="*")
    parser.add_argument("-n", "--no-generate", dest="generate", action="store_false")
    parser.add_argument("-g", "--gpu_id", type=int, default=0)
    parser.add_argument("--all", action="store_true", default=False)
    parser.add_argument("--render_path", type=str, default="render")
    parser.add_argument("-s", "--render_strategy", type=str, default="canonical")
    parser.add_argument("--no-render", dest="render", action="store_false")

    args = parser.parse_args()

    if args.all:
        generate_all_scenes()
        exit()

    for scene in args.scenes:
        return_code = 0
        if args.generate:
            return_code = clear_scene_render_dir(scene, render_path=args.render_path)
            try:
                return_code = generate_images(scene,
                                              args.gpu_id,
                                              args.render_path,
                                              args.render_strategy,
                                              args.render)
            except:
                return_code = -1

        if return_code == 0:
            generate_video(scene, render_path=args.render_path)
