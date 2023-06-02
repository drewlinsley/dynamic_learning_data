import argparse
import os
import json
import subprocess
import sys
import imageio
from PIL import Image
import numpy as np


def get_co3d_list():
    co3d_list_rel_path = "dataloader/co3d_lists/co3d_list.json"
    with open(co3d_list_rel_path) as fp:
        co3d_list = json.load(fp)
    return co3d_list

def clear_scene_render_dir(scene):
    render_prefix = "render"
    co3d_list = get_co3d_list()
    cls = co3d_list[scene]
    scene_render_dir = os.path.join(render_prefix, cls, scene)
    return_code = subprocess.call(['rm', '-r', scene_render_dir])
    return return_code

def generate_images(scene):
    cmd_str = "python3 -m run --ginc configs/PeRFception-v1-1.gin --scene_name"
    cmd = cmd_str.split() + [scene]
    return_code = subprocess.call(cmd)
    return return_code

def generate_video(scene):
    render_path = "render"
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("scenes", nargs="+")
    parser.add_argument("--generate", action="store_true", default=True)
    parser.add_argument("-n", "--no-generate", dest="generate", action="store_false")

    args = parser.parse_args()

    for scene in args.scenes:
        return_code = 0
        if args.generate:
            return_code = clear_scene_render_dir(scene)
            return_code = generate_images(scene)

        if return_code == 0:
            generate_video(scene)
