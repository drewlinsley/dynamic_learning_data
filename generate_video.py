import os
import json
import sys
import imageio
from PIL import Image
import numpy as np

def generate_video(scenes):
    render_path = "render"
    assets_list = ["fgbg"]

    with open("dataloader/co3d_lists/co3d_list.json") as fp:
        co3d_lists = json.load(fp)

    cls_list = []
    for scene in scenes:
        if scene not in co3d_lists:
            print(f"SCENE {scene} NOT IN CO3D_LISTS - SKIPPING")
            continue
        cls = co3d_lists[scene]
        print(f"Processing {cls}, {scene}")
        scene_path = os.path.join(render_path, cls, scene)
        for asset in assets_list:
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


if __name__ == "__main__":
    scenes = sys.argv[1:]
    generate_video(scenes)
