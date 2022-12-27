import os
import imageio
from PIL import Image
import numpy as np


render_path = "render"
cls_list = os.listdir(render_path)
for cls_name in cls_list: 
    cls_path = os.path.join(render_path, cls_name)
    seq_list = os.listdir(cls_path)
    for seq in seq_list: 
        seq_path = os.path.join(cls_path, seq)
        assets_list = ["fg", "fgbg", "mask"]
        for asset in assets_list:
            asset_path = os.path.join(seq_path, asset)
            img_path_list = os.listdir(asset_path)
            img_path_list = sorted(img_path_list)
            img_list = []
            for img in img_path_list:
                if img.endswith(".mp4"):
                    continue
                imgpath = os.path.join(asset_path, img)
                img_list.append(np.asarray(Image.open(imgpath)))
            imageio.mimwrite(os.path.join(asset_path, f"{seq}_{asset}.mp4"), img_list)
        print(cls_name, seq)
        break
    break

