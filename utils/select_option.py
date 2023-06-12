import os
from typing import *

import gdown

from dataloader.litdata import LitDataCo3D, LitDataScannet
from model.plenoxel_torch.model import LitPlenoxel, ResampleCallBack

url_co3d_list = "https://drive.google.com/uc?id=1jCDaA41ZddkgPl4Yw2h-XI7mt9o56kb7"

def select_model(
    model_name: str,
    render_path: str,
    do_render: bool,
):  
    return LitPlenoxel(render_path=render_path, do_render=do_render)


def select_dataset(
    dataset_name: str,
    datadir: str, 
    scene_name: str, 
    accelerator: str,
    num_gpus: int,
    num_tpus: int,
    perturb_scale: float,
    perturb_pose: float,
    render_strategy: str,
    do_render: bool,
):
    if dataset_name == "co3d":
        data_fun = LitDataCo3D
        co3d_list_json_path = os.path.join("dataloader/co3d_lists/co3d_list.json")
        if not os.path.exists(co3d_list_json_path):
            gdown.download(url_co3d_list, co3d_list_json_path)
    elif dataset_name == "scannet":
        data_fun = LitDataScannet

    return data_fun(
        datadir=datadir, 
        scene_name=scene_name, 
        accelerator=accelerator,
        num_gpus=num_gpus,
        num_tpus=num_tpus,
        perturb_scale=perturb_scale,
        perturb_pose=perturb_pose,
        render_strategy=render_strategy,
        do_render=do_render,
    )

def select_callback(model_name):

    return [ResampleCallBack()]
