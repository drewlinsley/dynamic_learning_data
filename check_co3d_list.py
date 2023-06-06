import os
import json

co3d_list_path = "dataloader/co3d_lists/co3d_list.json"
with open(co3d_list_path) as fp:
    co3d_list = json.load(fp)

scenes = list(co3d_list.keys())
scenes = scenes

data_path = "/media/data_cifs/projects/prj_video_imagenet/PeRFception/data/co3d/"

#toytruck/403_53249_103858/../frame_annotations.jgz
good_paths = {}
bad_paths = {}
for scene in scenes:
    category = co3d_list[scene]
    path = os.path.join(data_path, category, scene)
    d = good_paths
    if not os.path.exists(path):
        d = bad_paths
    d[scene] = category

with open("dataloader/co3d_lists/co3d_list_good.json", "w") as fp:
    json.dump(good_paths, fp, indent=4)

with open("dataloader/co3d_lists/co3d_list_bad.json", "w") as fp:
    json.dump(bad_paths, fp, indent=4)
print(f"good: {len(good_paths.keys())}, bad: {len(bad_paths.keys())}")
