import gzip
import json
import os

data_root = "/media/data_cifs/projects/prj_video_imagenet/PeRFception/data"
co3d_path = os.path.join(data_root, "co3d")
annotation_output_path = os.path.join(data_root, "co3d_annotations")
categories = [category for category in os.listdir(co3d_path) if os.path.isdir(os.path.join(co3d_path, category))]
annotations_name = "frame_annotations.jgz"
for category in categories:
    scenes_to_frames = {}
    print(f"Processing category {category}...")
    frame_annotations_path = os.path.join(co3d_path, category, annotations_name)
    if not os.path.exists(frame_annotations_path):
        print(f"Skipping {category} due to missing {annotations_name}")
        continue

    with gzip.open(frame_annotations_path, "r") as fp:
        frame_annotations = json.load(fp)
    
    for frame in frame_annotations:
        scene_name = frame["sequence_name"]
        if scene_name not in scenes_to_frames:
            scenes_to_frames[scene_name] = []
        scenes_to_frames[scene_name].append(frame)

    for scene, frames in scenes_to_frames.items():
        output_path = os.path.join(annotation_output_path, category, scene)
        os.makedirs(output_path, exist_ok=True)
        output_path = os.path.join(output_path, annotations_name)
        with gzip.open(output_path, "w") as fp:
            fp.write(json.dumps(frames).encode('utf-8'))
