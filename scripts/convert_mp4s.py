import os
import os.path as osp
import subprocess
import glob as glob
import random

out_res = 224
out_fps = 1

base_dir = "/user/work/tp8961/eka_scratch/new_mp4s"
new_dir = f"/user/work/tp8961/eka_scratch/rgb_{out_res}_{out_fps}_vig"


def extract_single_video(vid_name):
    # create new folder
    mp4_name = osp.split(vid_name)[-1]
    vid_folder = "/".join(osp.split(vid_name)[:-1])

    out_folder = osp.join(new_dir, vid_folder)
    if not osp.exists(out_folder):
        os.makedirs(out_folder, exist_ok=True)

    out_path = osp.join(out_folder, mp4_name)

    if osp.exists(out_path):
        return

    try:
        call_string = f"ffmpeg -i {osp.join(base_dir, vid_name)} -vf scale={out_res}:{out_res} -q:v 5 -r {out_fps} {out_path}"
        print(call_string)
        subprocess.run(call_string, shell=True, check=True)
        print("Done with {}".format(vid_name))

    except subprocess.CalledProcessError as e:
        print(e.output)


# get list of videos
orig_videos = glob.glob(osp.join(base_dir, "*/*.mp4"))
# shuffle orig_videos
random.shuffle(orig_videos)


for vid in orig_videos:
    vid_name = "/".join(vid.split("/")[-2:])
    extract_single_video(vid_name)
