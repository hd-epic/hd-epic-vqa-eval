import os
import os.path as osp
from torch.utils.data import Dataset

import numpy as np
import decord

from utils.questions import load_questions


def DictCollate(batch):
    """
    Returns the batch as is - just a list of whatever the input is.
    The standard collate function will try to merge tensors between batches, ignore empty lists and merge within dicts, which we don't want.
    This format may be preferable for API models, but can change later if required.
    """
    return batch


class VQADataset(Dataset):
    def __init__(self, args, config, output_dir=None):
        self.args = args
        self.config = config
        self.questions = load_questions(args, config)

        self.idx_2_key = {i: k for i, k in enumerate(self.questions.keys())}

        # remove cached items from idx_2_key if they exist in the output directory
        if args.cached:
            cache_ids = [osp.splitext(f)[0] for f in os.listdir(output_dir)]
            self.idx_2_key = {
                i: k for i, k in enumerate(self.questions.keys()) if k not in cache_ids
            }

        # decord.bridge.set_bridge('torch')

    def __len__(self):
        return len(self.idx_2_key)

    def load_video(self, video_fn, start_s=-1, stop_s=-1, n_frames=16, resolution=224):
        """
        Loads a video from a file and returns a list of frames.
        """
        vr = decord.VideoReader(
            video_fn, ctx=decord.cpu(0), width=resolution, height=resolution
        )

        start_frame = int(start_s * vr.get_avg_fps()) if start_s != -1 else 0
        stop_frame = int(stop_s * vr.get_avg_fps()) if stop_s != -1 else (len(vr) - 1)

        sample_idxs = np.linspace(start_frame, stop_frame, n_frames, dtype=int)
        frames = vr.get_batch(sample_idxs)

        timestamps_sec = sample_idxs / vr.get_avg_fps()

        return frames, timestamps_sec

    def load_video_from_question(self, question):
        vid = question["video_id"]
        assert len(vid) == 1, "Only single video questions supported for now"
        vid = vid[0]
        start_s = question["start_s"][0] if "start_s" in question else -1
        stop_s = question["stop_s"][0] if "stop_s" in question else -1
        n_frames = self.config["n_frames"]
        resolution = self.config["resolution"]

        video_fn = osp.join(self.args.videos_dir, f"{vid}.mp4")

        frames, timestamps_sec = self.load_video(
            video_fn, start_s, stop_s, n_frames, resolution
        )
        return [frames], [timestamps_sec]

    def __getitem__(self, idx):
        """
        Returns a dictionary with the following keys:
        - question_id: int
        - incorrect_answers: list of strings
        - correct_answers: list of strings
        - images which make up the video: list of t h w c numpy arrays (as we may need multiple videos).
        - timestamps_sec: list of lists of frame timestamps in seconds (as we may need multiple videos).
        """
        key = self.idx_2_key[idx]
        q = self.questions[key]

        return_dict = {}
        return_dict["question_id"] = key
        return_dict["incorrect_answers"] = q["incorrect"]
        return_dict["correct_answers"] = q["correct"]

        if self.config["mode"] == "visual_text":
            return_dict["images"], return_dict["timestamps_sec"] = (
                self.load_video_from_question(q)
            )
        else:
            return_dict["images"] = []
            return_dict["timestamps_sec"] = []

        return return_dict
