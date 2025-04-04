import os
import os.path as osp
import numpy as np
import glob

import time
import subprocess
import re

from utils.questions import (
    load_questions,
    group_questions_by_identical_input,
    secs_from_time_str,
)
from utils.narrations import NarrationReader

try:
    from decord import VideoReader, cpu
except:
    print(
        "Decord not installed. Please install decord if your model requires it as a backend."
    )


class BaseVLM:
    def __init__(self, args, config, run_output_dir, backend="ffmpeg"):
        self.args = args
        self.config = config

        self.backend = backend

        questions = load_questions(
            args, config, run_output_dir=run_output_dir, cached=self.args.cached
        )
        self.run_output_dir = run_output_dir
        self.tmp_dir = osp.join(run_output_dir, "tmp")
        os.makedirs(self.tmp_dir, exist_ok=True)

        self.questions_by_vid = group_questions_by_identical_input(questions)

        if "narration" in config["mode"]:
            self.narration_reader = NarrationReader(args, config)
        else:
            self.narration_reader = None

        self.cached_video_lengths = {}

    def parse_response(self, response, n_choices):
        regex_retun = re.search(r"[A-Z]", response)
        if regex_retun is None:
            return -1

        response_char = regex_retun.group(0)[0]
        response_idx = ord(response_char) - ord("A")
        response_idx = response_idx if 0 <= response_idx < n_choices else -1

        return response_idx

    def get_full_video_fn(self, vid_id):
        pid = vid_id.split("-")[0]
        full_video_fn = osp.join(self.args.mp4_dir, pid, vid_id + ".mp4")
        return full_video_fn

    def load_single_video(self, vid_id, start_s, end_s, temporal_divisor):
        full_video_fn = self.get_full_video_fn(vid_id)

        out_fn = osp.join(
            self.tmp_dir, f"{vid_id}_{start_s}_{end_s}_{temporal_divisor}.mp4"
        )

        if self.backend == "ffmpeg":
            assert self.args.input_fps == 1 and temporal_divisor >= 1, (
                "ffmpeg backend only for 1fps videos for gemini for now"
            )

            if start_s == -1 and end_s == -1 and temporal_divisor == 1:
                return full_video_fn

            out_fn = osp.join(
                self.tmp_dir, f"{vid_id}_{start_s}_{end_s}_{temporal_divisor}.mp4"
            )

            if osp.exists(out_fn):
                return out_fn

            if temporal_divisor < 100:
                filter_cmd = f'-filter_complex "[0:v]setpts={1.0 / temporal_divisor}*PTS[v];[0:a]atempo={temporal_divisor}[a]" -map "[v]" -map "[a]"'
            else:
                filter_cmd = f'-filter_complex "[0:v]setpts={1.0 / temporal_divisor}*PTS[v]" -map "[v]" -an'

            if start_s == -1 and end_s == -1:
                start_end_cmd = ""
            else:
                if end_s <= start_s:
                    end_s = start_s + 1
                start_end_cmd = f"-ss {start_s / temporal_divisor} -t {(end_s - start_s) / temporal_divisor}"

            cmd = f"ffmpeg -n -hide_banner -loglevel error -i {full_video_fn} {start_end_cmd} {filter_cmd} {out_fn}"
            subprocess.run(cmd, shell=True, check=True)
            return out_fn

        elif self.backend == "decord":
            vr = VideoReader(full_video_fn, ctx=cpu(0))
            vid_frames = len(vr)
            fps = vr.get_avg_fps()

            if start_s == -1 and end_s == -1:
                start_s = 0
                end_s = int(np.floor(vid_frames / fps))
            elif end_s <= start_s:
                end_s = start_s + 1

            n_frames_to_load = max(1, int((end_s - start_s) / float(temporal_divisor)))
            frame_idxs = np.linspace(
                start_s * fps, end_s * fps, n_frames_to_load
            ).astype(int)
            if frame_idxs[-1] == vid_frames:
                frame_idxs = frame_idxs[:-1]

            frames = vr.get_batch(frame_idxs).asnumpy()
            return frames

        else:
            raise NotImplementedError(f"Backend {self.backend} not implemented.")

    def remove_tmp_videos(self, video_fns=None):
        if video_fns is None:
            video_fns = glob.glob(osp.join(self.tmp_dir, "*.mp4"))

        for fn in video_fns:
            if fn.startswith(self.tmp_dir):
                os.remove(fn)

    # use to work out the temporal subsampling required to fit all videos in the total length supported by a model
    def compute_temporal_divisor(self, vid_ids, start_secs, end_secs):
        vid_lens = []
        for i in range(len(vid_ids)):
            if start_secs[i] == -1 or end_secs[i] == -1:
                secs = self.cached_video_lengths.get(vid_ids[i], None)
                if secs is None:
                    cmd = f"ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {self.get_full_video_fn(vid_ids[i])}"
                    result = subprocess.run(
                        cmd, shell=True, check=True, stdout=subprocess.PIPE
                    )
                    secs = float(result.stdout)
                    self.cached_video_lengths[vid_ids[i]] = secs
                vid_lens.append(secs)
            else:
                vid_lens.append(end_secs[i] - start_secs[i])

        total_len = float(sum(vid_lens))

        if "min_temporal_divisor" in self.config:
            temporal_divisor = float(self.config["min_temporal_divisor"])
        else:
            temporal_divisor = 1.0

        while True:
            subsampled_lengths = [
                max(1, np.ceil(v / temporal_divisor)) for v in vid_lens
            ]
            total_subsampled_length = sum(subsampled_lengths)

            if total_subsampled_length <= self.config["max_video_length_seconds"]:
                break

            if temporal_divisor == 1 and self.args.input_fps == 1:
                temporal_divisor = 2
            elif total_subsampled_length > self.config["max_video_length_seconds"] * 2:
                temporal_divisor = float(np.ceil(temporal_divisor * 1.5))
            else:
                temporal_divisor = float(np.ceil(temporal_divisor * 1.25))

        return temporal_divisor, total_len / temporal_divisor, total_len

    def get_vids_info_from_question(self, q0):
        vid_ids = []
        start_secs = []
        end_secs = []
        input_keys = []
        for k, v in q0["inputs"].items():
            vid_ids.append(v["id"])
            input_keys.append(k)
            if "image" in k:
                t = secs_from_time_str(v["time"])
                start_secs.append(t)
                end_secs.append(t + 1)
            else:
                start = secs_from_time_str(v["start_time"]) if "start_time" in v else -1
                end = secs_from_time_str(v["end_time"]) if "end_time" in v else -1
                start_secs.append(start)
                end_secs.append(end)
        return vid_ids, start_secs, end_secs, input_keys

    def load_videos(self, q0):
        if self.config["mode"] != "visual_text":
            return [], [], 1.0, 0.0, 0.0

        vid_ids, start_secs, end_secs, input_keys = self.get_vids_info_from_question(q0)

        temporal_divisor, loaded_video_seconds, total_video_seconds = (
            self.compute_temporal_divisor(vid_ids, start_secs, end_secs)
        )

        video_fns = []
        for vid_id, start, end in zip(vid_ids, start_secs, end_secs):
            fn = self.load_single_video(vid_id, start, end, temporal_divisor)
            video_fns.append(fn)

        return (
            input_keys,
            video_fns,
            temporal_divisor,
            loaded_video_seconds,
            total_video_seconds,
        )

    def time_from_tag(self, x, temporal_divisor=1.0, question=None):
        seconds = secs_from_time_str(x.groups()[0])
        input_id = x.groups()[1]

        if question == None or "start_time" not in question["inputs"][input_id]:
            input_start_seconds = 0
        else:
            input_start_str = question["inputs"][input_id]["start_time"]
            input_start_seconds = secs_from_time_str(input_start_str)

        seconds = seconds - input_start_seconds

        seconds /= temporal_divisor

        if seconds >= 3600:
            parsed_time = time.strftime("%H:%M:%S", time.gmtime(seconds))
        else:
            parsed_time = time.strftime("%M:%S", time.gmtime(seconds))

        return parsed_time

    def bbox_from_tag(self, x):
        # print(x.groups())
        coords = [float(x) for x in x.groups()]
        coords = [int(x / self.args.dataset_orig_res * 1000) for x in coords]
        return f"({', '.join([str(x) for x in coords])})"

    def format_narrations(self, question):
        if "narration" not in self.config["mode"]:
            return ""

        narr_str = ""
        vid_ids, start_secs, end_secs, input_keys = self.get_vids_info_from_question(
            question
        )
        for i in range(len(vid_ids)):
            narrations = self.narration_reader.get_narrations(
                vid_ids[i], start_secs[i], end_secs[i]
            )
            if len(narrations) == 0:
                continue

            narr_str += f"Subtitles for {input_keys[i]}:"
            for narr_text, time_s in narrations:
                # get time as HH:MM:SS.MSS
                time_str = time.strftime("%H:%M:%S.000", time.gmtime(time_s))
                narr_str += f" <TIME {time_str} {vid_ids[i]}>: {narr_text}"
            narr_str += ". "

        narr_str = self.parse_tags(narr_str)

        return narr_str

    def parse_tags(self, text, temporal_divisor=1.0, question=None):
        time_pattern = r"<TIME\s+([\d:.]+)\s+(.+?)>"

        text = re.sub(
            time_pattern,
            lambda x: self.time_from_tag(
                x, temporal_divisor=temporal_divisor, question=question
            ),
            text,
        )

        bbox_pattern = r"<BBOX\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*>"
        text = re.sub(bbox_pattern, lambda x: self.bbox_from_tag(x), text)
        return text

    def formulate_question(self, question, temporal_divisor=1.0):
        if self.config["answer_only"]:
            text = "Answers: "
        else:
            text = f"Question: {question['question']}. Answers: "

        for idx, choice in enumerate(question["choices"]):
            text += f"({chr(ord('A') + idx)}) {choice}. "
        text += "Correct: "

        text = self.parse_tags(
            text, temporal_divisor=temporal_divisor, question=question
        )

        correct_idx = int(question["correct_idx"]) if "correct_idx" in question else -1

        return text, correct_idx

    def run_video(self, vids_comb_id, questions):
        raise NotImplementedError("This should be implemented in the derived class.")

    def run_eval(self):
        raise NotImplementedError("This should be implemented in the derived class.")
