import os.path as osp
import time
import json
import re

import numpy as np

from models.base_model import BaseVLM
from .videollama2 import model_init
from .videollama2 import depic_model_infer
# from .videollama2.utils import disable_torch_init


class Model(BaseVLM):
    def __init__(self, args, config, run_output_dir):
        super().__init__(args, config, run_output_dir, backend="decord")
        self.model, self.processor, self.tokenizer = model_init(
            model_path=self.config["variant"], device="cuda:0", use_depic=True
        )

    def run_video(self, vids_comb_id, questions):
        q0 = next(iter(questions.values()))

        # Truncate video inputs to max_video_length_seconds
        input_truncated = False
        if len(q0["inputs"]) > self.config["max_video_length_seconds"]:
            q0["inputs"] = dict(
                list(q0["inputs"].items())[: self.config["max_video_length_seconds"]]
            )
            input_truncated = True

        (
            input_keys,
            video_nps,
            temporal_divisor,
            loaded_video_seconds,
            total_video_seconds,
        ) = self.load_videos(q0)
        narration = self.format_narrations(q0)

        # input_keys: ['video 1', 'image 2', ...]
        # video_paths: array of decord reads, one per video, currently as numpy
        # temporal_divisor: temporal downsampling factor. E.g. 2 if we need to sample one out of every two frames.
        # loaded_video_seconds: total seconds of all loaded videos
        print([v.shape for v in video_nps])

        # Hack to handle multiple video input: concatenate all videos into one and mention it in prompt
        if len(video_nps) > 1:
            if input_truncated:
                # Few videos have been truncated due to input length limit.
                multivideo_prompt_tag = f"{len(video_nps)} input videos have been concatenated together and the remaining videos have been truncated due to input length limit."
            else:
                multivideo_prompt_tag = (
                    f"{len(video_nps)} input videos have been concatenated together."
                )
            video_nps = [np.concatenate(video_nps, axis=0)]
        else:
            multivideo_prompt_tag = ""

        video_tensors = [self.processor["video"](v) for v in video_nps]

        contents = []  # ["{video_id}: ", video array, "{video_id}: ", video array, ... "{narration}", ...]. But video_id is not used in inference.
        for i in range(len(video_tensors)):
            contents.append(f"{input_keys[i]}: ")
            contents.append(video_tensors[i])

        if narration:
            contents.append(narration)

        results = {}
        for q_id, question in questions.items():
            q, correct_idx = self.formulate_question(
                question=question, temporal_divisor=temporal_divisor
            )
            start = time.time()

            if multivideo_prompt_tag:
                q = f"{multivideo_prompt_tag} {q}"

            input = [*contents, q]
            response = depic_model_infer(input, self.model, self.tokenizer, self.config)
            pattern = r"\([A-Z]\)"
            match = re.search(pattern, response)
            match = match.group() if match else ""
            print(f"{response}, {match}")

            answer = self.parse_response(match, len(question["choices"]))

            stop = time.time()
            total_time = stop - start
            entry = {
                "id": q_id,
                "time": total_time,
                "answer": answer,
                "correct": 1.0 if answer == correct_idx else 0.0,
                "input_total_seconds": total_video_seconds,
                "input_subsampled_seconds": loaded_video_seconds,
            }
            print(q)
            print(entry)

            with open(osp.join(self.run_output_dir, f"{q_id}.json"), "w") as f:
                json.dump(entry, f)
            results[q_id] = entry

        return results

    def run_eval(self):
        results = {}

        for vids_comb_id, questions in self.questions_by_vid.items():
            if self.args.check_input_only:
                self.run_video_check_input(vids_comb_id, questions)

            else:
                vid_results = self.run_video(vids_comb_id, questions)
                for k, v in vid_results.items():
                    results[k] = v

        return results
