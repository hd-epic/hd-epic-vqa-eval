import os.path as osp
import numpy as np
import decord
import torch
import json
import time
import math
import os
import re

from transformers import MllamaForConditionalGeneration, AutoProcessor
from huggingface_hub import login

from models.base_model import BaseVLM


class Model(BaseVLM):
    def __init__(self, args, config, run_output_dir):
        super().__init__(args, config, run_output_dir)

        # To access the model via transformers, can be done locally by downloading first then changing config["variant"]
        login(token=os.environ["ACCESS_TOKEN"])

        self.model = MllamaForConditionalGeneration.from_pretrained(
            self.config["variant"],
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        # Remove probabilistic parameters for reproducibility
        self.model.generation_config.temperature = None
        self.model.generation_config.top_p = None
        self.processor = AutoProcessor.from_pretrained(self.config["variant"])

        self.system_config = {
            "role": "system",
            "content": {"type": "text", "text": self.config["prompts"]["sys"]},
        }

        if self.config["clear_local_tmp"]:
            self.remove_tmp_videos()

    def parse_response(self, response, n_choices):
        response_char = response.split(
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )[-1][0]

        regex_retun = re.search(r"[A-Z]", response_char)
        if regex_retun is None:
            return -1

        response_char = regex_retun.group(0)[0]
        response_idx = ord(response_char) - ord("A")
        response_idx = response_idx if 0 <= response_idx < n_choices else -1

        return response_idx

    def format_prompt(self, narration, q, video_files):
        if self.config["mode"] != "visual_text":
            input_prompt = [
                self.system_config,
                {"role": "user", "content": [{"type": "text", "text": narration + q}]},
            ]
        else:
            input_content = [{"type": "image"}] * len(video_files)
            input_content.append(
                {
                    "type": "text",
                    "text": self.system_config["content"]["text"] + " " + q,
                }
            )
            input_prompt = [{"role": "user", "content": input_content}]

        return self.processor.apply_chat_template(
            input_prompt, add_generation_prompt=True
        )

    def retrieve_frames(self, video_paths):
        video_files = []

        if self.config["mode"] == "visual_text":
            frames_per_video = max(
                self.config["max_input_frames"] // len(video_paths), 1
            )

            for p in video_paths:
                video = decord.VideoReader(p, ctx=decord.cpu(0), width=224, height=224)
                sampled_indices = np.linspace(
                    0, len(video) - 1, frames_per_video, dtype=int
                )
                frames = video.get_batch(sampled_indices).asnumpy()
                video_files.extend([f for f in frames])

        return video_files

    def run_video_check_input(self, vids_comb_id, questions):
        q0 = next(iter(questions.values()))
        _, video_paths, temporal_divisor, _ = self.load_videos(q0)
        narration = self.format_narrations(q0)

        video_files = self.retrieve_frames(video_paths)

        if len(narration) > self.config["min_cache_chars"]:
            stride = math.ceil(len(narration) / self.config["min_cache_chars"])
            narration = " ".join(narration.split(" ")[::stride])

        for q_id, question in questions.items():
            q, correct_idx = self.formulate_question(
                question=question, temporal_divisor=temporal_divisor
            )
            input_prompt = self.format_prompt(narration, q, video_files)
            inputs = self.processor(
                images=video_files if self.config["mode"] == "visual_text" else None,
                text=input_prompt,
                add_special_tokens=False,
                return_tensors="pt",
            )
            print(inputs)

    def run_video(self, vids_comb_id, questions):
        q0 = next(iter(questions.values()))
        _, video_paths, temporal_divisor, loaded_video_seconds, total_video_seconds = (
            self.load_videos(q0)
        )
        narration = self.format_narrations(q0)

        video_files = self.retrieve_frames(video_paths)

        if len(narration) > self.config["min_cache_chars"]:
            stride = math.ceil(len(narration) / self.config["min_cache_chars"])
            narration = " ".join(narration.split(" ")[::stride])

        results = {}
        for q_id, question in questions.items():
            q, correct_idx = self.formulate_question(
                question=question, temporal_divisor=temporal_divisor
            )
            start = time.time()
            n_attempts = 0
            success = False
            while n_attempts < self.config["max_attempts"] and not success:
                try:
                    input_prompt = self.format_prompt(narration, q, video_files)
                    inputs = self.processor(
                        images=video_files
                        if self.config["mode"] == "visual_text"
                        else None,
                        text=input_prompt,
                        add_special_tokens=False,
                        return_tensors="pt",
                    ).to(self.model.device)

                    response = self.model.generate(
                        **inputs, max_new_tokens=1, do_sample=False
                    )

                    answer = self.parse_response(
                        self.processor.decode(response[0]), len(question["choices"])
                    )
                    success = True
                except Exception as e:
                    answer = -1
                    if "blocked prompt" in str(e):
                        print("BLOCKED PROMPT")
                        success = True
                    else:
                        print(e)
                    n_attempts += 1

            stop = time.time()
            total_time = stop - start
            entry = {
                "id": q_id,
                "total_time": total_time,
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

        self.remove_tmp_videos(video_paths)

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
