import os.path as osp
import time
import json

from models.base_model import BaseVLM
from longva.model.builder import load_pretrained_model
from longva.mm_utils import tokenizer_image_token
from longva.constants import IMAGE_TOKEN_INDEX

import torch


class Model(BaseVLM):
    def __init__(self, args, config, run_output_dir):
        super().__init__(args, config, run_output_dir, backend="decord")
        self.gen_kwargs = {
            "do_sample": False,
            "top_p": None,
            "num_beams": 1,
            "use_cache": True,
            "max_new_tokens": 2,
            "top_k": None,
        }
        # # you can also set the device map to auto to accomodate more frames
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            self.config["model_cache"],
            None,
            "llava_qwen",
            device_map="cuda:0",
            attn_implementation=None,
        )

    def run_video(self, vids_comb_id, questions):
        q0 = next(iter(questions.values()))
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

        video_tensors = [
            self.image_processor.preprocess(v, return_tensors="pt")["pixel_values"].to(
                self.model.device, dtype=torch.float16
            )
            for v in video_nps
        ]

        contents = []
        for i in range(len(input_keys)):
            contents.append(f"{input_keys[i]}: <image>\n")

        if narration:
            contents.append(narration)

        results = {}
        for q_id, question in questions.items():
            q, correct_idx = self.formulate_question(
                question=question, temporal_divisor=temporal_divisor
            )
            start = time.time()

            prompt = f"<|im_start|>system\n{self.config['prompts']['sys']}.<|im_end|>\n<|im_start|>user\n{contents}{q}<|im_end|>\n<|im_start|>assistant\n"
            input_ids = (
                tokenizer_image_token(
                    prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                )
                .unsqueeze(0)
                .to(self.model.device)
            )

            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=video_tensors,
                    modalities=["video"] * len(video_tensors),
                    **self.gen_kwargs,
                )
                response = self.tokenizer.batch_decode(
                    output_ids, skip_special_tokens=True
                )[0].strip()
            print(response)

            answer = self.parse_response(response, len(question["choices"]))

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
