import os
import os.path as osp
import time
import datetime
import json
import numpy as np

from models.base_model import BaseVLM

import google.generativeai as genai
from google.generativeai import caching
from google.generativeai import GenerationConfig

import concurrent.futures
import multiprocessing


class Model(BaseVLM):
    def __init__(self, args, config, run_output_dir):
        super().__init__(args, config, run_output_dir)

        if not self.args.check_input_only:
            genai.configure(api_key=os.environ["API_KEY"])

        self.gen_config = GenerationConfig(temperature=0.0, max_output_tokens=1)

        self.vid_file_to_upload_name = {}

        if self.args.check_input_only:
            return

        if self.config["clear_local_tmp"]:
            self.remove_tmp_videos()

        if self.config["clear_uploaded_storage"]:
            self.clean_up_uploaded_files()
        else:
            self.populate_vid_file_to_upload_name()

    def clean_up_uploaded_files(self):
        existing_files = [f for f in genai.list_files()]
        print(f"Cleaning up {len(existing_files)} existing files.")
        for f in existing_files:
            print(f.name, f.display_name)
            f.delete()

    def populate_vid_file_to_upload_name(self):
        for f in genai.list_files():
            self.vid_file_to_upload_name[f.display_name] = f.name
        print(self.vid_file_to_upload_name)

    def remove_uploaded_videos(self, video_files):
        for f in video_files:
            display_name = f.display_name
            # delete from self.vid_file_to_upload_name if value == f.display_name
            keys = [
                k for k, v in self.vid_file_to_upload_name.items() if v == display_name
            ]
            for k in keys:
                del self.vid_file_to_upload_name[k]
            f.delete()

    def run_video_check_input(self, vids_comb_id, questions):
        q0 = next(iter(questions.values()))
        (
            input_keys,
            video_paths,
            temporal_divisor,
            loaded_video_seconds,
            total_video_seconds,
        ) = self.load_videos(q0)

        narration = self.format_narrations(q0)
        contents = []
        for i in range(len(input_keys)):
            contents.append(f"{input_keys[i]}: ")
            contents.append(video_paths[i])
        if narration:
            contents.append(narration)

        for q_id, question in questions.items():
            q, correct_idx = self.formulate_question(
                question=question, temporal_divisor=temporal_divisor
            )
            input = [*contents, q]
            print(input)
            print(video_paths)
        # self.remove_tmp_videos(video_paths)

    def run_video(self, vids_comb_id, questions, load_lock, upload_lock):
        q0 = next(iter(questions.values()))
        narration = self.format_narrations(q0)

        model_display_name = vids_comb_id
        if len(model_display_name) > 128:
            model_display_name = model_display_name[:128]

        with load_lock:
            (
                input_keys,
                video_paths,
                temporal_divisor,
                loaded_video_seconds,
                total_video_seconds,
            ) = self.load_videos(q0)

        with upload_lock:
            video_files = []
            for p in video_paths:
                try:
                    upload_name = self.vid_file_to_upload_name[p]
                    video_file = genai.get_file(upload_name)
                    while video_file.state.name == "PROCESSING":
                        # print('Waiting for video to be processed.')
                        time.sleep(2)
                        video_file = genai.get_file(video_file.name)
                    print(
                        f"Using cached video {upload_name}: {video_file.display_name}"
                    )
                except Exception:
                    video_file = genai.upload_file(path=p, display_name=p)
                    while video_file.state.name == "PROCESSING":
                        # print('Waiting for video to be processed.')
                        time.sleep(2)
                        video_file = genai.get_file(video_file.name)
                    self.vid_file_to_upload_name[p] = video_file.name

                video_files.append(video_file)

        cached_context = False
        if len(questions) > 1 and self.config["enable_cache"]:
            if loaded_video_seconds > self.config["min_cache_video_length_seconds"]:
                cached_context = True
            if len(narration) > self.config["min_cache_chars"]:
                cached_context = True
        else:
            cached_context = False

        contents = []
        for i in range(len(video_files)):
            contents.append(f"{input_keys[i]}: ")
            contents.append(video_files[i])

        if narration:
            contents.append(narration)

        if cached_context:
            print("Using cached context")
            cache = caching.CachedContent.create(
                model=self.config["variant"],
                display_name=model_display_name,
                system_instruction=(self.config["prompts"]["sys"]),
                contents=contents,
                ttl=datetime.timedelta(minutes=self.config["cache_time_minutes"]),
            )
            model = genai.GenerativeModel.from_cached_content(cached_content=cache)
        else:
            model = genai.GenerativeModel(
                model_name=self.config["variant"],
                system_instruction=self.config["prompts"]["sys"],
            )

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
                    if cached_context:
                        cache.update(
                            ttl=datetime.timedelta(
                                minutes=self.config["cache_time_minutes"]
                            )
                        )
                        input = [(q)]
                    else:
                        input = [*contents, q]
                    response = model.generate_content(
                        input, generation_config=self.gen_config
                    )
                    answer = self.parse_response(
                        response.text, len(question["choices"])
                    )
                    success = True
                    failure_reason = None

                except Exception as e:
                    failure_reason = str(e)
                    answer = -1
                    if "blocked prompt" in str(e):
                        print("BLOCKED PROMPT")
                        success = True
                    elif "HARM" in str(e):
                        print("HARM")
                        success = True
                    else:
                        sleep_time = np.random.randint(
                            2 ** (n_attempts + 3), 2 ** (n_attempts + 4)
                        )
                        time.sleep(sleep_time)
                        print(e)
                        answer = -2
                    n_attempts += 1

            stop = time.time()
            total_time = stop - start
            entry = {
                "id": q_id,
                "time": total_time,
                "answer": answer,
                "correct": 1.0 if answer == correct_idx else 0.0,
                "input_total_seconds": total_video_seconds,
                "input_subsampled_seconds": loaded_video_seconds,
                "failure_reason": failure_reason,
            }
            print(q)
            print(entry)

            with open(osp.join(self.run_output_dir, f"{q_id}.json"), "w") as f:
                json.dump(entry, f)
            results[q_id] = entry

            time.sleep(self.config["wait_time_seconds"])

        self.remove_tmp_videos(video_paths)
        # self.remove_uploaded_videos(video_files)
        self.remove_uploaded_videos(
            [
                v
                for v in video_files
                if "_" in osp.splitext(osp.basename(v.display_name))[0]
            ]
        )
        if cached_context:
            cache.delete()

        return results

    def run_eval(self):
        results = {}

        if self.args.check_input_only:
            for vids_comb_id, questions in self.questions_by_vid.items():
                self.run_video_check_input(vids_comb_id, questions)
            return

        load_lock = multiprocessing.Lock()
        upload_lock = multiprocessing.Lock()

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.config["concurrent_pool_size"]
        ) as executor:
            future_to_vids_comb_id = {
                executor.submit(
                    self.run_video, vids_comb_id, questions, load_lock, upload_lock
                ): vids_comb_id
                for vids_comb_id, questions in self.questions_by_vid.items()
            }
            for future in concurrent.futures.as_completed(future_to_vids_comb_id):
                vids_comb_id = future_to_vids_comb_id[future]
                try:
                    vid_results = future.result()
                    for k, v in vid_results.items():
                        results[k] = v
                except Exception as exc:
                    print(f"Combination {vids_comb_id} generated an exception: {exc}")

        return results
