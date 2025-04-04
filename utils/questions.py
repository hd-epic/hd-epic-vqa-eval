import json
import os
import os.path as osp
import datetime


def load_questions(args, config, run_output_dir=None, cached=False):
    questions = {}
    if args.question_file_idx is not None:
        question_files = [config["questions"][int(i)] for i in args.question_file_idx]
    else:
        question_files = config["questions"]

    for q_file in question_files:
        question_fn = osp.join(args.questions_dir, f"{q_file}.json")
        with open(question_fn, "r") as f:
            tmp_questions = json.load(f)
        questions.update(tmp_questions)

    print(f"Loaded {len(questions)} questions")

    if cached:
        # get question_ids already in run_output_dir
        if run_output_dir is not None:
            existing_qs = [
                osp.basename(osp.splitext(f)[0]) for f in os.listdir(run_output_dir)
            ]
            for q in existing_qs:
                questions.pop(q, None)

    print(f"Running {len(questions)} questions which are not cached")

    return questions


def group_questions_by_identical_input(questions):
    questions_by_vid = {}
    for k, v in questions.items():
        q_vids = v["inputs"]

        all_strs = []
        for input_k, input_v in q_vids.items():
            vid = input_v["id"]
            if "image" in input_k:
                time_str = input_v["time"]
            else:
                start = input_v["start_time"] if "start_time" in input_v else -1
                end = input_v["end_time"] if "end_time" in input_v else -1
                time_str = f"{start}_{end}"
            vid_start_end = f"{vid}_{time_str}"
            all_strs.append(vid_start_end)

        vid_start_end = "_".join(all_strs)
        if vid_start_end not in questions_by_vid:
            questions_by_vid[vid_start_end] = {}
        questions_by_vid[vid_start_end][k] = v

    # get average lengths of questions_by_vid
    lengths = [len(v) for k, v in questions_by_vid.items()]
    print(
        f"Average number of questions per video/group of videos: {sum(lengths) / (len(lengths) + 1e-6)}"
    )

    return questions_by_vid


def secs_from_time_str(s):
    t = datetime.datetime.strptime(s, "%H:%M:%S.%f").time()
    total_seconds = t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6
    return int(total_seconds)
