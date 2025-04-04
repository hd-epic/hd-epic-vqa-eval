import argparse
import json
import os
import os.path as osp
import importlib

from utils.config import load_check_config

from collections import OrderedDict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mp4_dir",
        type=str,
        default="storage/rgb_768_1_vig",
        help="Path to directory containing mp4 files - should set to BC4 default.",
    )
    parser.add_argument(
        "--input_res",
        type=int,
        default=768,
        help="Resolution of dataset passed to this program as --mp4_dir.",
    )
    parser.add_argument(
        "--input_fps",
        type=int,
        default=1,
        help="fps of the dataset passed to this program as --mp4_dir.",
    )
    parser.add_argument(
        "--dataset_orig_res",
        type=int,
        default=1408,
        help="Original resolution of the dataset, not the version passed to this program.",
    )
    parser.add_argument(
        "--dataset_orig_fps",
        type=int,
        default=30,
        help="Original fps of the dataset, not the version passed to this program.",
    )
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--config_dir", type=str, default="configs")
    parser.add_argument(
        "--questions_dir", type=str, default="/Users/toby/Code/d-epic-benchmark-team1"
    )
    parser.add_argument("--models_dir", type=str, default="models")
    parser.add_argument("--narrations_dir", type=str, default="narrations")
    parser.add_argument(
        "--config",
        type=str,
        default="debug_gemini_flash",
        help="Config file to use, without json extension",
    )
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--cached",
        type=int,
        default=1,
        help="If 0 empty the output directory and answer all questions. If 1, do not answer questions which have already been answered and saved.",
    )
    parser.add_argument(
        "--check_input_only",
        action="store_true",
        help="If true, only print parsed questions, and do not delete any cropped videos.",
    )
    parser.add_argument(
        "--question_file_idx",
        nargs="+",
        help="If set, only run this question file from the config. If -1, run all question files.",
    )
    return parser.parse_args()


def main(args):
    # Load config file
    config = load_check_config(osp.join(args.config_dir, f"{args.config}.json"))

    # Create output directory
    run_output_dir = osp.join(args.output_dir, f"{args.config.split('/')[-1]}")
    os.makedirs(run_output_dir, exist_ok=True)

    # Delete files separately - we don't want to accidentally rm -rf stuff.
    if not args.cached:
        for f in os.listdir(run_output_dir):
            try:
                os.remove(osp.join(run_output_dir, f))
            except:
                pass

    # Setup VQA model
    model_name = config["model"]
    module_path = f"{args.models_dir}.{model_name}"
    module = importlib.import_module(module_path)
    model = module.Model(args, config, run_output_dir=run_output_dir)

    # Run evaluation
    results = model.run_eval()
    if args.check_input_only:
        return

    # Combine all results from last run in output directory
    with open(
        osp.join(args.output_dir, f"{args.config.split('/')[-1]}_last_run.json"), "w"
    ) as f:
        json.dump(results, f)

    # Combine all results from all runs in output directory
    all_results = {}
    for f in os.listdir(run_output_dir):
        if not f.endswith(".json"):
            continue
        idx = osp.splitext(f)[0]
        with open(osp.join(run_output_dir, f), "r") as f:
            single_result = json.load(f)
            if (
                "failure_reason" in single_result
                and single_result["failure_reason"] is None
            ):
                del single_result["failure_reason"]
            all_results[idx] = single_result
            # all_results[idx] = json.load(f)

    # sort all_results by key
    all_results = OrderedDict(sorted(all_results.items(), key=lambda x: x[0]))

    with open(
        osp.join(args.output_dir, f"{args.config.split('/')[-1]}_all_runs.json"), "w"
    ) as f:
        json.dump(all_results, f, indent=4)

    # Print overall accuracy
    overall_accuracy = sum([int(v["correct"]) for k, v in all_results.items()]) / float(
        len(all_results)
    )
    print(f"Overall accuracy: {overall_accuracy}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
