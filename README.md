This repo contains the code used to run the benchmark in https://hd-epic.github.io/ (CVPR 2025). 

## Quickstart

Download the HD-epic videos, and encode them at your desired resolution and framerate using `scripts/convert_mp4s.py`

Download the VQA from `https://github.com/hd-epic/hd-epic-annotations/tree/main/vqa-benchmark`

Decide which model you'd like to use, and check the config is correct. In particular, check that all the question types are included.

```bash
python run_benchmark.py --mp4_dir ENCODED_MP4_PATH --questions_dir QUESTIONS_PATH --config CONFIG_PATH
```

By default, results are cached in case your run dies. If you want to wipe the cache and start over, use ```--cached 0```. Outputs will appear in ```output/CONFIG_NAME```.

Once a run has finished, you can get the results by doing:

```bash
python scripts/analyse_output.py --input output/CONFIG_NAME_all_runs.json
```

## Installation

Rather than providing separate installation instructions for every model here, just follow the installation instructions for each model in the repos listed below. Or just keep running that command and using conda/pip to install the necessary packages as you encounter them.

## A note on model versions

We have forked the code from the publicly available models used at the time we ran the benchmarks for repeatability. Thanks to the respective authors:

LongVA: `https://github.com/EvolvingLMMs-Lab/LongVA`

Llava-Video: `https://llava-vl.github.io/blog/2024-09-30-llava-video/`

Llama: `https://www.llama.com/`

VideoLlama2: `https://github.com/DAMO-NLP-SG/VideoLLaMA2`

We cannot guarantee that models behind APIs (or the APIs) won't have changed. This is:

Gemini: `https://ai.google.dev/gemini-api/docs`

## Adding a new model

You can use our dataloader (decord based) by inheriting from `models/base_model`. As an example, see how it is done in `llavavideo.py`.
