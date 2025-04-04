from random import randint
from data.vqa_dataset import VQADataset, DictCollate
import os.path as osp
import json

from torch.utils.data import DataLoader


class Model:
    """
    Dummy model that returns a random number between 0 and 10 for each question.
    """

    def __init__(self, args, config):
        self.config = config
        self.loader = DataLoader(
            VQADataset(args, config), batch_size=1, collate_fn=DictCollate
        )

    def run_eval(self, run_output_dir=None):
        # Run model
        results = {}
        for i, batch in enumerate(self.loader):
            question_ids = [batch[i]["question_id"] for i in range(len(batch))]
            batch_results = {q: randint(0, 10) for q in question_ids}
            for q, v in batch_results.items():
                results[q] = v

                if run_output_dir is not None:
                    # Save every result as it's produced in case of crash
                    output_json = osp.join(run_output_dir, f"{q}.json")
                    with open(output_json, "w") as f:
                        json.dump(v, f)
