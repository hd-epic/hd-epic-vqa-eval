import os.path as osp
import pickle


class NarrationReader:
    def __init__(self, args, config):
        self.args = args
        self.config = config
        fn = osp.join(args.narrations_dir, f"{config['narrations']}.pkl")
        with open(fn, "rb") as f:
            self.narrations = pickle.load(f)

    def get_narrations(self, vid_id, start_s, end_s):
        if start_s == -1 and end_s == -1:
            narrations = self.narrations[(self.narrations["Video ID"] == vid_id)]
        else:
            narrations = self.narrations[
                (self.narrations["Timestamp"] >= start_s)
                & (self.narrations["Timestamp"] <= end_s)
                & (self.narrations["Video ID"] == vid_id)
            ]
        if not narrations.empty:
            if "verb_noun" in self.config["mode"]:
                narrations = (
                    narrations[["Main Actions", "Timestamp"]]
                    .sort_values("Timestamp")
                    .values.tolist()
                )
                new_narrations = []
                for a, t in narrations:
                    if len(a) > 0:
                        new_narrations.append([f"{' '.join(a[0])}", t])
                narrations = new_narrations
            else:
                narrations = (
                    narrations[["Narration", "Timestamp"]]
                    .sort_values("Timestamp")
                    .values.tolist()
                )
        else:
            narrations = []

        return narrations


if __name__ == "__main__":
    # create dummy args class with narrations dir
    class Args:
        narrations_dir = "narrations"

    args = Args()
    config = {"narrations": "eka_all_df_v0.5"}

    nr = NarrationReader(args, config)

    n = nr.get_narrations("P05-20240427-151808", 10, 20)
    print(n)
    print(type(n[0][1]))
