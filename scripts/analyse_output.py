import json
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    return parser.parse_args()


def read_json(input):
    with open(input, "r") as f:
        data = json.load(f)
    return data


def acc_by_key(data):
    # only split by last occurence of '_' in key_types
    key_types = set([k.rsplit("_", 1)[0] for k in data.keys()])
    key_types = list(key_types)
    key_types.sort()
    print(key_types)

    count_dict = {}
    correct_dict = {}

    # get accuracy by key type
    for key in key_types:
        count = 0
        correct = 0
        for k, v in data.items():
            if k.startswith(key):
                count += 1
                correct += v["correct"]
        count_dict[key] = count
        correct_dict[key] = correct

    # print(count_dict)
    # print(correct_dict)
    # print(key_types)

    for key in key_types:
        print(f"{key}: {correct_dict[key] / count_dict[key]}")


def model_choice(data):
    response_counts = {}
    for k, v in data.items():
        response = v["answer"]
        response_counts[response] = response_counts.get(response, 0) + 1

    # sort by key
    response_counts = dict(sorted(response_counts.items(), key=lambda x: x[0]))
    print(response_counts)

    print(f"Total responses: {sum(response_counts.values())}")


def find_invalid_answers(data):
    invalid_answers = []
    failures = []
    for k, v in data.items():
        response = v["answer"]
        if response == -1:
            invalid_answers.append(k)
        elif response == -2:
            failures.append(k)

    invalid_answers.sort()

    print()
    print(f"Invalid answers (safety refusals, blocked etc.): {len(invalid_answers)}:")
    for ans in invalid_answers:
        print(ans)

    print()
    print(f"Failures: {len(failures)}:")
    for ans in failures:
        print(ans)


def main(args):
    data = read_json(args.input)
    acc_by_key(data)
    model_choice(data)
    find_invalid_answers(data)


if __name__ == "__main__":
    args = parse_args()
    main(args)
