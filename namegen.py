#!/usr/bin/python3

import argparse
import pickle
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field

BEGIN = ""
END = None


def extended_window(sequence, n):
    """A sliding window method that starts out of edges.

    For example, the input "abcd" and size 3 will yield "a", "ab", "abc" and "bcd".
    """
    for i in range(len(sequence)):
        start = max(0, i - n + 1)
        yield sequence[start : i + 1]


def state_transition_counts(data, n):
    """Creates a dictionary counting state transitions to estimate P(state|previous_states)."""
    counts = defaultdict(Counter)
    for row in data:
        if len(row) < n + 1:
            continue
        states = BEGIN
        for chunk in extended_window(row, n + 1):
            next_state = chunk[-1]
            counts[states][next_state] += 1
            states = chunk[-n:]

        counts[states][END] += 1

    return counts


def counts_to_probs(counts):
    """Convert counts to probabilities"""
    probs = defaultdict(dict)
    for k in counts:
        total = sum(counts[k].values())
        for v in counts[k]:
            probs[k][v] = counts[k][v] / total
    return probs


def temperature(probs, temp):
    """Adjust the uniformity of the distribution.

    1.0 results in the original distribution. Increasing the number will create a more uniform result.
    """
    probs_new = {}
    for k1, inner_probs in probs.items():
        invtemp = 1.0 / temp
        total_p = sum(p**invtemp for p in inner_probs.values())
        probs_new[k1] = {k: (inner_probs[k] ** invtemp) / total_p for k in inner_probs}
    return probs_new


def clean(data, min_size: int):
    for word in data:
        if len(word) < min_size:
            continue
        yield word.strip()


@dataclass
class MarkovModel:
    order: int = 2
    probs: dict = field(default_factory=dict, init=False)

    def train(self, data):
        window = self.order
        counts = state_transition_counts(data, window)
        probs = counts_to_probs(counts)
        self.probs = probs
        return self

    def sample(self, temp=1):
        n = self.order
        word = ""

        states = BEGIN
        temp_probs = temperature(self.probs, temp)
        while True:
            inner_probs = temp_probs[states]
            next_state = random.choices(
                list(inner_probs.keys()), list(inner_probs.values())
            )[0]
            if next_state == END:
                break
            word += next_state
            states = word[-n:]

        return word

    def save(self, path):
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as fh:
            return pickle.load(fh)


def cmd_train(input_file, output_file, order):
    model = MarkovModel(order=order)
    with open(input_file) as fh:
        model.train(clean(fh, order + 1))
    model.save(output_file)


def cmd_sample(input_file, num_samples, temp):
    model = MarkovModel.load(input_file)
    for i in range(num_samples):
        print(model.sample(temp))


def cmd():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help="Command to run.")
    parser_train = subparsers.add_parser(
        "train",
        help="Train a model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_sample = subparsers.add_parser(
        "sample",
        help="Sample from a model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser_train.add_argument(
        "input_file",
        type=str,
        help="Input text file name. Expects a list of names separated by newlines.",
    )
    parser_train.add_argument("output_file", type=str, help="Output model file name.")
    parser_train.add_argument(
        "--order", type=int, help="Markov model order.", default=1
    )
    parser_train.set_defaults(command="train")

    parser_sample.add_argument("input_file", type=str, help="Input model file name.")
    parser_sample.add_argument(
        "--num-samples", type=int, help="Number of samples to produce.", default=1
    )
    parser_sample.add_argument(
        "--temp",
        type=float,
        default=1.0,
        help="Temperature for the sampling. 1.0 results in the original distribution. "
        "A higher value leads to a more uniform distribution.",
    )

    parser_sample.set_defaults(command="sample")

    args = parser.parse_args()

    if hasattr(args, "command"):
        if args.command == "sample":
            return cmd_sample(args.input_file, args.num_samples, args.temp)
        elif args.command == "train":
            return cmd_train(args.input_file, args.output_file, args.order)

    parser.print_help()


if __name__ == "__main__":
    cmd()
