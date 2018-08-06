import argparse
from sys import stdin
from typing import List, NamedTuple

import numpy as np


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=argparse.FileType('r'), default=stdin)
    return parser


def blocks(lines: List[str]):
    while lines:
        header = lines.pop(0)

        att = []
        for line in lines:
            att.append(line)
            if line == '':
                break
        lines = lines[len(att):] #  remove in lines; hacky

        dimensions = header.split('|||')[-2:]
        dimensions = [int(d) for d in dimensions]
        matrix = np.fromstring(' '.join(att), dtype=float, sep=' ').reshape(dimensions)

        yield header, matrix


if __name__ == '__main__':
    args = create_parser().parse_args()

    with args.input as f:
        lines = f.read().splitlines()

    for header, matrix in blocks(lines):
        splitted = header.split('|||')
        source = splitted[3].split('<BREAK>')

        pass




