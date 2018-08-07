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

    max_attention_scores = []
    headers = []
    for header, matrix in blocks(lines):
        splitted = header.split('|||')
        source = splitted[3].split('<BREAK>')
        if len(source) == 3:
            prev2 = source[0].split()
            prev1 = source[1].split()
            context_length = len(prev2) + len(prev1)
        elif len(source) == 2:
            prev1 = source[0].split()
            context_length = len(prev1)
        else:
            continue

        attention_score = np.max(matrix[0:context_length, 0:context_length])
        max_attention_scores.append(attention_score)
        headers.append(header)

    # print("Maximum attention score in context sentences: {}".format(np.max(max_attention_scores)))
    print("Mean attention score in context sentences: {}\n".format(np.mean(max_attention_scores)))

    sorted_indices = (-np.array(max_attention_scores)).argsort()
    for i in range(10):
        index = sorted_indices[i]
        print('attenion score: {}'.format(max_attention_scores[index]))
        print("Header: {}\n".format(headers[index]))



