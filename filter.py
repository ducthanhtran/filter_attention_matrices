import argparse
from sys import stdin
from typing import List

import numpy as np


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=argparse.FileType('r'), default=stdin)
    parser.add_argument('--k', type=int, default=3)
    return parser


def blocks(lines: List[str]):
    while lines:
        header = lines.pop(0)

        att = []
        for line in lines:
            att.append(line)
            if line == '':
                break
        lines = lines[len(att):]  # remove in lines; hacky solution

        dimensions = header.split('|||')[-2:]
        dimensions = [int(d) for d in dimensions]
        matrix = np.fromstring(' '.join(att), dtype=float, sep=' ').reshape(dimensions)
        yield header, matrix


if __name__ == '__main__':
    args = create_parser().parse_args()

    with args.input as f:
        lines = f.read().splitlines()

    headers = []  # type: List[str]
    highest_attention_scores = []  # type: List[np.ndarray]
    # highest_attention_indices = []  # type: List[np.ndarray]
    # source_sentences = []  # type: List[str]

    for header, matrix in blocks(lines):
        split_header = header.split('|||')
        source = split_header[3].split('<BREAK>')

        if len(source) == 3:
            # source_sentences.append(' <BREAK> '.join(source[:2]) + ' <BREAK> ')
            prev2 = source[0].split()
            prev1 = source[1].split()
            context_length = len(prev2) + len(prev1) + 2  # two BREAK tokens
        elif len(source) == 2:
            # source_sentences.append(source[0] + ' <BREAK> ')
            prev1 = source[0].split()
            context_length = len(prev1) + 1  # one BREAK token
        else:
            continue

        headers.append(header)

        matrix_subset = matrix[0:context_length, :]
        # first 3 maximum attention scores
        sorted_indices = (-matrix_subset.ravel()).argsort()
        highest_attention_scores.append(np.array([matrix_subset.ravel()[i] for i in sorted_indices[:args.k]]))
        # highest_attention_indices.append(sorted_indices[:args.k])

    # sort by 1st, 2nd and 3rd attention score
    highest_att_scores = np.array(highest_attention_scores)
    lex_sort = np.lexsort((highest_att_scores[:, 2], highest_att_scores[:, 1], highest_att_scores[:, 0]))

    sorted_scores = highest_att_scores[lex_sort][::-1]
    # sorted_indices = np.array(highest_attention_indices)[lex_sort][::-1]

    for i in range(10):
        index = lex_sort[i]
        print("\nHeader: {}".format(headers[index]))
        print('attention scores.: {}'.format(sorted_scores[i, :]))
        # print('attention indices: {}'.format(sorted_indices[i, :]))

        # words = source_sentences[index].split()
        # print(len(words))
        # print('source words: {}'.format([words[j] for j in sorted_indices[i, :]]))




