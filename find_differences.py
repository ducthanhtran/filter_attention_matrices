#!/usr/bin/env python3
import argparse
from functools import cmp_to_key
from sys import stdin, stdout
from typing import Generator, List, NamedTuple

import numpy as np


BREAK_TOKEN = '<BREAK>'


Comparison = NamedTuple('Comparison', [('sentence_id', int),
                                       ('bleu_difference', float),
                                       ('bleu_left', float),  # TODO: needed?
                                       ('bleu_right', float)  # TODO: needed?
                                       ])

TranslationAttention = NamedTuple('TranslationAttention', [('sentence_id', int),
                                                           ('source', str),
                                                           ('target', str),  # TODO: needed?
                                                           ('attention_matrix', np.ndarray)
                                                           ])

Value = NamedTuple('Value', [('sentence_id', int),
                             ('bleu_difference', float),
                             ('attention_max', np.ndarray),
                             ('attention_sum', np.ndarray),
                             ('ordered_source_tokens', str)
                             ])


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-translation', type=argparse.FileType('r'), default=stdin)
    parser.add_argument('--input-comparison', type=argparse.FileType('r'), required=True,
                        help="From Miguel's hyp_compare script.")
    parser.add_argument('--k', type=float, default=0.10)
    parser.add_argument('--output', type=argparse.FileType('w'), default=stdout)
    return parser


def comparisons(comparison_text: List[str]) -> Generator[Comparison, None, None]:
    for comp in comparison_text:
        entries = comp.split('#')
        yield Comparison(sentence_id=int(entries[1]),
                         bleu_difference=float(entries[0]),
                         bleu_left=float(entries[2]),
                         bleu_right=float(entries[3]))


def translation_attentions(translation_with_attention_text: List[str]) -> Generator[TranslationAttention, None, None]:
    is_header_line = True
    for line in translation_with_attention_text:
        if is_header_line:
            header = line.split('|||')

            sentence_id = int(header[0])
            source = header[3]
            target = header[1]
            source_length = int(header[4])
            target_length = int(header[5])

            rows = ''
            is_header_line = False
        elif line != '':
            rows = ' '.join([rows, line])
        else:
            # encountered blank line - end of attention matrix
            is_header_line = True  # reset header flag
            attention_matrix = np.fromstring(rows, dtype=float, sep=' ').reshape(source_length, target_length)
            yield TranslationAttention(sentence_id=sentence_id,
                                       source=source,
                                       target=target,
                                       attention_matrix=attention_matrix)


def compute_values(translation_with_attention_text: List[str], comps: List[Comparison]) -> List[Value]:
    values = []

    for trans_att_matrix in translation_attentions(translation_with_attention_text):
        # skip translation hypothesis that are the same for both models
        sentence_id = trans_att_matrix.sentence_id
        if comps[sentence_id-1].bleu_difference == 0:
            continue

        sentences = trans_att_matrix.source.split(BREAK_TOKEN)
        break_indices = [i for i, token in enumerate(trans_att_matrix.source.split()) if token == BREAK_TOKEN]
        if len(sentences) == 3:
            context_words = sentences[0].split() + sentences[1].split()
            context_length = len(context_words) + 2  # two missing <BREAK> tokens
        elif len(sentences) == 2:
            context_words = sentences[0].split()
            context_length = len(context_words) + 1  # one missing <BREAK> token
        else:
            continue  # no context sentence (only occurs for first translation sentence)

        # remove <BREAK>-row(s)
        sub_matrix = trans_att_matrix.attention_matrix[:context_length, :]
        sub_matrix = np.delete(sub_matrix, break_indices, axis=0)

        # compute values
        scores_max = np.max(sub_matrix, axis=1)
        sorted_indices = (-np.array(scores_max)).argsort()

        values.append(Value(sentence_id=trans_att_matrix.sentence_id,
                            bleu_difference=comps[sentence_id-1].bleu_difference,
                            attention_max=scores_max,
                            attention_sum=np.sum(sub_matrix, axis=0),
                            ordered_source_tokens=' '.join([context_words[i] for i in sorted_indices])))
    return values


def value_compare(value1: Value, value2: Value) -> int:
    bleu_diff = value1.bleu_difference - value2.bleu_difference
    attention_max = np.max(value1.attention_max) - np.max(value2.attention_max)

    if bleu_diff != 0:
        return np.sign(bleu_diff)
    else:
        return np.sign(attention_max)


def array_to_string(array: np.ndarray) -> str:
    sorted_array = np.sort(array)
    sorted_array = sorted_array[::-1]
    strings = ['{0:.2f}'.format(a) for a in sorted_array]
    return ' '.join(strings)


if __name__ == '__main__':
    args = create_parser().parse_args()

    # Read in data
    with args.input_translation as trans_att_matrix, args.input_comparison as comp:
        comparisons_text = comp.read().splitlines()
        translation_with_attention_text = trans_att_matrix.read().splitlines()

    comps = list(comparisons(comparisons_text))
    comps.sort(key=lambda x: x.sentence_id)

    values = compute_values(translation_with_attention_text, comps)
    values.sort(key=cmp_to_key(value_compare))

    with args.output as out:
        for v in values:
            output = ' # '.join([str(v.sentence_id),
                                 '{0:.2f}'.format(v.bleu_difference),
                                 array_to_string(v.attention_max),
                                 v.ordered_source_tokens,
                                 array_to_string(v.attention_sum)])
            out.write(output + '\n')
