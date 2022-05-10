import re
from typing import List

SENTENCE_SPLIT_REGEX = re.compile(r"(\W+)")


def tokenize(sentence: str) -> List[str]:
    tokens = SENTENCE_SPLIT_REGEX.split(sentence.lower())
    tokens = [t.strip() for t in tokens if len(t.strip()) > 0]
    return tokens


def load_str_list(fname: str):
    with open(fname) as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines]
    return lines
