import re
from typing import List


SENTENCE_SPLIT_REGEX = re.compile(r"(\W+)")


def tokenize(sentence: str):
    tokens = SENTENCE_SPLIT_REGEX.split(sentence.lower())
    tokens = [t.strip() for t in tokens if len(t.strip()) > 0]
    return tokens


def load_str_list(fname: str):
    with open(fname) as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines]
    return lines


class VocabDict:
    """ """

    def __init__(self, vocab_file: str):
        self.word_list = load_str_list(vocab_file)
        self.word2idx_dict = {w: n_w for n_w, w in enumerate(self.word_list)}
        self.vocab_size = len(self.word_list)
        self.unk2idx = (
            self.word2idx_dict["<unk>"] if "<unk>" in self.word2idx_dict else None
        )

    def idx2word(self, index: int) -> str:
        return self.word_list[index]

    def word2idx(self, word: str) -> int:
        if word in self.word2idx_dict:
            return self.word2idx_dict[word]
        elif self.unk2idx is not None:
            return self.unk2idx
        else:
            raise ValueError(
                f"{word} %s not in dictionary (while dictionary does not contain <unk>)"
            )

    def tokenize_and_index(self, sentence: str) -> List[int]:
        inds = [self.word2idx(w) for w in tokenize(sentence)]

        return inds
