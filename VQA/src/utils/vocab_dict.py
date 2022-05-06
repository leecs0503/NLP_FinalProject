from .text_helper import tokenize, load_str_list
from typing import List


class VocabDict:
    """ """

    def __init__(self, vocab_file: str):
        """
        Args:
                vocab_file: vocab file이 저장되어 있는 경로
        """
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
