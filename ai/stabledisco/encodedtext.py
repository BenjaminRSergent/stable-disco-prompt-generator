import typing

import clip
import torch


class EncodedText:
    def __init__(self, text: str, tokens: torch.Tensor) -> None:
        self._text = text
        self._tokens = tokens

    def get_text(self) -> str:
        return self._text

    def get_tokens(self) -> str:
        return self._tokens

    def get_nonzero_tokens(self) -> str:
        return self._tokens[: self.get_tokens_len() - 1]

    def get_tokens_len(self) -> int:
        return len((self._tokens != 0).nonzero(as_tuple=True)[0])

    @classmethod
    def from_text(cls, text: str, ignore_long=False, cuda=True) -> "EncodedText":
        self = cls.__new__(cls)
        tokens = clip.tokenize(text, truncate=ignore_long)
        if cuda:
            tokens = tokens.cuda()

        self.__init__(text, tokens[0])
        return self

    @classmethod
    def from_text_list(cls, text_lst: typing.List[str], ignore_long=False, cuda=True) -> typing.List["EncodedText"]:
        if type(text_lst) != list:
            text_lst = [text_lst]
        ret = [cls.from_text(x, ignore_long, cuda) for x in text_lst]
        return [x for x in ret if x is not None and x.get_tokens_len() > 2]

    def __add__(self, other: typing.Any) -> typing.Tuple[None, "EncodedText"]:
        words = f"{self._text} {other._text}"

        end_idx = self.get_tokens_len()
        new_tokens = self.get_tokens().clone()
        end_code = new_tokens[end_idx - 1]
        rem_space = len(new_tokens) - end_idx
        tokens_to_copy = other.get_tokens()[1 : other.get_tokens_len()]
        # Copy over the last
        to_copy = min(len(tokens_to_copy), rem_space)
        new_tokens[end_idx - 1 : end_idx - 1 + to_copy] = tokens_to_copy[:to_copy]
        if len(tokens_to_copy) > rem_space:
            # print("Warning: truncting tokens after adding")
            new_tokens[-1] = end_code

        return EncodedText(words, new_tokens)

    @staticmethod
    def words_from_encoded_list(encoded_lst):
        return [encoded.get_text() for encoded in encoded_lst]

    def __hash__(self) -> int:
        return self.get_text().__hash__()

    def __str__(self):
        return f"{self.get_text()}: {self.get_nonzero_tokens()}"

    def __repr__(self):
        return str(self)
