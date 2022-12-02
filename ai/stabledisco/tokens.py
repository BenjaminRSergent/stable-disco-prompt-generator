import clip
import open_clip
import torch
import ai.stabledisco.utils as sdutils


class Tokens(torch.Tensor):
    @staticmethod
    def __new__(cls, tens, clone=False):
        ret = tens
        if clone:
            ret = ret.clone()
        ret = ret.long()
        ret.__class__ = cls
        return ret

    @classmethod
    def from_str(cls, prompt, cuda=True):
        self = Tokens(open_clip.tokenize(prompt)[0].long())
        if cuda:
            return self.cuda()
        else:
            return self

    def is_reversed(self):
        return sdutils.is_rev_tokens(self)

    def get_end_idx(self):
        return sdutils.find_end_idx(self)

    def as_reversed(self, rev):
        return sdutils.change_rev(self, rev)[0]

    def decode(self):
        return sdutils.decode_tokens(self)[0]

    def append(self, token):
        pass

    def trim(self, clip_model, other=None):
        pass

    def truncate(self, include_end=False):
        end_idx = self.get_end_idx()
        if end_idx == -1:
            return self

        if include_end:
            end_idx += 1

        ret = self[:end_idx]
        print(ret)
        return ret

    def insert(self, pos, token):
        pass

    def complete(self):
        pass

    def is_complete(self):
        pass
