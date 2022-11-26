import numpy as np
import torch
from open_clip.tokenizer import _tokenizer as clip_tokenizer


def decode_clip_tokens(tokens):
    if isinstance(tokens, np.ndarray):
        tokens = torch.from_numpy(tokens)
    if isinstance(tokens, list):
        tokens = torch.stack(tuple(tokens))
    if len(tokens.shape) == 1:
        tokens = tokens.view(1, *tokens.shape)

    texts = [clip_tokenizer.decode(toks[1:].cpu().numpy()) for toks in tokens]
    ends = [text.find("<end_of_text>") for text in texts]
    for idx in range(len(texts)):
        if ends[idx] != -1:
            texts[idx] = texts[idx][: ends[idx]]

    return texts
