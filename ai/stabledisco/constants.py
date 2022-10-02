from clip.clip import _tokenizer as clip_tokenizer

feature_width = 768

sot_token = clip_tokenizer.encoder["<|startoftext|>"]
eot_token = clip_tokenizer.encoder["<|endoftext|>"]
num_tokens = len(clip_tokenizer.encoder)
prompt_token_len = 77
