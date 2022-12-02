from open_clip.tokenizer import _tokenizer as clip_tokenizer

vit_l_feature_width = 768
feature_width = 1024

pruned_expander_out = 5202
sot_token = clip_tokenizer.encoder["<start_of_text>"]
eot_token = clip_tokenizer.encoder["<end_of_text>"]
num_tokens = len(clip_tokenizer.encoder)
prompt_token_len = 77
