import torch
from clip.clip import _tokenizer as clip_tokenizer

_eot_token = clip_tokenizer.encoder["<|endoftext|>"]

class PromptUpgrader:
    def __init__(self, tokens_model, clip_model):
        self._tokens_model = tokens_model
        self._clip_model = clip_model

    def upgrade(self, target_features, tokens, pass_cnts=None, ascii_only=True):

        target_features = target_features.view(1,-1)
        memory = self._tokens_model.features_to_memory(target_features).squeeze(0)

        if pass_cnts is None:
            pass_cnts = [10, 100, 1000]

        for curr_end in range(1, tokens.size(0)-1):
            if tokens[curr_end] == _eot_token: 
                break
            for single_rep_cands in pass_cnts:
                replacement_probs = []

                # Two passes to allow async GPU processing before getting the top k of each
                replacement_probs.append(self._tokens_model.get_next_probs(memory, tokens[:curr_end].unsqueeze(0), ascii_only=ascii_only)[0])

                to_test = [tokens.clone()]
                for idx in range(len(replacement_probs)):
                    _, candidate_tokens = replacement_probs[idx].topk(single_rep_cands, dim=-1)
                    for replacement_token in candidate_tokens:
                        new_tensor = tokens.clone()

                        new_tensor[curr_end] = replacement_token
                        to_test.append(new_tensor)

                # TODO: Fix duplcate work
                test_stack = torch.stack(tuple(to_test))
                top_tokens, top_sim = self._clip_model.rank_similarity(target_features, test_stack, 1)

                tokens = top_tokens[0]

            if curr_end % 10 == 0:
                print(f"Scanned {curr_end} tokens")

        return tokens, top_sim[0]


            
