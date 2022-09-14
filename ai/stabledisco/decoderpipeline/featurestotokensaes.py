import re

import ai.torchmodules as torchmodules
import ai.torchmodules.layers as torchlayers
import ai.torchmodules.utils as torchutils
import torch
import torch.nn as nn
from clip.clip import _tokenizer as clip_tokenizer

_sot_token = clip_tokenizer.encoder["<|startoftext|>"]
_eot_token = clip_tokenizer.encoder["<|endoftext|>"]


class FeaturesToTokensAesModel(torchmodules.BaseModel):
    def __init__(self, clip_model, device=None):
        super().__init__("FeaturesToTokensAesModelV1")

        self._ascii_mask = None
        self._dtype = clip_model.dtype
        if device is None:
            device = torchutils.get_default_device()
        self._device = device

        self._clip_model = clip_model
        for param in self._clip_model.parameters():
            param.requires_grad = False

        clip_state_dict = self._clip_model.state_dict()
        self._seq_len = clip_state_dict["positional_embedding"].shape[0]
        self._vocab_size = clip_state_dict["token_embedding.weight"].shape[0]

        self._transformer_width = clip_state_dict["ln_final.weight"].shape[0]
        self._transformer_heads = self._transformer_width // 64
        self._transformer_layers = len(
            set(
                k.split(".")[2]
                for k in clip_state_dict
                if k.startswith("transformer.resblocks")
            )
        )

        block_width = self._transformer_width * 6

        self._latent_to_latent_dense_stack = torchlayers.ResDenseStack(
            self._transformer_width, block_width, self._transformer_layers, dropout=0.2
        )
        self._seq_expander = torchlayers.LinearWithActivation(
            self._transformer_width,
            self._seq_len * self._transformer_width,
            dropout=0.2,
            batch_norm_type=None,
        )

        self._seq_reshaper = torchmodules.layers.Reshaper(
            (-1, self._seq_len, self._transformer_width)
        )
        self._pre_encode_ln = nn.LayerNorm(self._transformer_width)

        self._encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self._transformer_width,
                nhead=self._transformer_heads,
                dim_feedforward=block_width,
                dropout=0.2,
                batch_first=True,
            ),
            num_layers=self._transformer_layers,
        )

        self._decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=self._transformer_width,
                nhead=self._transformer_heads,
                dim_feedforward=block_width,
                dropout=0.25,
                batch_first=True,
            ),
            num_layers=self._transformer_layers,
        )

        self._vocab_out = torch.nn.Linear(self._transformer_width, self._vocab_size)
        nn.init.xavier_uniform_(self._vocab_out.weight)

        self._loss_func = nn.CrossEntropyLoss(ignore_index=0)

        base_learning = 9e-5
        self._optimizer = torch.optim.NAdam(
            self.parameters(), base_learning, betas=(0.85, 0.985)
        )

        self._scheduler = torch.optim.lr_scheduler.CyclicLR(
            self._optimizer,
            base_lr=base_learning / 25,
            max_lr=base_learning,
            step_size_up=10000,
            mode="triangular",
            cycle_momentum=False,
        )

        target_mask = nn.Transformer.generate_square_subsequent_mask(
            self._seq_len
        ).cuda()
        self.register_buffer("_target_mask", target_mask)

    def _calc_batch_loss(self, x_inputs, y_targets):
        with torch.autocast(device_type="cuda"):
            outputs = self((x_inputs, y_targets))
            return self._loss_func(
                outputs.permute(0, 2, 1)[:, :, :-1], y_targets[:, 1:].long()
            )

    def forward(self, x_inputs):
        latent_img_features, tgt_tokens = x_inputs
        latent_img_features = latent_img_features / latent_img_features.norm(
            dim=-1, keepdim=True
        )
        encoder_out = self.features_to_memory(latent_img_features)

        tgt = (
            self._clip_model.token_embedding(tgt_tokens)
            + self._clip_model.positional_embedding
        )

        decoder_out = self._decoder(
            memory=encoder_out,
            tgt=tgt,
            tgt_mask=self._target_mask,
            tgt_key_padding_mask=(tgt_tokens == 0),
        )

        vocab_out = self._vocab_out(decoder_out)

        return vocab_out

    def features_to_memory(self, latent_img_features, dtype=None):
        if not dtype:
            dtype = self._dtype
        # latent_img_features = latent_img_features.to(dtype)

        x = self._latent_to_latent_dense_stack(latent_img_features.float())
        x = self._seq_expander(x)

        seq_features = self._seq_reshaper(x) + self._clip_model.positional_embedding
        seq_features = self._pre_encode_ln(seq_features)

        return self._encoder(seq_features)

    def generate_square_subsequent_mask(self, size):
        attn_shape = (1, size, size)
        subsequent_mask = torch.triu(
            torch.ones(attn_shape, device=self._device), diagonal=1
        )
        return subsequent_mask == 0

    def decode(self, tokens):
        if isinstance(tokens, list):
            tokens = torch.stack(tuple(tokens))
        if len(tokens.shape) == 1:
            tokens = tokens.view(1, *tokens.shape)

        if tokens.shape[-1] == self._vocab_size:
            tokens = self.tokens_from_output(tokens)

        texts = [clip_tokenizer.decode(toks[1:].cpu().numpy()) for toks in tokens]
        ends = [text.find("<|endoftext|>") for text in texts]
        for idx in range(len(texts)):
            if ends[idx] != -1:
                texts[idx] = texts[idx][: ends[idx]]

        return texts

    def get_next_probs(self, memory, curr_tokens, ascii_only=False):
        num_batch = curr_tokens.size(0)
        size = curr_tokens.size(1)
        if size == self._seq_len - 1:
            probs = torch.zeros(len(clip_tokenizer.encoder), device=self._device)
            probs[_eot_token] = 1
            return probs.repeat((curr_tokens.shape[0], 1))

        curr_embedded = self._clip_model.token_embedding(curr_tokens)
        curr_embedded = curr_embedded + self._clip_model.positional_embedding[:size]
        curr_embedded = curr_embedded
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(size).cuda()
        memory = torch.cat(num_batch * [memory]).view(num_batch, 77, 768)

        decoder_out = self._decoder(
            tgt=curr_embedded,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=(curr_tokens == 0),
        )

        vocab_out = self._vocab_out(decoder_out[:, -1])

        probs = torch.softmax(vocab_out, dim=-1)
        if ascii_only:
            probs *= self._get_ascii_mask()
        probs[:, -1] = 0
        probs[:, -2] = 0

        return probs

    def _get_ascii_mask(self):
        if self._ascii_mask is None:
            self._ascii_mask = torch.ones(len(clip_tokenizer.encoder), device="cuda")
            norm_char_regex = re.compile(
                r"^[a-zA-Z0-9 !\"#$%&'()*+,\-\./:;<=>?@[\]^_`{|}~\\]*$"
            )
            num_ascii = 0
            for token in clip_tokenizer.decoder.keys():
                text = clip_tokenizer.decode([token])
                if norm_char_regex.match(text):
                    num_ascii += 1
                else:
                    self._ascii_mask[token] = 0

        return self._ascii_mask

    def beam_search(
        self,
        rating_model,
        clip_model,
        features,
        rating_weight=1.25,
        clip_weight=8,
        model_beams=40,
        clip_beams=40,
        num_inter_beams=3000,
        topk=1,
        add_beams_to_final=False,
        feature_weights=None,
        ascii_only=True,
        verbose=True,
    ):
        epsilon = 1e-8
        self.train(False)
        with torch.no_grad():
            features = features.float()

            if feature_weights is None:
                feature_weights = torch.ones(
                    (features.shape[0], 1), device=features.device
                )
            features = (
                torch.sum(features * feature_weights.view(-1, 1), dim=0)
                / features.size(0)
            ).unsqueeze(0)
            orig_features_norm = features.norm(dim=-1, keepdim=True)
            features = features / orig_features_norm
            orig_features_norm = torch.sum(
                orig_features_norm
            ) / orig_features_norm.size(0)

            memory = self.features_to_memory(features).squeeze(0)

            torch_zero = torch.tensor([0.0], device=self._device)
            torch_start = torch.tensor(
                [_sot_token], device=self._device, dtype=torch.long
            )
            curr_beams = [(torch_zero, torch_start)]
            final_beam_tokens = []

            beams = model_beams + clip_beams
            best_match_cosine = torch.tensor(0, device=self._device)

            tokens_to_find = self._seq_len - 2
            for iter_num in range(tokens_to_find):
                curr_tokens = torch.stack(tuple((x[1] for x in curr_beams)))
                token_probs = torch.log(
                    self.get_next_probs(memory, curr_tokens, ascii_only=ascii_only)
                    + epsilon
                )

                token_probs = (
                    token_probs.size(-1)
                    * token_probs
                    / token_probs.norm(dim=-1, keepdim=True)
                )
                new_probs = tuple(
                    curr_beams[idx][0] + token_probs[idx]
                    for idx in range(len(curr_beams))
                )
                next_probs = torch.cat(new_probs).view(-1)

                new_beam_probs, args = next_probs.topk(num_inter_beams, dim=-1)
                args = torchutils.unravel_torch_idx(
                    args, next_probs.shape[0] // len(curr_beams)
                )
                prob_beam_token = [
                    (prob, curr_beams[arg[0]][1], arg[1])
                    for prob, arg in zip(new_beam_probs, args)
                ]

                next_beam_probs = torch.cat(
                    tuple((x[0].unsqueeze(0) for x in prob_beam_token))
                )
                model_arg_sort = (
                    torch.argsort(next_beam_probs, dim=-1).cpu().numpy().tolist()
                )
                prob_beam_token = [prob_beam_token[idx] for idx in model_arg_sort]

                model_args = model_arg_sort[-model_beams:]

                next_beam_tokens = []
                next_beam_tokens_full = torch.zeros(
                    len(prob_beam_token), 77, device=self._device, dtype=torch.long
                )
                idx = 0
                for prob, last_tokens, new_token in prob_beam_token:
                    if len(last_tokens) == self._seq_len - 1:
                        new_token = _eot_token

                    new_beam = torch.cat(
                        (last_tokens, torch.tensor([new_token], device=self._device))
                    )
                    next_beam_tokens.append(new_beam)
                    next_beam_tokens_full[idx] = add_token_and_end(
                        last_tokens, new_token
                    )
                    idx += 1

                # The first iteration has the start token place the first selection
                eot_idx = iter_num + 2
                next_beam_probs_aug = clip_model.cosine_similarity(
                    features, next_beam_tokens_full, end_idx=eot_idx, verbosity=2
                )
                probs, clip_args = next_beam_probs_aug.topk(beams, dim=-1)
                if probs[0] > best_match_cosine:
                    # TODO: Optimize. Assigning here takes crazy time
                    best_match_cosine = probs[0]
                    top_tokens = next_beam_tokens_full[clip_args[0]]
                    final_beam_tokens.append(top_tokens)

                if clip_weight != 0:
                    clip_bonus = torch.softmax(next_beam_probs_aug, dim=-1)
                    clip_bonus = torch.log(clip_bonus + epsilon) * clip_weight
                    new_beam_probs += clip_bonus

                if rating_weight != 0:
                    aug_ratings = rating_model(
                        clip_model.features_from_tokens(next_beam_tokens_full)
                    ).view(-1)
                    aug_rating_probs = torch.softmax(aug_ratings, dim=-1)
                    aug_ratings_bonus = (
                        torch.log(aug_rating_probs + epsilon) * rating_weight
                    )
                    new_beam_probs += aug_ratings_bonus

                _, model_args = (new_beam_probs).topk(model_beams, dim=-1)

                clip_args = clip_args.cpu().numpy()
                clip_args = [
                    arg
                    for arg in clip_args
                    if arg not in model_args and prob_beam_token[arg][0] != _eot_token
                ][:clip_beams]
                top_args = model_args.cpu().numpy().tolist() + clip_args
                curr_beams = []
                for top_idx in top_args:
                    prob = prob_beam_token[top_idx][0]
                    new_beam = next_beam_tokens[top_idx]
                    curr_beams.append((prob, new_beam))
                    if add_beams_to_final:
                        # Used for experimental evolutionary algorithms
                        final_beam_tokens.append(next_beam_tokens_full[top_idx])

                if verbose and (iter_num % 10 == 0 or iter_num == tokens_to_find - 1):
                    print(f"{iter_num} of {self._seq_len} tokens searched")
                    rating = rating_model(
                        clip_model.features_from_tokens(
                            final_beam_tokens[-1].view(1, -1)
                        )
                    )[0].item()
                    print(
                        f"curr top {best_match_cosine: 0.4f} with estimated quality {rating: 0.2f}: {self.decode(final_beam_tokens[-1])}"
                    )

            top_k = min(topk, len(final_beam_tokens))
            final_beam_tokens, final_probs = clip_model.rank_similarity(
                features, torch.stack(tuple(final_beam_tokens)).long(), top_k
            )

            best = final_beam_tokens

            text_features = text_features = clip_model.features_from_tokens(best)
            scale_diff = (orig_features_norm / text_features.norm(dim=-1)).cpu().numpy()

            return self.decode(best), scale_diff, final_probs


def add_token_and_end(curr_tokens, new_token, max_size=77):
    if curr_tokens.size(0) >= max_size:
        return curr_tokens

    device = curr_tokens.get_device()

    possible_token = torch.tensor([new_token], device=device, dtype=torch.long)
    if new_token == _eot_token:
        ret = torch.cat((curr_tokens, possible_token))
    else:
        ret = torch.cat(
            (
                curr_tokens,
                possible_token,
                torch.tensor([_eot_token], device=device, dtype=torch.long),
            )
        )

    rem_tokens = max(max_size - ret.size(0), 0)
    if rem_tokens > 0:
        ret = torch.cat((ret, torch.zeros(rem_tokens, device=device, dtype=torch.long)))

    return ret
