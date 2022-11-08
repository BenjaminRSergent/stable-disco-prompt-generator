import re

import ai.torchmodules as torchmodules
import ai.torchmodules.layers as torchlayers
import ai.torchmodules.scheduler as torchscheduler
import torch
import torch.nn as nn
from clip.clip import _tokenizer as clip_tokenizer

_eot_token = clip_tokenizer.encoder["<|endoftext|>"]


class FeaturesToTokensAesResModel(torchmodules.BaseModel):
    def __init__(
        self, clip_model, transformer_width=768, seq_len=77, vocab_size=49408, heads=12, layers=12, device=None
    ):
        super().__init__("FeaturesToTokensAesResModelV1", device=device)

        self._ascii_mask = None
        self._dtype = clip_model.dtype

        self._transformer_width = transformer_width
        self._transformer_heads = heads
        self._transformer_layers = layers

        self._token_embedding = nn.Embedding(vocab_size, transformer_width)
        self._embedding_dropout = nn.Dropout(0.15)
        self._positional_embedding = clip_model.positional_embedding

        self._seq_len = seq_len
        self._vocab_size = vocab_size

        self._seq_expander = torchlayers.LinearWithActivation(
            self._transformer_width,
            self._seq_len * transformer_width,
            dropout=0.1,
            activation=torchlayers.QuickGELU,
            batch_norm_type=None,
        )

        self._seq_reshaper = torchmodules.layers.Reshaper((-1, self._seq_len, transformer_width))

        self._pre_encode_ln = nn.LayerNorm(transformer_width)
        self._res_block = torchlayers.ResDenseStack(
            transformer_width,
            transformer_width * 2,
            layers=4,
            activation=torchlayers.QuickGELU,
            dropout=0.1,
            batch_norm_type=torchlayers.Normalization.NormType.LAYER,
        )

        block_width = transformer_width * 4
        self._encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=transformer_width,
                nhead=self._transformer_heads,
                dim_feedforward=block_width,
                activation=torchlayers.QuickGELU(),
                dropout=0.075,
                batch_first=True,
            ),
            num_layers=self._transformer_layers,
        )

        block_width = transformer_width * 4
        self._decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=transformer_width,
                nhead=self._transformer_heads,
                dim_feedforward=block_width,
                activation=torchlayers.QuickGELU(),
                dropout=0.075,
                batch_first=True,
            ),
            num_layers=3 * self._transformer_layers,
        )

        self._vocab_out = torch.nn.Linear(transformer_width, self._vocab_size)
        nn.init.xavier_uniform_(self._vocab_out.weight)

        self._loss_func = nn.CrossEntropyLoss(ignore_index=0)

        base_learning = 1e-3
        self._optimizer = torch.optim.NAdam(self.parameters(), base_learning, betas=(0.89, 0.998))

        self._scheduler = torchscheduler.make_cyclic_with_warmup(
            optimizer=self._optimizer,
            epoch_batches=116734,
            max_lr=base_learning,
            min_lr_divisor=10,
            step_size_up_epoch_mul=0.5,
            warmup_period_epoch_mul=0.25,
            gamma=0.75,
            cycle_momentum=False,
        )

        target_mask = nn.Transformer.generate_square_subsequent_mask(self._seq_len).cuda()
        self.register_buffer("_target_mask", target_mask)

    def _calc_batch_loss(self, x_inputs: torch.Tensor, y_targets: torch.Tensor):
        with torch.autocast(device_type="cuda"):
            outputs = self((x_inputs, y_targets))
            if y_targets.dtype.is_floating_point:
                return self._loss_func(outputs.permute(0, 2, 1), y_targets)
            else:
                return self._loss_func(outputs.permute(0, 2, 1)[:, :, :-1], y_targets[:, 1:].long())

    def forward(self, x_inputs):
        latent_img_features, tgt_tokens = x_inputs
        latent_img_features = latent_img_features / latent_img_features.norm(dim=-1, keepdim=True)
        encoder_out = self.features_to_memory(latent_img_features)

        tgt = self._embedding_dropout(self._token_embedding(tgt_tokens)) + self._positional_embedding

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

        x = self._seq_expander(latent_img_features.float())

        seq_features = self._seq_reshaper(x) + self._positional_embedding
        seq_features = self._pre_encode_ln(seq_features)

        return self._encoder(self._res_block(seq_features))

    def generate_square_subsequent_mask(self, size):
        attn_shape = (1, size, size)
        subsequent_mask = torch.triu(torch.ones(attn_shape, device=self._device), diagonal=1)
        return subsequent_mask == 0

    def decode(self, tokens):
        if isinstance(tokens, list):
            tokens = torch.stack(tuple(tokens))
        if len(tokens.shape) == 1:
            tokens = tokens.view(1, *tokens.shape)

        if tokens.shape[-1] == self._vocab_size:
            tokens = self.tokens_from_output(tokens)

        texts = [clip_tokenizer.decode(toks.cpu().numpy()) for toks in tokens]
        for idx in range(len(texts)):
            texts[idx] = texts[idx].replace("<|startoftext|>", "")
            end_idx = texts[idx].find("<|endoftext|>")

            if end_idx != -1:
                texts[idx] = texts[idx][:end_idx]

        return texts

    def get_next_probs(self, memory, curr_tokens, ascii_only=True):
        num_batch = curr_tokens.size(0)
        size = curr_tokens.size(1)
        if size == self._seq_len - 1:
            probs = torch.zeros(len(clip_tokenizer.encoder), device=self._device)
            probs[_eot_token] = 1
            return probs.repeat((curr_tokens.shape[0], 1))

        curr_embedded = self._token_embedding(curr_tokens)
        curr_embedded = curr_embedded + self._positional_embedding[:size]
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
        else:
            print("Asked for non-ascii!")
            raise Exception()

        probs[:, -1] = 0
        probs[:, -2] = 0

        return probs

    def _get_ascii_mask(self):
        if self._ascii_mask is None:
            print("\n\n\n\n\n\n\n")
            self._ascii_mask = torch.ones(len(clip_tokenizer.encoder), device="cuda")

            r"""
            norm_char_regex = re.compile(
                r"^[a-zA-Z0-9 ,\.!\"\'\?():;_-{|}<=>]*$"
            )
            alphanum_regex = re.compile(
                r"^[a-zA-Z0-9 ]*$"
            )
            
            norm_char_regex = re.compile(
                r"^[a-zA-Z0-9 !\"#$%&'()*+,\-./:;<=>?@[\]^_`{|}~\\]*$"
            )
            """
            norm_char_regex = re.compile(r"^[a-zA-Z0-9#\$=+%@\^ ,\.!\"\'\?():;_-{|}<=>]*$")
            num_ascii = 0
            for token in clip_tokenizer.decoder.keys():
                text = clip_tokenizer.decode([token])
                # is_ascii = norm_char_regex.match(text) and text[-1] == ' '
                is_ascii = " " in text[-1] and (norm_char_regex.match(text) is not None)

                if is_ascii:
                    num_ascii += 1
                else:
                    self._ascii_mask[token] = 0

            print("Total ", num_ascii)
        return self._ascii_mask


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
