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


        dense_stack_units = [(3, self._transformer_width*2),
                             (3, self._transformer_width*4),
                             (1, self._transformer_width*6),
                             (1, self._transformer_width*8)]
        self._latent_to_latent_dense_stack = torchlayers.DenseStack(
            self._transformer_width,
            dense_stack_units,
            activation=nn.LeakyReLU,
            dropout=0.25,
        )
        self._seq_expander = torchlayers.LinearWithActivation(
             dense_stack_units[-1][1],
            self._seq_len * self._transformer_width,
            dropout=0.25,
            batch_norm_type=None,
        )

        self._seq_reshaper = torchmodules.layers.Reshaper(
            (-1, self._seq_len, self._transformer_width)
        )
        self._pre_encode_ln = nn.LayerNorm(self._transformer_width)
        
        block_width = self._transformer_width * 4
        self._encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self._transformer_width,
                nhead=self._transformer_heads,
                dim_feedforward=block_width,
                dropout=0.25,
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
            num_layers=int(self._transformer_layers*1.5),
        )

        self._vocab_out = torch.nn.Linear(self._transformer_width, self._vocab_size)
        nn.init.xavier_uniform_(self._vocab_out.weight)

        self._loss_func = nn.CrossEntropyLoss(ignore_index=0)

        base_learning = 9e-5
        self._optimizer = torch.optim.NAdam(
            self.parameters(), base_learning, betas=(0.88, 0.998)
        )

        self._scheduler = torch.optim.lr_scheduler.CyclicLR(
            self._optimizer,
            base_lr=base_learning / 6,
            max_lr=base_learning,
            step_size_up=22500,
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
