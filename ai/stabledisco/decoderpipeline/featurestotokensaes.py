import copy
import re
import typing

import ai.stabledisco.constants as sdconsts
import ai.torchmodules as torchmodules
import ai.torchmodules.layers as torchlayers
import torch
import torch.nn as nn
from ai.stabledisco.decoderpipeline.knowledgetransfernetwork import \
    KnowledgeTransferNetwork
from ai.stabledisco.decoderpipeline.lowerfeaturelayers import \
    LowerFeatureLayers
from clip.clip import _tokenizer as clip_tokenizer


class FeaturesToTokensAesModel(torchmodules.BaseModel):
    name = "FeaturesToTokensAesModelV2"
    
    @staticmethod
    def build_large(clip_model: nn.Module, freeze_embedding=True):
        return FeaturesToTokensAesModel(clip_model, freeze_embedding=freeze_embedding)
    
    @staticmethod
    def build_medium(clip_model: nn.Module, freeze_embedding=True):
        return FeaturesToTokensAesModel(clip_model, name_suffix="Student", block_mul=3, layers=4, decoder_mul=2, heads=12, freeze_embedding=freeze_embedding)
    
    @staticmethod
    def build_small(clip_model: nn.Module, freeze_embedding=True):
        return FeaturesToTokensAesModel(clip_model, name_suffix="Student", block_mul=2, layers=3, decoder_mul=2.5, heads=12, freeze_embedding=freeze_embedding)
    
    @staticmethod
    def build_knowledge_transfer_model(clip_model: nn.Module, teacher=None,  teacher_checkpoint="best"):
        if teacher is None:
            teacher = FeaturesToTokensAesModel.build_medium(clip_model)
            teacher.load_weights(teacher_checkpoint, strict=False)
        
        student = FeaturesToTokensAesModel.build_small(clip_model)
        student.initialize_student(teacher)
        
        return KnowledgeTransferNetwork(student, teacher, name_suffix=FeaturesToTokensAesModel.name)
        
    def __init__(self, clip_model: nn.Module, name_suffix="",
                 lower_dropout=0.1, upper_dropout=0.05,
                 block_mul=4,
                 heads=12, layers=12, decoder_mul=1.5,
                 device=None, freeze_embedding=True):
        super().__init__(FeaturesToTokensAesModel.name+name_suffix, device=device)

        self._ascii_mask = None
        self._banned_mask = None
        self._bool_to_mask = {}
        self._dtype = clip_model.dtype

        self._clip_model = clip_model
        
        for param in self._clip_model.parameters():
            param.requires_grad = False

        self._seq_len = sdconsts.prompt_token_len

        self._transformer_heads = heads
        self._transformer_layers = layers
        
        self._token_embedding = nn.Embedding(sdconsts.num_tokens, sdconsts.feature_width)
        self._token_embedding.weight.data = clip_model.token_embedding.weight.data.clone()
        
        for param in self._clip_model.parameters():
            param.requires_grad = not freeze_embedding
        
        self.register_buffer("_positional_embedding", clip_model.positional_embedding)
                             
        self._feature_expander = LowerFeatureLayers(dropout=lower_dropout)

        self._seq_expander = torchlayers.LinearWithActivation(
            self._feature_expander.out_features,
            self._seq_len * sdconsts.feature_width,
            dropout=lower_dropout,
            batch_norm_type=None,
        )

        self._seq_reshaper = torchlayers.Reshaper(
            (-1, self._seq_len, sdconsts.feature_width)
        )
        self._pre_encode_ln = nn.LayerNorm(sdconsts.feature_width)

        block_width = int(sdconsts.feature_width * block_mul)
        self._encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=sdconsts.feature_width,
                nhead=self._transformer_heads,
                dim_feedforward=block_width,
                activation=torchlayers.QuickGELU(),
                dropout=upper_dropout,
                batch_first=True,
            ),
            num_layers=self._transformer_layers,
        )

        self._decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=sdconsts.feature_width,
                nhead=self._transformer_heads,
                dim_feedforward=block_width,
                activation=torchlayers.QuickGELU(),
                dropout=0.0,
                batch_first=True,
            ),
            num_layers=int(self._transformer_layers*decoder_mul),
        )

        self._vocab_out = torch.nn.Linear(sdconsts.feature_width, sdconsts.num_tokens)
        nn.init.xavier_uniform_(self._vocab_out.weight)

        self._loss_func = nn.CrossEntropyLoss()

        base_learning = 1e-4
        self._optimizer = torch.optim.NAdam(
            self.parameters(), base_learning, betas=(0.89, 0.998)
        )

        self._scheduler = torch.optim.lr_scheduler.CyclicLR(
            self._optimizer,
            base_lr=base_learning / 6,
            max_lr=base_learning,
            step_size_up=100000,
            mode="triangular",
            cycle_momentum=False,
        )

        target_mask = nn.Transformer.generate_square_subsequent_mask(
            self._seq_len
        ).cuda()
        self.register_buffer("_target_mask", target_mask)
        
    def initialize_student(self, teacher: nn.Module):
        self._feature_expander = copy.deepcopy(teacher._feature_expander)
        self._feature_expander.freeze()
        
        self._seq_expander = copy.deepcopy(teacher._seq_expander)
        for param in self._seq_expander.parameters():
            param.requires_grad = False
        
        self._token_embedding.weight.data = teacher._token_embedding.weight.data.clone()
        
        for param in self._token_embedding.parameters():
            param.requires_grad = False

    def _calc_batch_loss(self, x_inputs: torch.Tensor, y_targets: torch.Tensor, target_prob: torch.Tensor = None):
        with torch.autocast(device_type="cuda"):
            outputs = self((x_inputs, y_targets.long()))
            if y_targets.dtype.is_floating_point:
                return self._loss_func(
                outputs.permute(0, 2, 1), y_targets
                )
            else:
                return self._loss_func(
                    outputs.permute(0, 2, 1)[:, :, :-1], y_targets[:, 1:].long()
                )

            
    def set_clip_model(self, clip_model: nn.Module):
        self._clip_model = clip_model
        

    def forward(self, x_inputs, y_targets=None, reverse=False):
        if y_targets is None:
            latent_img_features, tgt_tokens = x_inputs
        else:
            latent_img_features, tgt_tokens = x_inputs, y_targets
        latent_img_features = latent_img_features / latent_img_features.norm(
            dim=-1, keepdim=True
        )
        encoder_out = self.features_to_memory(latent_img_features)

        tgt = (
            self._token_embedding(tgt_tokens)
            + self._positional_embedding
        )

        if reverse:
            # TODO: implement predicting previous token from the end
            raise NotImplementedError("Reverse Decoding is not implemented")
            """
            decoder_out = self._rev_decoder(encoder_out, 
                memory=encoder_out,
                tgt=tgt,
                tgt_mask=self._target_mask,
                tgt_key_padding_mask=(tgt_tokens == 0),
            )
            """
        else:
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

        x = self._feature_expander(latent_img_features)
        x = self._seq_expander(x)

        seq_features = self._seq_reshaper(x) + self._positional_embedding
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

        if tokens.shape[-1] ==  sdconsts.num_tokens:
            tokens = self.tokens_from_output(tokens)

        texts = [clip_tokenizer.decode(toks.cpu().numpy()) for toks in tokens]
        for idx in range(len(texts)):
            texts[idx] = texts[idx].replace("<|startoftext|>", "")
            end_idx = texts[idx].find("<|endoftext|>")

            if end_idx != -1:
                texts[idx] = texts[idx][:end_idx]

        return texts

    # TODO: custom mask
    def get_next_probs(self, memory, curr_tokens, ascii_only=True, no_banned=True, custom_mask=None):
        num_batch = curr_tokens.size(0)
        size = curr_tokens.size(1)
        if size == self._seq_len - 1:
            probs = torch.zeros(len(clip_tokenizer.encoder), device=self._device)
            probs[sdconsts.eot_token] = 1
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
        if custom_mask:
            probs *= custom_mask
        elif ascii_only or no_banned:
            probs *= self.get_mask(ascii_only, no_banned)
            
        probs[:, -1] = 0
        probs[:, -2] = 0

        return probs

    def get_mask(self, ascii_only, no_banned):
        mask_key = self.get_mask_key(ascii_only, no_banned)
        if mask_key in self._mask_dict:
            return self._mask_dict[mask_key]
        
        self._mask_dict[self.get_mask_key(ascii_only=False, no_banned=False)] = torch.ones(len(clip_tokenizer.encoder), device="cuda")
        
        print("Creating ascii mask")
        ascii_mask = torch.ones(len(clip_tokenizer.encoder), device="cuda")
        norm_char_regex = re.compile(
            r"^[a-zA-Z0-9#\$=+%@\^ ,\.!\"\'\?():;_-{|}<=>]*$"
        )
        num_ascii = 0
        for token in clip_tokenizer.decoder.keys():
            text = clip_tokenizer.decode([token])
            #is_ascii = norm_char_regex.match(text) and text[-1] == ' '
            is_ascii = ' ' in text[-1] and (norm_char_regex.match(text) is not None)

            if is_ascii:
                num_ascii += 1
            else:
                ascii_mask[token] = 0
        print("Total ascii", num_ascii)
        
        self._mask_dict[self.get_mask_key(ascii_only=True, no_banned=False)] = ascii_mask
        
        banned_mask = torch.ones(len(clip_tokenizer.encoder), device="cuda")
        banned_words = ["nude", "naked", "kid", "child", "lolita", "cum", "xxx", "anus"]
        for word in banned_words:
            banned_mask[clip_tokenizer.encoder[word + '</w>']] = 0
            
        self._mask_dict[self.get_mask_key(ascii_only=False, no_banned=True)] = banned_mask 
        self._mask_dict[self.get_mask_key(ascii_only=True, no_banned=True)] = banned_mask * ascii_mask
            
        return self._mask_dict[mask_key]
    
    @staticmethod
    def get_mask_key(ascii_only, no_banned):
        return (ascii_only, no_banned)


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
