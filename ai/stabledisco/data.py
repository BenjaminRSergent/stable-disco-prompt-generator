from enum import Enum

import ai.stabledisco.utils as sdutils
import ai.torchmodules.data as torchdata
import ai.torchmodules.utils as torchutils
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class PipelineStage(Enum):
    IMAGE_LATENT = 0
    TOKEN_LATENT = 1
    TRANSFORMED = 2
    EMB_TOKENS = 3
    TOKENS = 4

    def __eq__(self, other: object) -> bool:
        return other.value == self.value


class PipelineChunk(torchdata.CudaChunk):
    def __init__(self, tokens, clip_model, in_stage, out_stage, chunk_size=64 * 5):
        super().__init__(tokens, chunk_size)
        self._clip_model = clip_model
        self._in_stage = in_stage
        self._out_stage = out_stage
        self._get_chunk_func = self._get_mixed_for_range

        if self._in_stage == self._out_stage:
            if self._in_stage == PipelineStage.TOKENS:
                self._get_chunk_func = self._get_tokens_for_range
            if self._in_stage == PipelineStage.TOKEN_LATENT:
                self._get_chunk_func = self._get_features_for_range

    def _get_chunk_for_range(self, start_idx, end_idx):
        return self._get_chunk_func(start_idx, end_idx)

    def _get_tokens_for_range(self,  start_idx, end_idx):
        return self._data[start_idx:end_idx].cuda()

    def _get_features_for_range(self,  start_idx, end_idx):
        ret = list(zip(self._data[start_idx:end_idx].cuda(), self._clip_model.encode_text(self._data[start_idx:end_idx].cuda()).half()))
        print(len(ret))
        return ret

    def _get_mixed_for_range(self, start_idx, end_idx):
        in_vals = None
        out_vals = None

        # Return true if done
        def set_if_stage(vals, stage):
            nonlocal in_vals
            nonlocal out_vals
            if stage == self._in_stage:
                in_vals = vals
            if stage == self._out_stage:
                out_vals = vals

            return (in_vals is not None) and (out_vals is not None)

        def get_ret():
            if self._in_stage != self._out_stage:
                return list(zip(in_vals, out_vals))

            return in_vals

        curr_vals = self._data[start_idx:end_idx].cuda()
        ends = curr_vals.argmax(dim=-1)
        if set_if_stage(curr_vals, PipelineStage.TOKENS):
            return get_ret()

        curr_vals = self._clip_model.token_embedding(curr_vals).type(torch.half)
        if set_if_stage(curr_vals, PipelineStage.EMB_TOKENS):
            return get_ret()

        curr_vals = curr_vals + self._clip_model.positional_embedding.type(torch.half)
        curr_vals = curr_vals.permute(1, 0, 2)  # NLD -> LND
        curr_vals = self._clip_model.transformer(curr_vals)
        curr_vals = curr_vals.permute(1, 0, 2)  # LND -> NLD
        curr_vals = self._clip_model.ln_final(curr_vals).type(torch.half)
        if set_if_stage(curr_vals, PipelineStage.TRANSFORMED):
            return get_ret()

        curr_vals = (
            curr_vals[torch.arange(curr_vals.shape[0]), ends]
            @ self._clip_model.text_projection
        )

        if set_if_stage(curr_vals, PipelineStage.TOKEN_LATENT):
            return get_ret()

        raise Exception(
            f"Did not find input and output vals for stages {self._in_stage} -> {self._out_stage}"
        )


class PipelineDataset(Dataset):
    def __init__(
        self,
        tokens,
        clip_model,
        in_stage,
        out_stage,
        start_idx,
        end_idx,
        shuffle=True,
        chunk_size=64 * 10,
    ):
        self._all_tokens = tokens

        self._in_stage = in_stage
        self._out_stage = out_stage
        self._clip_model = clip_model
        self._start_idx = start_idx
        self._end_idx = end_idx
        self._shuffle = shuffle
        self._chunk_size = chunk_size

        self._pipeline_chunk = None
        self._prepare_data()

    def _prepare_data(self):
        if self._shuffle:
            data_idxs = (
                torch.randperm(self._end_idx - self._start_idx) + self._start_idx
            )
        else:
            data_idxs = torch.arange(self._start_idx, self._end_idx)

        if "_pipeline_chunk" in dir(self):
            del self._pipeline_chunk

        self._pipeline_chunk = PipelineChunk(
            self._all_tokens[data_idxs],
            self._clip_model,
            self._in_stage,
            self._out_stage,
            chunk_size=self._chunk_size,
        )

    def __len__(self):
        return self._end_idx - self._start_idx

    def clear(self):
        self._pipeline_chunk = None

    def __getitem__(self, idx):
        if self._pipeline_chunk is None:
            self._prepare_data()

        ret = self._pipeline_chunk[idx]
        if idx == self._end_idx - 1:
            self.clear()

        return ret


class DirectTextFeaturesSet(Dataset):
    def __init__(self, features, tokens, start_idx, end_idx, shuffle=True):
        self._all_features = features
        self._all_tokens = tokens

        self._start_idx = start_idx
        self._end_idx = end_idx
        self._shuffle = shuffle
        self._device = torchutils.get_default_device()

        self._data_idxs = None
        # TODO: Add device?

        self._prepare_data()

    def _prepare_data(self):
        if self._shuffle:
            self._data_idxs = (
                torch.randperm(self._end_idx - self._start_idx) + self._start_idx
            )
        else:
            self._data_idxs = None

    def __len__(self):
        return self._end_idx - self._start_idx

    def clear(self):
        self._prepare_data()

    def __getitem__(self, idx):
        data_idx = self._data_idxs[idx] if self._data_idxs is not None else idx + self._start_idx
        ret = ( self._all_tokens[data_idx], self._all_features[data_idx])
        if idx == self._end_idx - 1:
            self._prepare_data()

        return ret


class ImgFeaturesToTokensSet(Dataset):
    def __init__(self, features, tokens, clip_model, start_idx, end_idx, shuffle=True):
        self._all_features = features.half()
        self._all_tokens = tokens.long()
        self._clip_model = clip_model

        self._start_idx = start_idx
        self._end_idx = end_idx
        self._shuffle = shuffle
        self._device = torchutils.get_default_device()

        self._curr_text_tokens = None
        self._curr_image_features = None

    def _prepare_data(self):
        if self._shuffle:
            data_idxs = (
                torch.randperm(self._end_idx - self._start_idx) + self._start_idx
            )
        else:
            data_idxs = torch.arange(self._start_idx, self._end_idx)

        if self._curr_text_tokens:
            del self._curr_text_tokens
        if self._curr_image_features:
            del self._curr_image_features

        self._curr_text_tokens = PipelineChunk(
            self._all_tokens[data_idxs],
            self._clip_model,
            PipelineStage.TOKENS,
            PipelineStage.TOKENS,
            chunk_size=128 * 96,
        )
        self._curr_image_features = torchdata.CudaChunk(
            self._all_features[data_idxs], chunk_size=128 * 96
        )

    def clear(self):
        self._curr_image_features = None
        self._curr_text_tokens = None

    def __len__(self):
        return self._end_idx - self._start_idx

    def __getitem__(self, idx):
        if self._curr_image_features is None:
            self._prepare_data()
        ret = (self._curr_image_features[idx].to(self._device, non_blocking=True), self._curr_text_tokens[idx].to(self._device, non_blocking=True))
        if idx == self._end_idx - 1:
            self.clear()

        return ret


class TextFeaturesToRatingSet(Dataset):
    def __init__(
        self, tokens, ratings, start_idx, end_idx, shuffle=True
    ):
        self._all_ratings = ratings
        # TODO: Change name to all_features
        self._all_tokens = tokens

        self._start_idx = start_idx
        self._end_idx = end_idx
        
        self._shuffle = shuffle
        self._data_idxs = None
        self._prepare_data()

    def _prepare_data(self):
        self.clear()
        if self._shuffle:
            self._data_idxs = (
                torch.randperm(self._end_idx - self._start_idx) + self._start_idx
            )
        else:
            self._data_idxs = torch.arange(self._start_idx, self._end_idx)

    def clear(self):
        self._data_idxs = None

    def __len__(self):
        return self._end_idx - self._start_idx

    def __getitem__(self, idx):
        if self._data_idxs is None:
            self._prepare_data()
        ret = (self._all_tokens[idx], self._all_ratings[idx])

        if idx == self._end_idx - 1:
            self.clear()

        return ret

class AlteredFeaturesSet(Dataset):
    def __init__(
        self, features, start_idx, end_idx, feature_width=768, num_steps=5, shuffle=True, device=None
    ):
        self.all_features = features

        self._start_idx = start_idx
        self._end_idx = end_idx
        self._shuffle = shuffle
        self._feature_width = feature_width
        self._data_idxs = None
        self._num_steps = num_steps

        if device is None:
            device = torchutils.get_default_device()
        self._device = device
        
        self._last_features=None
        self._prepare_data

    def _prepare_data(self):
        self.clear()
        if self._shuffle:
            self._data_idxs = (
                torch.randperm(self._end_idx - self._start_idx) + self._start_idx
            )
        else:
            self._data_idxs = torch.arange(self._start_idx, self._end_idx)
            

    def clear(self):
        self._data_idxs = None

    def _make_rand_shift(self, scale_div):
        shift = torch.randn((768,))
        shift /= shift.norm(dim=-1, keepdim=True)
        min_scale = 0.6
    
        scale = (torch.rand((1,))+min_scale) / scale_div 
        return (scale * shift).view(1, 768) 
    
    def __len__(self):
        return self._num_steps*(self._end_idx - self._start_idx)

    def __getitem__(self, idx):
        if self._data_idxs is None:
            self._prepare_data()
        
        with torch.no_grad():
            #todo cross product
            features = self._get_features_for_idx(idx)

            
            if idx % 50 == 0:
                scaled_features = features
                sim = torch.ones(1)
            else:
                cycle_idx = idx % self._num_steps
                scale_div = 1.5+2*(cycle_idx+1)/self._num_steps
                scaled_features = features + self._make_rand_shift(scale_div)
                scaled_features = scaled_features / scaled_features.norm(dim=-1, keepdim=True)
                sim = sdutils.cosine_sim(features, scaled_features)
            
        
        ret = (scaled_features.view(-1), sim)
        if idx//self._num_steps == self._end_idx - 1:
            self.clear()

        return ret

    def _get_features_for_idx(self, idx):
        actual_idx = self._data_idxs[idx // self._num_steps]
        features = self.all_features[actual_idx].float()
        return (features / features.norm(dim=-1, keepdim=True)).view(-1)
        
def get_pipeline_data_loader(
    text_tokens,
    vit14_clip_model,
    in_stage,
    out_stage,
    batch_size=100,
    val_split=0.05,
    chunk_size=100 * 20,
    shuffle=True,
):
    tokens = torch.tensor(np.array(text_tokens).astype(np.int32))
    training_idx, val_idx = torchutils.get_split_idxs(tokens.size(0), val_split)

    train_data_set = PipelineDataset(
        tokens,
        vit14_clip_model,
        in_stage,
        out_stage,
        *training_idx,
        chunk_size=chunk_size,
        shuffle=shuffle,
    )
    if val_split != 0:
        val_data_set = PipelineDataset(
            tokens,
            vit14_clip_model,
            in_stage,
            out_stage,
            *val_idx,
            chunk_size=chunk_size,
            shuffle=shuffle,
        )
    else:
        val_data_set = None
    train_data_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=False)
    test_data_loader = DataLoader(val_data_set, batch_size=batch_size, shuffle=False)
    return train_data_loader, test_data_loader

def get_tokens_to_features(
    text_tokens,
    features,
    batch_size=500,
    val_split=0.05,
    num_workers=8,
    shuffle=True,
    pin_memory=True,
):
    """
    if not isinstance(text_tokens, torch.Tensor):
        text_tokens = torch.tensor(np.array(text_tokens).astype(np.int32))
        
    
    if not isinstance(features, torch.Tensor):
        features = torch.tensor(np.array(features).astype(np.float32))
    """     
    torch.int
    training_idx, val_idx = torchutils.get_split_idxs(text_tokens.size(0), val_split)
    
    train_data_set = DirectTextFeaturesSet(
        text_tokens,
        features,
        *training_idx,
        shuffle=shuffle,
    )
    if val_split != 0:
        val_data_set = DirectTextFeaturesSet(
            text_tokens,
            features,
            *val_idx,
            shuffle=shuffle,
        )
    else:
        val_data_set = None
        
    print("Data Loaders")
    train_data_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, num_workers=num_workers)
    test_data_loader = DataLoader(val_data_set, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, num_workers=num_workers)
    return train_data_loader, test_data_loader


def get_feature_to_rating_data_loader(
    features,
    ratings,
    batch_size=500,
    val_split=0.05,
    pin_memory=True,
    num_workers=8,
):
    if not isinstance(features, torch.Tensor):
        features = torch.tensor(np.array(features).astype(np.float32))
    if not isinstance(ratings, torch.Tensor):
        ratings = torch.tensor(np.array(ratings).astype(np.float32))

    training_idx, val_idx = torchutils.get_split_idxs(features.size(0), val_split)

    train_data_set = TextFeaturesToRatingSet(
        features, ratings, *training_idx
    )
    val_data_set = TextFeaturesToRatingSet(
        features, ratings, *val_idx
    )

    train_data_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, num_workers=num_workers)
    test_data_loader = DataLoader(val_data_set, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, num_workers=num_workers)
    return train_data_loader, test_data_loader

def get_altered_feature_data_loader(
    features,
    batch_size=500,
    val_split=0.05,
    num_workers=12,
    val_workers=4,
    pin_memory=True,
):
    if not isinstance(features, torch.Tensor):
        features = torch.tensor(np.array(features).astype(np.float32))
    training_idx, val_idx = torchutils.get_split_idxs(features.size(0), val_split)

    train_data_set = AlteredFeaturesSet(
        features, *training_idx
    )
    val_data_set = AlteredFeaturesSet(
        features, *val_idx
    )

    train_data_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, num_workers=num_workers)
    test_data_loader = DataLoader(val_data_set, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, num_workers=val_workers)
    return train_data_loader, test_data_loader
