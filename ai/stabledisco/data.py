from enum import Enum

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


class PipelineChunk(torchdata.CudaChunk):
    def __init__(self, tokens, clip_model, in_stage, out_stage, chunk_size=64 * 5):
        super().__init__(tokens, chunk_size)
        self._clip_model = clip_model
        self._in_stage = in_stage
        self._out_stage = out_stage

    def _get_chunk_for_range(self, start_idx, end_idx):
        # Return true if done
        in_vals = None
        out_vals = None

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


class ImgToTextFeaturesSet(Dataset):
    def __init__(self, features, tokens, clip_model, start_idx, end_idx, shuffle=True):
        self._all_features = features.float()
        self._all_tokens = tokens.long()
        self._clip_model = clip_model

        self._start_idx = start_idx
        self._end_idx = end_idx
        self._shuffle = shuffle

        self._curr_text_features = None
        self._curr_image_features = None

        self._prepare_data()

    def _prepare_data(self):
        if self._shuffle:
            data_idxs = (
                torch.randperm(self._end_idx - self._start_idx) + self._start_idx
            )
        else:
            data_idxs = torch.arange(self._start_idx, self._end_idx)

        if self._curr_text_features:
            del self._curr_text_features
        if self._curr_image_features:
            del self._curr_image_features

        self._curr_text_features = PipelineChunk(
            self._all_tokens[data_idxs],
            self._clip_model,
            PipelineStage.TOKEN_LATENT,
            PipelineStage.TOKEN_LATENT,
            chunk_size=128 * 20,
        )
        self._curr_image_features = torchdata.CudaChunk(
            self._all_features[data_idxs], chunk_size=128 * 20
        )

    def __len__(self):
        return self._end_idx - self._start_idx

    def clear(self):
        self._curr_image_features = None
        self._curr_text_features = None

    def __getitem__(self, idx):
        if not self._curr_image_features:
            self._prepare_data()
        ret = (self._curr_image_features[idx], self._curr_text_features[idx])
        if idx == self._end_idx - 1:
            self.clear()

        return ret


class ImgFeaturesToTokensSet(Dataset):
    def __init__(self, features, tokens, clip_model, start_idx, end_idx, shuffle=True):
        self._all_features = features.half()
        self._all_tokens = tokens.long()
        self._clip_model = clip_model

        self._start_idx = start_idx
        self._end_idx = end_idx
        self._shuffle = shuffle

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
        ret = (self._curr_image_features[idx], self._curr_text_tokens[idx])
        if idx == self._end_idx - 1:
            self.clear()

        return ret


class TextFeaturesToRatingSet(Dataset):
    def __init__(
        self, clip_model, tokens, ratings, start_idx, end_idx, feature_chunk_size, rating_chunk_size, shuffle=True
    ):
        self._all_ratings = ratings
        self._all_tokens = tokens

        self._clip_model = clip_model

        self._start_idx = start_idx
        self._end_idx = end_idx
        self._feature_chunk_size = feature_chunk_size
        self._rating_chunk_size = rating_chunk_size
        self._shuffle = shuffle

        self._curr_ratings = None
        self._curr_text_features = None

    def _prepare_data(self):
        if self._shuffle:
            data_idxs = (
                torch.randperm(self._end_idx - self._start_idx) + self._start_idx
            )
        else:
            data_idxs = torch.arange(self._start_idx, self._end_idx)

        self.clear()

        self._curr_ratings = torchdata.CudaChunk(
            self._all_ratings[data_idxs], chunk_size=self._rating_chunk_size
        )
        self._curr_text_features = PipelineChunk(
            self._all_tokens[data_idxs],
            self._clip_model,
            PipelineStage.TOKEN_LATENT,
            PipelineStage.TOKEN_LATENT,
            chunk_size=self._feature_chunk_size,
        )

    def clear(self):
        self._curr_text_features = None
        self._curr_ratings = None

    def __len__(self):
        return self._end_idx - self._start_idx

    def __getitem__(self, idx):
        if self._curr_text_features is None:
            self._prepare_data()
        ret = (self._curr_text_features[idx], self._curr_ratings[idx])
        if idx == self._end_idx - 1:
            self.clear()

        return ret

class AlteredFeaturesSet(Dataset):
    def __init__(
        self, clip_model, tokens, start_idx, end_idx, feature_chunk_size, feature_width=768, shuffle=True, device=None
    ):
        self._all_tokens = tokens

        self._clip_model = clip_model

        self._start_idx = start_idx
        self._end_idx = end_idx
        self._feature_chunk_size = feature_chunk_size
        self._shuffle = shuffle
        self._feature_width = feature_width
        
        self._curr_ratings = None
        self._curr_text_features = None
        

        if device is None:
            device = torchutils.get_default_device()
        self._device = device
        self._is_text = torch.tensor([1.0, 0.0], device=self._device)
        self._last_features=None

    def _prepare_data(self):
        if self._shuffle:
            data_idxs = (
                torch.randperm(self._end_idx - self._start_idx) + self._start_idx
            )
        else:
            data_idxs = torch.arange(self._start_idx, self._end_idx)

        self.clear()

        self._curr_text_features = PipelineChunk(
            self._all_tokens[data_idxs],
            self._clip_model,
            PipelineStage.TOKEN_LATENT,
            PipelineStage.TOKEN_LATENT,
            chunk_size=self._feature_chunk_size,
        )

    def clear(self):
        self._curr_text_features = None
        self._curr_ratings = None

    def _make_rand_shift(self):
        shift = torch.randn((self._feature_width), device=self._device)
        shift /= shift.norm(dim=-1, keepdim=True)
        scale = self._random_scale()
        return scale * shift

    def _random_scale(self, mean=0.0, std=1.0, min_scale=0.002):
        return torch.abs(torch.rand(1, device=self._device) * std + mean) + min_scale

    def __len__(self):
        return (self._end_idx - self._start_idx)*2

    def __getitem__(self, idx):
        if self._curr_text_features is None:
            self._prepare_data()
        
        with torch.no_grad():
            if idx % 2 == 1:
                features = self._last_features
                if features == None:
                    features = self._get_features_for_idx(idx)

                rand_shift = self._make_rand_shift()
                
                features = features + rand_shift
                features /= features.norm(dim=-1, keepdim=True)
            else:
                features = self._get_features_for_idx(idx)
                self._last_features = features
        
        ret = (features, self._is_text[idx%2])
        if idx == self._end_idx - 1:
            self.clear()

        return ret

    def _get_features_for_idx(self, idx):
        actual_idx = idx // 2
        features = self._curr_text_features[actual_idx].float()
        return features / features.norm(dim=-1, keepdim=True)
        



def get_pipeline_data_loader(
    prompt_dataframe,
    vit14_clip_model,
    in_stage,
    out_stage,
    batch_size=100,
    val_split=0.05,
    chunk_size=100 * 20,
):
    tokens = tuple(
        (torch.tensor(x.astype(np.int64)) for x in prompt_dataframe["text_tokens"])
    )
    tokens = torch.stack(tokens)
    total_data = len(tokens)
    training_idx, val_idx = torchutils.get_split_idxs(total_data, val_split)

    train_data_set = PipelineDataset(
        tokens,
        vit14_clip_model,
        in_stage,
        out_stage,
        *training_idx,
        chunk_size=chunk_size,
    )
    val_data_set = PipelineDataset(
        tokens,
        vit14_clip_model,
        in_stage,
        out_stage,
        *val_idx,
        chunk_size=chunk_size
    )
    train_data_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=False)
    test_data_loader = DataLoader(val_data_set, batch_size=batch_size, shuffle=False)
    return train_data_loader, test_data_loader


def get_feature_to_rating_data_loader(
    prompt_dataframe,
    vit14_clip_model,
    batch_size=400,
    val_split=0.05,
    feature_chunk_size=1024 * 15,
    rating_chunk_size=1024 * 50,
):
    tokens = tuple((torch.tensor(x) for x in prompt_dataframe["text_tokens"]))
    tokens = torch.stack(tokens)
    total_data = len(tokens)
    training_idx, val_idx = torchutils.get_split_idxs(total_data, val_split)

    ratings = tuple((torch.tensor(x) for x in prompt_dataframe["aesthetic"]))
    ratings = torch.stack(ratings)

    train_data_set = TextFeaturesToRatingSet(
        vit14_clip_model, tokens, ratings, *training_idx, feature_chunk_size=feature_chunk_size, rating_chunk_size=rating_chunk_size
    )
    val_data_set = TextFeaturesToRatingSet(
        vit14_clip_model, tokens, ratings, *val_idx, feature_chunk_size=feature_chunk_size, rating_chunk_size=rating_chunk_size
    )

    train_data_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=False)
    test_data_loader = DataLoader(val_data_set, batch_size=batch_size, shuffle=False)
    return train_data_loader, test_data_loader

def get_altered_feature_data_loader(
    prompt_dataframe,
    vit14_clip_model,
    batch_size=350,
    val_split=0.05,
    feature_chunk_size=1024 * 15,
):
    tokens = tuple((torch.tensor(x) for x in prompt_dataframe["text_tokens"]))
    tokens = torch.stack(tokens)
    total_data = len(tokens)
    training_idx, val_idx = torchutils.get_split_idxs(total_data, val_split)

    train_data_set = AlteredFeaturesSet(
        vit14_clip_model, tokens, *training_idx, feature_chunk_size=feature_chunk_size
    )
    val_data_set = AlteredFeaturesSet(
        vit14_clip_model, tokens, *val_idx, feature_chunk_size=feature_chunk_size
    )

    train_data_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=False)
    test_data_loader = DataLoader(val_data_set, batch_size=batch_size, shuffle=False)
    return train_data_loader, test_data_loader


def get_feature_to_tokens_data_loader(
    prompt_dataframe, vit14_clip_model, batch_size=150, val_split=0.1
):
    training_rows = prompt_dataframe[prompt_dataframe["img_features"].notna()]
    tokens = tuple((torch.tensor(x) for x in training_rows["text_tokens"]))
    features = tuple((torch.tensor(x) for x in training_rows["img_features"]))

    tokens = torch.stack(tokens)
    features = torch.stack(features)

    total_data = len(tokens)
    training_idx, val_idx = torchutils.get_split_idxs(total_data, val_split)

    train_data_set = ImgFeaturesToTokensSet(
        features, tokens, vit14_clip_model, *training_idx
    )
    val_data_set = ImgFeaturesToTokensSet(features, tokens, vit14_clip_model, *val_idx)

    train_data_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=False)
    test_data_loader = DataLoader(val_data_set, batch_size=batch_size, shuffle=False)
    return train_data_loader, test_data_loader


def get_direct_feature_to_tokens_data_loader(
    prompt_dataframe, vit14_clip_model, batch_size=128, val_split=0.1
):
    training_rows = prompt_dataframe[prompt_dataframe["img_features"].notna()]
    tokens = tuple((torch.tensor(x) for x in training_rows["text_tokens"]))
    features = tuple((torch.tensor(x) for x in training_rows["img_features"]))

    tokens = torch.stack(tokens)
    features = torch.stack(features)

    total_data = len(tokens)
    training_idx, val_idx = torchutils.get_split_idxs(total_data, val_split)

    train_data_set = ImgFeaturesToTokensSet(
        features, tokens, vit14_clip_model, *training_idx
    )
    val_data_set = ImgFeaturesToTokensSet(features, tokens, vit14_clip_model, *val_idx)

    train_data_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=False)
    test_data_loader = DataLoader(val_data_set, batch_size=batch_size, shuffle=False)
    return train_data_loader, test_data_loader
