import json
import os
import re
import typing

import numpy as np
import torch
from ai.torchmodules.utils import get_default_device
from torch.utils.data import Dataset

from utils import get_default_path, sort_num_asc


class CudaChunk:
    def __init__(self, data, chunk_size=64 * 10):
        self._data = data
        self._chunk_size = chunk_size
        self._curr_chunk = -1
        self._chunk = None

    def _ensure_chunk_loaded(self, idx):
        target_chunk = self._get_chunk_for_idx(idx)
        if self._curr_chunk == target_chunk:
            return
        self._curr_chunk = target_chunk
        self._load_chunk(self._curr_chunk)

    def _load_chunk(self, chunk_idx):
        if "chunk" in dir(self) and self._chunk is not None:
            del self._chunk

        start_idx = chunk_idx * self._chunk_size
        with torch.no_grad():
            self._chunk = self._get_chunk_for_range(start_idx, start_idx + self._chunk_size)

    def _get_chunk_for_range(self, start_idx, end_idx):
        return self._data[start_idx : end_idx + self._chunk_size].cuda()

    def _get_chunk_for_idx(self, idx):
        return idx // self._chunk_size

    def _get_chunk_offset(self, idx):
        return idx % self._chunk_size

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        self._ensure_chunk_loaded(idx)
        offset_idx = self._get_chunk_offset(idx)
        return self._chunk[offset_idx]


class ShuffledData(Dataset):
    def __init__(self, data, seed=None) -> None:
        self._data = data
        self._shuffle_idx = np.arange(len(data))
        if seed:
            pre_state = np.random.get_state()
            np.random.seed(seed)
        np.random.shuffle(self._shuffle_idx)
        if seed:
            np.random.set_state(pre_state)

    def get_backing(self):
        return self._data

    def __len__(self):
        return len(self._data)

    def __getitem__(self, key):
        if isinstance(key, slice):
            start = key.start if key.start else 0
            stop = key.stop if key.stop else len(self)
            step = key.step if key.step else 1
            return [self[idx] for idx in range(start, stop, step)]

        return self._data[self._shuffle_idx[key]]


class MergedDataset(Dataset):
    def __init__(self, training_files_x, training_files_y, start_idx, end_idx):
        self._training_set_x = training_files_x
        self._training_set_y = training_files_y

        if self._training_set_x.get_total_data_num() != self._training_set_y.get_total_data_num():
            raise Exception("Can't merge different length training sets")

        if self._training_set_x.get_data_per_file() != self._training_set_y.get_data_per_file():
            raise Exception("Can't merge training sets with different data per file")

        self._start_idx = start_idx
        self._end_idx = end_idx
        self._file_data_x = None
        self._file_data_y = None
        self._curr_file_idx = -1

    def _ensure_loaded_for(self, idx):
        target_file = self._training_set_x.file_num_for_data(idx)
        if target_file == self._curr_file_idx:
            return

        self._curr_file_idx = target_file
        self._file_data_x = self._training_set_x.load_file_num(self._curr_file_idx)
        self._file_data_y = self._training_set_y.load_file_num(self._curr_file_idx)

    def _get_file_offset(self, idx):
        return idx % self._training_set_x.get_data_per_file()

    def _get_real_idx(self, idx):
        return self._start_idx + idx

    def __len__(self):
        return self._end_idx - self._start_idx

    def __getitem__(self, idx):
        real_idx = self._get_real_idx(idx)
        self._ensure_loaded_for(real_idx)
        offset_idx = self._get_file_offset(real_idx)

        return self._file_data_x[offset_idx], self._file_data_y[offset_idx]


class SplitTrainingDataSet(Dataset):
    def __init__(self, training_file, start_idx, end_idx, device=None, shuffle=False):
        if device is None:
            device = get_default_device()
        self._device = device

        self._training_file = training_file
        self._start_idx = start_idx
        self._end_idx = end_idx

        self._shuffle = shuffle
        self._curr_file_idx = -1
        self._data = None

    def _ensure_loaded_for(self, idx):
        target_file = self._training_file.file_num_for_data(idx)
        if target_file == self._curr_file_idx:
            return

        self._curr_file_idx = target_file
        file_data = self._training_file.load_file_num(self._curr_file_idx)
        if self._shuffle:
            file_data = ShuffledData(file_data, seed=self._seed)
            if self._seed:
                self._seed += 1

        self._data = file_data

    def _get_file_offset(self, idx):
        return idx % self._training_file.get_data_per_file()

    def _get_real_idx(self, idx):
        return self._start_idx + idx

    def __len__(self):
        return self._end_idx - self._start_idx

    def __getitem__(self, idx):
        real_idx = self._get_real_idx(idx)
        self._ensure_loaded_for(real_idx)
        offset_idx = self._get_file_offset(real_idx)

        return self._data[offset_idx]


class SplitTrainingFiles:
    def __init__(self, training_dir, filename_start, data_per_file):
        self._training_dir = training_dir
        self._filename_start = filename_start
        self._data_per_file = data_per_file
        self._training_data_parts = []
        self.refresh()

    @classmethod
    def from_meta_file(cls, training_dir, filename_start):
        self = cls.__new__(cls)
        path = get_default_path(training_dir, f"{filename_start}_meta.json")

        with open(path, "r") as infile:
            meta = json.load(infile)
        self.__init__(meta["dir"], meta["name"], meta["per_file"])
        return self

    def save_meta_file(self):
        path = get_default_path(self._training_dir, f"{self._filename_start}_meta.json")
        metadata = {
            "dir": self._training_dir,
            "name": self._filename_start,
            "per_file": self._data_per_file,
        }
        with open(path, "w+") as outfile:
            json.dump(metadata, outfile)

    def refresh(self):
        self._training_data_parts = get_training_data_parts(self._training_dir, self._filename_start)

    def save_data_at_file_num(self, data, file_idx):
        path = get_default_path(self._training_dir, f"{self._filename_start}_{file_idx}.pk")
        if type(data) is list:
            data = torch.stack(tuple(data))
        torch.save(data.cpu(), path)
        self.refresh()

    def load_all(self):
        all_data = []

        for filename in self._training_data_parts:
            all_data += torch.load(filename)

        return torch.stack(tuple(all_data))

    def load_file_num(self, file_idx):
        file_data = torch.load(self._training_data_parts[file_idx])
        return torch.stack(tuple(file_data))

    def file_num_for_data(self, data_idx):
        if data_idx > self.get_total_data_num():
            raise Exception(f"Dataset contains fewer than {data_idx} elements")
        return data_idx // self._data_per_file

    def get_total_data_num(self):
        return len(self._training_data_parts) * self._data_per_file

    def get_data_per_file(self):
        return self._data_per_file

    def get_num_files(self):
        return len(self._training_data_parts)


def get_training_data_parts(training_dir, filename_start) -> typing.List[str]:
    training_data_dir = get_default_path(training_dir)
    token_files = [
        os.path.join(training_data_dir, filename)
        for filename in os.listdir(training_data_dir)
        if re.findall(f"{filename_start}.*[0-9]+.pk", filename)
    ]
    sort_num_asc(token_files)
    return token_files


def load_training_data_parts(training_dir, filename_start) -> torch.Tensor:
    all_data = []
    data_files = get_training_data_parts(training_dir, filename_start)
    for filename in data_files:
        all_data += torch.load(filename)

    return torch.stack(tuple(all_data))
