import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sktime.datasets import load_from_tsfile_to_dataframe
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import warnings
import random
import psutil
import gc
import hashlib 
from tqdm import tqdm
import h5py
from torch.utils.data import Dataset
from collections import OrderedDict
from collections import Counter
warnings.filterwarnings('ignore')
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from prefetch_generator import BackgroundGenerator
from torch.utils.data import ConcatDataset

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def load_multi_center_dataset(dataset_config, flag, args):
    """센터별 데이터셋 로드 후 ConcatDataset으로 묶음"""
    datasets = []
    for center_name, cfg in dataset_config.items():
        serial_file = cfg.get("serial_file")
        if not serial_file or not os.path.exists(serial_file):
            print(f"[{center_name}] skipped: no serial file")
            continue

        ds = H5Dataset(
            data_path=cfg["data_path"],
            label_path=cfg["label_path"],
            serial_file=serial_file,
            seq_len=args.seq_len,
            flag=flag,
        )
        print(f"{flag} | {center_name}: {len(ds)} samples")
        datasets.append(ds)

    return ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]


class H5Dataset(Dataset):
    """
     HDF5 EEG dataset
    - 모든 샘플은 고정 길이 (C, L)
    - 단일 에폭 단위 입력
    - HDF5: {data: (C, L)}
    """

    def __init__(self, data_path, label_path, serial_file, seq_len=3840, flag='train'):
        super().__init__()
        self.data_path = data_path
        self.seq_len = seq_len
        self.flag = flag
        self.h5_files = {}

        # 라벨 로드
        labels = pd.read_csv(label_path)
        with open(serial_file, 'r') as f:
            keep_pids = [line.strip() for line in f.readlines()]
        labels = labels[labels["Patient_ID"].isin(keep_pids)]

        labels["serial_number"] = labels.apply(
            lambda r: f"{r['Patient_ID']}/{str(r['Epoch']).zfill(4)}", axis=1
        )
        labels["resp_label"] = labels["Respiratory_Event"].apply(lambda x: 1 if x in [1, 2] else 0)
        self.labels_df = labels

        self.windows = self._build_windows()

    def _build_windows(self):
        """환자별 모든 epoch을 윈도우 단위로 정리"""
        windows = []
        for pid, group in tqdm(self.labels_df.groupby("Patient_ID"), desc="Preparing windows"):
            h5_path = os.path.join(self.data_path, f"{pid}.h5")
            if not os.path.exists(h5_path):
                print(f"[WARN] Missing file: {h5_path}")
                continue
            group = group.sort_values("Epoch")
            for _, row in group.iterrows():
                serial = row["serial_number"]
                label_e = int(row["resp_label"])
                label_s = int(row["Sleep_Stage"])
                start = (int(row["Epoch"]) - 1) * self.seq_len
                windows.append({
                    "h5_path": h5_path,
                    "serial": serial,
                    "label_e": label_e,
                    "label_s": label_s,
                    "start": start,
                    "end": start + self.seq_len,
                })
        return windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        w = self.windows[idx]
        h5_path = w["h5_path"]

        if h5_path not in self.h5_files:
            self.h5_files[h5_path] = h5py.File(h5_path, "r", swmr=True, libver="latest", locking=False)
        raw = self.h5_files[h5_path]["data"]

        data = raw[:, w["start"]:w["end"]]
        if data.shape[1] < self.seq_len:
            pad = self.seq_len - data.shape[1]
            data = np.pad(data, ((0, 0), (0, pad)), mode='constant')

        X = torch.from_numpy(data.astype(np.float32))
        y_stage = torch.tensor(w["label_s"], dtype=torch.long)
        y_event = torch.tensor(w["label_e"], dtype=torch.long)

        return X, y_stage, y_event, w["serial"], torch.tensor([True])

