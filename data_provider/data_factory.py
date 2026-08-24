from data_provider.data_loader import H5Dataset, load_multi_center_dataset, DataLoaderX
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.data import DataLoader
from collections import OrderedDict
import torch
from torch.utils.data import Sampler
import numpy as np
import random

class BalancedSampler(Sampler):
    def __init__(self, dataset, num_samples_per_epoch=None):
        """균형 샘플링용 Sampler (단일 에폭 버전)"""
        if hasattr(dataset, 'datasets'):
            all_windows = []
            for ds in dataset.datasets:
                all_windows.extend(ds.windows)
        else:
            all_windows = dataset.windows

        self.labels = np.array([win['label_e'] for win in all_windows])
        self.indices = np.arange(len(self.labels))
        self.class_indices = {cls: np.where(self.labels == cls)[0] for cls in np.unique(self.labels)}
        self.num_classes = len(self.class_indices)
        self.num_samples_per_epoch = num_samples_per_epoch or len(self.labels)

    def __iter__(self):
        indices = []
        samples_per_class = self.num_samples_per_epoch // self.num_classes
        for cls, idxs in self.class_indices.items():
            sampled = np.random.choice(idxs, size=samples_per_class, replace=True)
            indices.extend(sampled.tolist())
        random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples_per_epoch



from torch.utils.data import DataLoader, WeightedRandomSampler, ConcatDataset

def data_provider(args, flag, center=None):
    timeenc = 0 if args.embed != 'timeF' else 1

    def worker_init_fn(worker_id):
        worker_info = torch.utils.data.get_worker_info()
        dataset = worker_info.dataset
        if hasattr(dataset, 'datasets'):  # ConcatDataset
            for ds in dataset.datasets:
                ds.h5_files = {}
        else:
            dataset.h5_files = {}

    # Flag별 설정
    if flag == 'train':
        dataset_config = args.train_centers  
        batch_size = args.batch_size
        drop_last = True
        shuffle_flag = True
    elif flag == 'val':
        dataset_config = args.val_centers
        batch_size = args.batch_size
        drop_last = True
        shuffle_flag = False
    elif flag == 'test':
        all_test_centers = args.test_centers
        if center is None:
            dataset_config = all_test_centers
        elif isinstance(center, (list, tuple)):
            dataset_config = {c: all_test_centers[c] for c in center}
        else:
            if center not in all_test_centers:
                raise ValueError(f"[ERROR] Center '{center}' not found in test_centers config.")
            dataset_config = {center: all_test_centers[center]}

        batch_size = 1 if args.task_name not in ['classification'] else args.batch_size
        drop_last = False
        shuffle_flag = False
    else:
        raise ValueError("Invalid flag")


    dataset = load_multi_center_dataset(dataset_config, flag, args)
    # Train mode: Balanced Sampler 적용
    sampler = BalancedSampler(dataset, num_samples_per_epoch=len(dataset)) if flag == 'train' else None


    
    data_loader = DataLoaderX(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle_flag and sampler is None,
        num_workers=args.num_workers,
        drop_last=drop_last,
        pin_memory=True,
        persistent_workers=False,
        worker_init_fn=worker_init_fn,
        prefetch_factor=4,
    )

    return dataset, data_loader

    