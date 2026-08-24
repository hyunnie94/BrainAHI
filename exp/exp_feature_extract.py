# exp_feature_extract.py
import os, h5py, torch, numpy as np
from tqdm import tqdm
from utils.data_utils import ( extract_center_from_serial, AHI_TST_Loader, collect_all_windows, load_all_nsrr_centers, merge_serial_with_meta )

class FeatureExtractor:
    def __init__(self, model, args, device):
        self.model = model.to(device)
        self.args = args
        self.device = device

    def extract(self, data_loader, data_set, save_path):
        """Extract patient-level event features and save as HDF5"""
        # ------------------------------------------------------------
        # 1. Meta info (AHI, TST) load
        # ------------------------------------------------------------
        meta_dir = "/hk_vol/data_OL1/hklee/0_Data/"
        ahi_loader = AHI_TST_Loader(meta_dir)
        data_windows = collect_all_windows(data_set)

        centers_in_data = set()
        for w in data_windows:
            s = w.get('serial_numbers', [w.get('serial_number')])[0]
            try:
                c = extract_center_from_serial(str(s).split('/')[0])
                centers_in_data.add(c)
            except Exception:
                pass

        centers_to_load = sorted(centers_in_data)
        print(f"[META] Detected centers in dataset → {centers_to_load}")
        load_all_nsrr_centers(ahi_loader, centers_to_load)
        ahi_dict, tst_dict = merge_serial_with_meta(data_windows, ahi_loader)
        print(f"[META] Loaded meta: AHI={len(ahi_dict)}, TST={len(tst_dict)}")

        # ------------------------------------------------------------
        # 2. Initialize output file
        # ------------------------------------------------------------
        if os.path.exists(save_path):
            os.remove(save_path)
        self.model.eval()
        patient_data = {}

        # ------------------------------------------------------------
        # 3. Extract loop
        # ------------------------------------------------------------
        with torch.no_grad():
            for i, (batch_x, _, _, serial, mask) in enumerate(tqdm(data_loader, desc="Extracting features")):
                batch_x, mask = batch_x.to(self.device), mask.to(self.device)

                outs = self.model(batch_x, mask=mask, return_features=True)
                features = outs["event_features"].detach().cpu().numpy()  # (B, N, D)
                B, T, D = features.shape
                assert T == 30, f"Unexpected time dimension: {T}, expected 30"

                for j, s in enumerate(serial):
                    pid, epoch = s.split('/')
                    epoch = int(epoch)
                    if pid not in patient_data:
                        patient_data[pid] = {
                            "features": [],
                            "epoch_nums": [],
                            "ahi": ahi_dict.get(pid, np.nan),
                            "tst": tst_dict.get(pid, np.nan),
                        }
                    patient_data[pid]["features"].append(features[j])
                    patient_data[pid]["epoch_nums"].append(epoch)

                # ------------------------------------------------------------
                # 4. Periodic flush to HDF5
                # ------------------------------------------------------------
                if (i % 5000 == 0) or (i == len(data_loader) - 1):
                    with h5py.File(save_path, "a") as f:
                        for pid, d in patient_data.items():
                            if pid not in f:
                                grp = f.create_group(pid)
                                grp.attrs["Actual_AHI"] = d["ahi"]
                                grp.attrs["TST_hours"] = d["tst"]
                                grp.create_dataset(
                                    "features",
                                    shape=(0, 30, D),
                                    maxshape=(None, 30, D),
                                    dtype="float32",
                                )
                                grp.create_dataset(
                                    "epoch_nums",
                                    shape=(0,),
                                    maxshape=(None,),
                                    dtype="int32",
                                )

                            grp = f[pid]
                            grp.attrs["Actual_AHI"] = float(ahi_dict.get(pid, np.nan))
                            grp.attrs["TST_hours"] = float(tst_dict.get(pid, np.nan))

                            fe_ds = grp["features"]
                            ep_ds = grp["epoch_nums"]
                            cur = fe_ds.shape[0]
                            new = cur + len(d["features"])
                            fe_ds.resize((new, 30, D))
                            fe_ds[cur:] = np.stack(d["features"])
                            ep_ds.resize((new,))
                            ep_ds[cur:] = d["epochs"]
                        patient_data.clear()
                    tqdm.write(f"[Batch {i}] Saved → {save_path}")

                    # GPU / CPU 메모리 정리
                    del outs, features
                    torch.cuda.empty_cache()

        print(f"✅ Feature extraction completed → {save_path}")