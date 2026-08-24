# exp_multitask_test.py
# -------------------------------------------------------
# Unified Multi-center testing pipeline for MultiTask2 model
# - Runs both internal and external evaluation automatically
# - Saves confusion matrix, classification reports, and per-epoch predictions
# - Optionally saves per-patient features & attention weights
# -------------------------------------------------------

import os, torch, h5py, numpy as np, pandas as pd
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

from utils.data_utils import (
    AHI_TST_Loader, extract_center_from_serial, collect_all_windows,
    load_all_nsrr_centers, merge_serial_with_meta
)
from utils.tools import cal_accuracy, cal_macro_f1, calculate_auroc
from data_provider.data_factory import data_provider


class Exp_MultiTask_Test:
    def __init__(self, model, args, device):
        """
        Args:
            model (nn.Module): trained MultiTask2 model
            args: argparse-style config
            device: torch.device
        """
        self.model = model.to(device)
        self.args = args
        self.device = device

    # ============================================================
    # Run all centers (internal + external)
    # ============================================================
    def run_all(self, save_features=True):
        print("\n==============================")
        print("🏠 Internal Centers Evaluation")
        print("==============================")
        # Internal 센터 전체 한꺼번에 테스트
        internal_dataset, internal_loader = data_provider(self.args, flag='test', center=self.args.internal_centers)
        internal_save_dir = os.path.join(self.args.test_results, self.args.model_id, "internal", "combined")
        self.run(internal_dataset, internal_loader, "internal_combined", internal_save_dir, save_features=save_features)

        print("\n==============================")
        print("🌍 External Centers Evaluation")
        print("==============================")
        # External 센터 개별 테스트
        for center in self.args.external_centers:
            dataset, loader = data_provider(self.args, flag='test', center=center)
            save_dir = os.path.join(self.args.test_results, self.args.model_id, "external", center)
            self.run(dataset, loader, center, save_dir, save_features=save_features)

    # ============================================================
    # Run test for a single center
    # ============================================================
    def run(self, test_data, test_loader, center, save_dir, save_features=True):
        os.makedirs(save_dir, exist_ok=True)
        print(f"\n=== Evaluating {center} ===")

        # ---------------------------------------------------------------------
        # Step 1. Build AHI/TST meta info
        # ---------------------------------------------------------------------
        meta_dir = "/home/data_OL1/hklee/0_Data/"
        ahi_loader = AHI_TST_Loader(meta_dir)
        data_windows = collect_all_windows(test_data)
        centers_in_test = set()

        for w in data_windows:
            s = w.get("serial_numbers", [w.get("serial_number")])[0]
            try:
                c = extract_center_from_serial(str(s).split("/")[0])
                centers_in_test.add(c)
            except Exception:
                pass

        centers_to_load = sorted(centers_in_test)
        load_all_nsrr_centers(ahi_loader, centers_to_load)
        ahi_dict, tst_dict = merge_serial_with_meta(data_windows, ahi_loader)
        print(f"[META] Loaded AHI={len(ahi_dict)}, TST={len(tst_dict)}, Centers={centers_to_load}")

        # ---------------------------------------------------------------------
        # Step 2. Evaluation setup
        # ---------------------------------------------------------------------
        self.model.eval()
        preds_stage, preds_event = [], []
        trues_stage, trues_event = [], []
        stage_probs_list, event_probs_list = [], []
        serials = []

        # Optional HDF5 storage
        if save_features:
            h5_path = os.path.join(save_dir, "patient_features_test.h5")
            attn_h5_path = os.path.join(save_dir, "attention_weights_test.h5")
            if os.path.exists(h5_path): os.remove(h5_path)
            if os.path.exists(attn_h5_path): os.remove(attn_h5_path)
            patient_data = {}

        # ---------------------------------------------------------------------
        # Step 3. Inference loop
        # ---------------------------------------------------------------------
        with torch.no_grad():
            for i, (batch_x, sleep_stages, labels, serial, mask) in enumerate(
                tqdm(test_loader, desc=f"Testing {center}")
            ):
                batch_x = batch_x.float().to(self.device)
                sleep_stages = sleep_stages.to(self.device)
                labels = labels.long().to(self.device)
                mask = mask.to(self.device)

                outs = self.model(batch_x, mask=mask, use_1sec_labels=False, return_features=True)

                # --- Stage task ---
                if outs.get("stage_logits") is not None:
                    stage_logits = outs["stage_logits"]
                    stage_probs = torch.softmax(stage_logits, dim=-1)
                    stage_preds = torch.argmax(stage_probs, dim=-1)
                    preds_stage.append(stage_preds.cpu().reshape(-1))
                    trues_stage.append(sleep_stages.cpu().reshape(-1))
                    Ks = stage_probs.shape[-1]
                    stage_probs_list.extend(stage_probs.cpu().numpy().reshape(-1, Ks).tolist())

                # --- Event task ---
                if outs.get("event_logits") is not None:
                    event_logits = outs["event_logits"]
                    event_probs = torch.softmax(event_logits, dim=-1)
                    event_preds = torch.argmax(event_probs, dim=-1)
                    preds_event.append(event_preds.cpu().reshape(-1))
                    trues_event.append(labels.cpu().reshape(-1))
                    Ke = event_probs.shape[-1]
                    event_probs_list.extend(event_probs.cpu().numpy().reshape(-1, Ke).tolist())

                # --- Serial alignment ---
                B = batch_x.size(0)
                if outs.get("event_logits") is not None:
                    E = outs["event_logits"].shape[1]
                elif outs.get("stage_logits") is not None:
                    E = outs["stage_logits"].shape[1]
                else:
                    E = 1

                if isinstance(serial, (list, tuple)):
                    ser_list = serial
                else:
                    ser_list = [serial] * B

                for s in ser_list:
                    pid = str(s).split("/")[0]
                    for e in range(E):
                        serials.append(f"{pid}/e{e}")

                # --- Save features ---
                if save_features and outs.get("event_features") is not None:
                    features = outs["event_features"].cpu().numpy()
                    Bf, T, H = features.shape
                    for j in range(Bf):
                        pid = ser_list[j].split("/")[0]
                        ep = int(ser_list[j].split("/")[1]) if "/" in ser_list[j] else -1
                        if pid not in patient_data:
                            patient_data[pid] = {"features": [], "epoch_nums": []}
                        patient_data[pid]["features"].append(features[j])
                        patient_data[pid]["epoch_nums"].append(ep)

                    if (i % 1000 == 0) or (i == len(test_loader) - 1):
                        with h5py.File(h5_path, "a") as f:
                            for pid, d in patient_data.items():
                                grp = f.require_group(pid)
                                if "features" not in grp:
                                    grp.create_dataset(
                                        "features", shape=(0, T, H),
                                        maxshape=(None, T, H), dtype="float32"
                                    )
                                    grp.create_dataset("epoch_nums", shape=(0,), maxshape=(None,), dtype="int32")
                                fe, ep = grp["features"], grp["epoch_nums"]
                                cur = fe.shape[0]; new = cur + len(d["features"])
                                fe.resize((new, T, H)); fe[cur:] = np.stack(d["features"])
                                ep.resize((new,)); ep[cur:] = d["epoch_nums"]
                            patient_data.clear()
                        tqdm.write(f"[Batch {i}] Features saved → {h5_path}")

        # ---------------------------------------------------------------------
        # Step 4. Metrics & reports
        # ---------------------------------------------------------------------
        if preds_stage:
            preds_stage_np = torch.cat(preds_stage).numpy()
            trues_stage_np = torch.cat(trues_stage).numpy()
            acc_stage = cal_accuracy(preds_stage_np, trues_stage_np)
            f1_stage = cal_macro_f1(trues_stage_np, preds_stage_np)
            auroc_stage = calculate_auroc(trues_stage_np, np.array(stage_probs_list), self.args.num_stage_classes)
            pd.DataFrame(confusion_matrix(trues_stage_np, preds_stage_np)).to_csv(
                f"{save_dir}/stage_confusion_matrix.csv", index=False)
            pd.DataFrame(classification_report(trues_stage_np, preds_stage_np, output_dict=True)).T.to_csv(
                f"{save_dir}/stage_classification_report.csv")
            with open(f"{save_dir}/stage_auroc.txt", "w") as f:
                f.write(f"{auroc_stage:.4f}\n")

        if preds_event:
            preds_event_np = torch.cat(preds_event).numpy()
            trues_event_np = torch.cat(trues_event).numpy()
            acc_event = cal_accuracy(preds_event_np, trues_event_np)
            f1_event = cal_macro_f1(trues_event_np, preds_event_np)
            auroc_event = calculate_auroc(trues_event_np, np.array(event_probs_list), self.args.num_event_classes)
            pd.DataFrame(confusion_matrix(trues_event_np, preds_event_np)).to_csv(
                f"{save_dir}/event_confusion_matrix.csv", index=False)
            pd.DataFrame(classification_report(trues_event_np, preds_event_np, output_dict=True)).T.to_csv(
                f"{save_dir}/event_classification_report.csv")
            with open(f"{save_dir}/event_auroc.txt", "w") as f:
                f.write(f"{auroc_event:.4f}\n")

        # ---------------------------------------------------------------------
        # Step 5. Save detailed probability table
        # ---------------------------------------------------------------------
        n = max(len(serials), len(stage_probs_list), len(event_probs_list))
        def _pad(lst, target, fill):
            while len(lst) < target:
                lst.append(fill)
        _pad(serials, n, "NA")
        _pad(stage_probs_list, n, [np.nan] * self.args.num_stage_classes)
        _pad(event_probs_list, n, [np.nan] * self.args.num_event_classes)

        df = pd.DataFrame({"Serial": serials})
        for k in range(self.args.num_stage_classes):
            df[f"Stage_P{k}"] = np.array(stage_probs_list)[:, k]
        for k in range(self.args.num_event_classes):
            df[f"Event_P{k}"] = np.array(event_probs_list)[:, k]
        df.to_csv(f"{save_dir}/detailed_results.csv", index=False)

        print(f"✅ Saved detailed results → {save_dir}")
        print(f"    Stage Acc={acc_stage:.3f}, F1={f1_stage:.3f}")
        print(f"    Event Acc={acc_event:.3f}, F1={f1_event:.3f}")