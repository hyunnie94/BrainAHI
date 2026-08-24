# exp_multitask_train.py
import os, time, torch, numpy as np
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import wandb

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy, cal_macro_f1
from utils.losses import FocalLoss, compute_losses
from models.MultiTask import MultiTaskModel

class Exp_Classification_Train(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)

    def _build_model(self):
        """ Build MultiTask2 model with ablation toggles """
        model = MultiTaskModel(
            backbone_cfg={
                "task_name": self.args.task_name,
                "patch_size": self.args.patch_size,
                "patch_stride": self.args.patch_stride,
                "stem_ratio": self.args.stem_ratio,
                "downsample_ratio": self.args.downsample_ratio,
                "ffn_ratio": self.args.ffn_ratio,
                "num_blocks": self.args.num_blocks,
                "large_size": self.args.large_size,
                "small_size": self.args.small_size,
                "dims": self.args.dims,
                "dw_dims": self.args.dw_dims,
                "nvars": self.args.enc_in,
            },
            num_stage_classes=self.args.num_stage_classes,
            num_event_classes=self.args.num_event_classes,
            event_window='30s',
            use_attention=self.args.use_attention,
            dropout=self.args.dropout,
            num_epochs_per_window=self.args.num_epochs_per_window,
            task_mode=self.args.task_mode,
            use_stage_context=self.args.use_stage_context,
        ).float()

        main_gpu = self.args.device_ids[0] if getattr(self.args, "use_multi_gpu", False) else self.args.gpu
        device = torch.device(f"cuda:{main_gpu}" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        if getattr(self.args, 'use_multi_gpu', False):
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        self.device = device
        return model

    def _get_data(self, flag, center=None):
        return data_provider(self.args, flag, center=center)

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)


    def vali(self, vali_data, vali_loader, w_stage=1.0, w_event=1.0):
        """Validation loop"""
        self.model.eval()
        total_losses, stage_losses, event_losses = [], [], []
        preds_stage, preds_event = [], []
        trues_stage, trues_event = [], []

        with torch.no_grad():
            for (batch_x, sleep_stages, labels, _, mask) in tqdm(vali_loader, desc="Validating"):
                batch_x = batch_x.float().to(self.device)
                sleep_stages = sleep_stages.to(self.device)
                labels = labels.to(self.device)
                mask = mask.to(self.device)

                outs = self.model(batch_x, mask=mask)
                loss, logs = compute_losses(
                    outs, y_stage=sleep_stages, y_event=labels,
                    w_stage=w_stage, w_event=w_event, mask=mask
                )

                total_losses.append(logs.get("loss", 0))
                stage_losses.append(logs.get("stage_loss", 0))
                event_losses.append(logs.get("event_loss", 0))

                # predictions
                if outs["stage_logits"] is not None:
                    stage_pred = torch.argmax(outs["stage_logits"], dim=-1)
                    preds_stage.append(stage_pred.detach().cpu())
                    trues_stage.append(sleep_stages.cpu())

                if outs["event_logits"] is not None:
                    event_pred = torch.argmax(outs["event_logits"], dim=-1)
                    preds_event.append(event_pred.detach().cpu())
                    trues_event.append(labels.cpu())

        # --- aggregate metrics ---
        val_loss_total = float(np.mean(total_losses))
        val_loss_stage = float(np.mean(stage_losses))
        val_loss_event = float(np.mean(event_losses))

        if preds_stage:
            preds_stage_np = torch.cat(preds_stage, dim=0).numpy().reshape(-1)
            trues_stage_np = torch.cat(trues_stage, dim=0).numpy().reshape(-1)
            acc_stage = cal_accuracy(preds_stage_np, trues_stage_np)
            mf1_stage = cal_macro_f1(trues_stage_np, preds_stage_np)
        else:
            acc_stage = mf1_stage = 0.0

        if preds_event:
            preds_event_np = torch.cat(preds_event, dim=0).numpy().reshape(-1)
            trues_event_np = torch.cat(trues_event, dim=0).numpy().reshape(-1)
            acc_event = cal_accuracy(preds_event_np, trues_event_np)
            mf1_event = cal_macro_f1(trues_event_np, preds_event_np)
        else:
            acc_event = mf1_event = 0.0

        self.model.train()
        return val_loss_stage, val_loss_event, val_loss_total, acc_stage, acc_event, mf1_stage, mf1_event

    # --------------------------
    # Training loop
    # --------------------------
    def train(self, setting):
        """Main training loop"""
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        model_optim = self._select_optimizer()
        early_stopping = EarlyStopping(self.args, patience=self.args.patience,
                                       path=os.path.join(self.args.checkpoints, f'checkpoint_{self.args.model_id}.pth'))

        w_stage, w_event = self.args.stage_loss_weight, self.args.event_loss_weight
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer=model_optim,
                                                  steps_per_epoch=len(train_loader),
                                                  epochs=self.args.train_epochs,
                                                  max_lr=self.args.learning_rate)

        #wandb.init(project=self.args.wandb_project, config={"epochs": self.args.train_epochs})

        for epoch in range(self.args.train_epochs):
            self.model.train()
            loss_stage_all, loss_event_all, loss_total_all = [], [], []
            preds_stage, preds_event, trues_stage, trues_event = [], [], [], []

            for i, (batch_x, sleep_stages, labels, serial, mask) in enumerate(
                    tqdm(train_loader, desc=f"Epoch {epoch+1}")):
                batch_x, sleep_stages, labels, mask = \
                    batch_x.to(self.device), sleep_stages.to(self.device), labels.to(self.device), mask.to(self.device)

                outs = self.model(batch_x, mask=mask)
                loss, logs = compute_losses(outs, y_stage=sleep_stages, y_event=labels,
                                            w_stage=w_stage, w_event=w_event, mask=mask)
                model_optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 4.0)
                model_optim.step()
                scheduler.step()

                loss_total_all.append(logs.get("loss", 0))
                loss_stage_all.append(logs.get("stage_loss", 0))
                loss_event_all.append(logs.get("event_loss", 0))

                # preds
                if outs["stage_logits"] is not None:
                    stage_pred = torch.argmax(outs["stage_logits"], dim=-1)
                    preds_stage.append(stage_pred.detach().cpu())
                    trues_stage.append(sleep_stages.cpu())
                if outs["event_logits"] is not None:
                    event_pred = torch.argmax(outs["event_logits"], dim=-1)
                    preds_event.append(event_pred.detach().cpu())
                    trues_event.append(labels.cpu())

            # === aggregate train metrics ===
            train_loss_stage = float(np.mean(loss_stage_all))
            train_loss_event = float(np.mean(loss_event_all))
            train_loss_total = float(np.mean(loss_total_all))

            if preds_stage:
                preds_stage_np = torch.cat(preds_stage, dim=0).numpy().reshape(-1)
                trues_stage_np = torch.cat(trues_stage, dim=0).numpy().reshape(-1)
                train_acc_stage = cal_accuracy(preds_stage_np, trues_stage_np)
                train_mf1_stage = cal_macro_f1(trues_stage_np, preds_stage_np)
            else:
                train_acc_stage = train_mf1_stage = 0.0

            if preds_event:
                preds_event_np = torch.cat(preds_event, dim=0).numpy().reshape(-1)
                trues_event_np = torch.cat(trues_event, dim=0).numpy().reshape(-1)
                train_acc_event = cal_accuracy(preds_event_np, trues_event_np)
                train_mf1_event = cal_macro_f1(trues_event_np, preds_event_np)
            else:
                train_acc_event = train_mf1_event = 0.0

            # === validation ===
            val_loss_stage, val_loss_event, val_loss_total, val_acc_stage, val_acc_event, val_mf1_stage, val_mf1_event = \
                self.vali(vali_data, vali_loader, w_stage, w_event)

            print(f"\nEpoch {epoch+1} Summary:")
            print(f"Stage | Train Loss: {train_loss_stage:.4f}, Acc: {train_acc_stage:.3f}, MF1: {train_mf1_stage:.3f} | "
                  f"Val Loss: {val_loss_stage:.4f}, Acc: {val_acc_stage:.3f}, MF1: {val_mf1_stage:.3f}")
            print(f"Event | Train Loss: {train_loss_event:.4f}, Acc: {train_acc_event:.3f}, MF1: {train_mf1_event:.3f} | "
                  f"Val Loss: {val_loss_event:.4f}, Acc: {val_acc_event:.3f}, MF1: {val_mf1_event:.3f}")

            # wandb.log({
            #     "epoch": epoch + 1,
            #     "train_loss_stage": train_loss_stage,
            #     "train_loss_event": train_loss_event,
            #     "train_loss_total": train_loss_total,
            #     "train_acc_stage": train_acc_stage,
            #     "train_acc_event": train_acc_event,
            #     "train_mf1_stage": train_mf1_stage,
            #     "train_mf1_event": train_mf1_event,
            #     "val_loss_stage": val_loss_stage,
            #     "val_loss_event": val_loss_event,
            #     "val_loss_total": val_loss_total,
            #     "val_acc_stage": val_acc_stage,
            #     "val_acc_event": val_acc_event,
            #     "val_mf1_stage": val_mf1_stage,
            #     "val_mf1_event": val_mf1_event,
            # })

            # === early stopping 모니터링 (event F1 기준) ===
            if self.args.task_mode == 'mtl':
                monitor_value = (w_stage * val_mf1_stage + w_event * val_mf1_event) / max((w_stage + w_event), 1e-8)
            elif self.args.task_mode == 'event':
                monitor_value = val_mf1_event
            else:  # stage-only
                monitor_value = val_mf1_stage

            early_stopping(monitor_value, self.model)

            if early_stopping.early_stop:
                print("Early stopping"); break

       # wandb.finish()
        print("✅ Training complete")
        return self.model