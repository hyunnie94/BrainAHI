import numpy as np
import torch
import matplotlib.pyplot as plt
import time

plt.switch_backend('agg')

from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

def calculate_auroc(trues, probs, num_classes):
    if num_classes == 2:
        if probs.ndim == 2: 
            probs = probs[:, 1] 
        auroc = roc_auc_score(trues, probs)
    else:
        trues_binarized = label_binarize(trues, classes=list(range(num_classes)))
        auroc = roc_auc_score(trues_binarized, probs, average="macro", multi_class="ovr")
    
    return auroc

def cal_macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')

def adjust_learning_rate(optimizer, scheduler, epoch, args, warmup_epochs=3, printout=True):
    """
    학습률 조정 함수
    """
    if args.lradj == 'warmup':
        # Warm-up 스케줄
        base_lr = args.learning_rate
        if epoch <= warmup_epochs:
            lr = base_lr * (epoch / warmup_epochs)  # 선형 증가
        else:
            lr = base_lr * (0.5 ** ((epoch - warmup_epochs)))  # 이후 감소
    elif args.lradj == 'type1':
        lr = args.learning_rate * (0.5 ** ((epoch - 1) // 1))
    elif args.lradj == 'type2':
        lr_dict = {2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 10: 5e-7, 15: 1e-7, 20: 5e-8}
        lr = lr_dict.get(epoch, args.learning_rate)  # 기본값은 초기 학습률
    elif args.lradj == 'type3':
        lr = args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** (epoch - 3))
    elif args.lradj == 'type4':
        lr = args.learning_rate if epoch < 20 else args.learning_rate * (0.5 ** ((epoch // 20)))
    elif args.lradj == 'constant':
        lr = args.learning_rate  # 일정한 학습률
    elif args.lradj == 'TST':
        lr = scheduler.get_last_lr()[0] if scheduler is not None else args.learning_rate
    else:
        raise ValueError(f"Unknown lradj type: {args.lradj}")
    
    # 학습률 업데이트
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    if printout:
        print(f"Epoch {epoch}: Updating learning rate to {lr}")


class EarlyStopping:
    def __init__(self, args, patience=10, verbose=False, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): 모델 개선 없을 시 학습 중단 epoch 수
            verbose (bool): True면, 학습 중단 메시지를 출력
            delta (float): 성능 향상 최소값 기준
            path (str): 모델 저장 경로
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.path = path  

    def __call__(self, mf1, model):
        score = mf1

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(mf1, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(mf1, model)
            self.counter = 0

    def __call__(self, mf1, model):
        score = mf1

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(mf1, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            old_best_score = self.best_score  
            self.best_score = score
            self.save_checkpoint(mf1, model, old_best_score)  
            self.counter = 0

    def save_checkpoint(self, mf1, model, old_best_score=None):
        """현재 모델 저장"""
        if self.verbose and old_best_score is not None:
            print(f'MF1 improved ({old_best_score:.4f} --> {mf1:.4f}). Saving model ...')  
        torch.save(model.state_dict(), self.path)  


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')

def test_params_flop(model,x_shape):
    """
    If you want to thest former's flop, you need to give default value to inputs in model.forward(), the following code can only pass one argument to forward()
    """
    model_params = 0
    for parameter in model.parameters():
        model_params += parameter.numel()
        print('INFO: Trainable parameter count: {:.2f}M'.format(model_params / 1000000.0))
    from ptflops import get_model_complexity_info    
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model.cuda(), x_shape, as_strings=True, print_per_layer_stat=True)
        # print('Flops:' + flops)
        # print('Params:' + params)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))

def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred

def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)