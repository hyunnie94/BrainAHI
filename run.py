import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import torch
import json

print(f"Available GPUs: {torch.cuda.device_count()}")
print(f"Using multi-GPU: {torch.cuda.device_count() > 1}")
print(f"Current device: {torch.cuda.current_device()}")


torch.autograd.set_detect_anomaly(True)

from exp.exp_multitask_train import Exp_Classification_Train as Exp_Classification

import random
import numpy as np
from utils.str2bool import str2bool

def load_centers(path):
    if path is None or not os.path.exists(path):
        return {}
    with open(path, 'r') as f:
        return json.load(f)

parser = argparse.ArgumentParser(description='ModernTCN')

# random seed
parser.add_argument('--random_seed', type=int, default=2025, help='random seed')

# basic config
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model', type=str, required=True, default='ModernTCN',
                    help='model name, options: [ModernTCN]')

# data loader
parser.add_argument('--data', type=str, required=True, default='KISS', help='dataset type')
parser.add_argument('--dataset_name', type=str, required=True, default='kiss', help='test dataset type')
parser.add_argument('--root_path', type=str, default='/home/data_OL1/hklee/BrainAHI/', help='root path of the data file')
parser.add_argument('--train_centers_json', type=str, help='Path to JSON config for training centers')
parser.add_argument('--val_centers_json', type=str, help='Path to JSON config for validation centers')
parser.add_argument('--test_centers_json', type=str, help='Path to JSON config for test centers')
parser.add_argument('--internal_centers', nargs='+', default=['kiss', 'mesa', 'mros1', 'mros2'],
                    help="List of internal centers for testing")
parser.add_argument('--external_centers', nargs='+', default=['cpap', 'shhs1', 'shhs2'],
                    help="List of external centers for testing")

parser.add_argument('--split_path', type=str, default='/home/data_OL1/hklee/BrainAHI/codes/splits/', help='train/test split file path')
parser.add_argument('--cache_path', type=str, default='/home/data_OL1/hklee/BrainAHI/cachefiles/', help='cache file path')
parser.add_argument('--num_epochs_per_window', type=int, default=1, help='Number of epochs per window')
parser.add_argument('--window_stride', type=int, default=3840, help='stride of window')

parser.add_argument('--stride', type=int, default=3840, help='stride between sequences')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task') 
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='../checkpoints/', help='location of model checkpoints') 
parser.add_argument('--test_results', type=str, default='../test_results/', help='location of model test results') 
parser.add_argument('--embed', type=str, default='timeF',      
                    help='time features encoding, options:[timeF, fixed, learned]') 
parser.add_argument('--scale', type=str2bool, default=False, help='Whether to apply scaling to the data')


# forecasting task
parser.add_argument('--seq_len', type=int, default=3840, help='input sequence length')
parser.add_argument('--seq_len_30s', type=int, default=3840, help='sequence length for 30s data')
parser.add_argument('--seq_len_1s', type=int, default=128, help='sequence length for 1s data')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')



#Multitask 
parser.add_argument('--use_multi_input', type=bool, default=True, help='use multiple inputs (e.g., 30s, 1s)')
parser.add_argument('--num_stage_classes', type=int, default=5, help='number of classes for sleep stage classification')
parser.add_argument('--num_event_classes', type=int, default=2, help='number of classes for event detection')

parser.add_argument('--stage_loss_weight', type=float, default=1.0, help='weight for stage classification loss')
parser.add_argument('--event_loss_weight', type=float, default=1.0, help='weight for event detection loss')
parser.add_argument('--use_attention', type=bool, default=True, help='use attention block in the shared backbone')


# ablation
parser.add_argument('--task_mode', type=str, default='mtl', choices=['mtl', 'stage', 'event'],
                    help='Ablation mode: mtl (multi-task), stage (STL-stage only), event (STL-event only)')
parser.add_argument('--use_stage_context', type=int, default=1,
                    help='Use stage logits as context for event head (MTL only, 1=yes, 0=no)')


#ModernTCN
parser.add_argument('--stem_ratio', type=int, default=6, help='stem ratio') #입력 채널을 초기 임베딩 차원으로 늘리는 비율. 입력 채널이 5(EEG band)라면, stem 출력 차원은 5 × 6 = 30
parser.add_argument('--downsample_ratio', type=int, default=1, help='downsample_ratio') # 다운샘플링 비율. 시퀀스 길이를 줄이는 정도 (1이면 줄이지 않음).
parser.add_argument('--ffn_ratio', type=int, default=2, help='ffn_ratio') # Feed-Forward Network(FFN)의 차원 확장 비율. 
parser.add_argument('--patch_size', type=int, default=16, help='the patch size') # 입력 시퀀스를 쪼개는 패치 크기 (샘플 단위).
parser.add_argument('--patch_stride', type=int, default=8, help='the patch stride')  # 패치 간 이동 간격.

parser.add_argument('--num_blocks', nargs='+',type=int, default=[1,1,1,1], help='num_blocks in each stage') #각 스테이지별 블록 수 (ex 4개 스테이지).
parser.add_argument('--large_size', nargs='+',type=int, default=[31,29,27,13], help='big kernel size') # 각 스테이지의 큰 커널 크기.
parser.add_argument('--small_size', nargs='+',type=int, default=[5,5,5,5], help='small kernel size for structral reparam') # 각 스테이지의 작은 커널 크기.
parser.add_argument('--dims', nargs='+',type=int, default=[256,256,256,256], help='dmodels in each stage') # 각 스테이지의 모델 차원 (채널 수).
parser.add_argument('--dw_dims', nargs='+',type=int, default=[256,256,256,256], help='dw dims in dw conv in each stage') # Depthwise Convolution의 차원.

parser.add_argument('--small_kernel_merged', type=str2bool, default=False, help='small_kernel has already merged or not') # 작은 커널이 구조적 재파라미터화로 합쳐졌는지 여부.
parser.add_argument('--call_structural_reparam', type=bool, default=False, help='structural_reparam after training') # 학습 후 구조적 재파라미터화 호출 여부.
parser.add_argument('--use_multi_scale', type=str2bool, default=True, help='use_multi_scale fusion') #멀티스케일 융합 사용 여부.


# PatchTST
parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')

# Formers
parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
parser.add_argument('--enc_in', type=int, default=5, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')

parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

# optimization
parser.add_argument('--num_workers', type=int, default=32, help='data loader num workers')
parser.add_argument('--itr', type=int, default=2, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=100, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='warmup', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

#multi task
parser.add_argument('--task_name', type=str, required=True, default='classification',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')

# inputation task
parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

# anomaly detection task
parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')

# classfication task
parser.add_argument('--class_dropout', type=float, default=0.05, help='classfication dropout')

## 피쳐 추출 모드
parser.add_argument('--extract_features', type=int, default=0, help='0: no feature extraction, 1: extract features only')

args = parser.parse_args()

# random seed
fix_seed = args.random_seed
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)


args.use_gpu = torch.cuda.is_available() and args.use_gpu

args.train_centers = load_centers(args.train_centers_json)
args.val_centers = load_centers(args.val_centers_json)
args.test_centers = load_centers(args.test_centers_json)

if args.use_gpu:
    # Parse GPU device IDs if using multi-GPU
    if args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')  # Remove spaces from device string
        device_ids = args.devices.split(',')  # Split into list of IDs
        args.device_ids = [int(id_) for id_ in device_ids]  # Convert to integers
        args.gpu = args.device_ids[0]  # Use the first GPU as the main GPU
    else:
        # Use only the first available GPU
        args.device_ids = [0]
        args.gpu = 0  # Default to the first GPU

print('Args in experiment:')
print(args)

if __name__ == '__main__':
    #Exp = Exp_Main
    if args.task_name == 'classification':
        Exp = Exp_Classification
    if args.large_size[0] < 13:
        args.small_kernel_merged = True

    exp = Exp(args)  # Experiment 객체 생성 (공통)

    # ★★ 실행 로직 수정 ★★
    setting = '{}_{}_{}_ft{}_sl{}_pl{}_dim{}_nb{}_lk{}_sk{}_ffr{}_ps{}_str{}_multi{}_merged{}_{}'.format(
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.pred_len,
        args.dims[0],
        args.num_blocks[0],
        args.large_size[0],
        args.small_size[0],
        args.ffn_ratio,
        args.patch_size,
        args.patch_stride,
        args.enc_in, 
        args.use_multi_scale,
        args.small_kernel_merged,
        args.des)

    # ==============================================================
    # 1️. 학습 모드 (Train + Test + Save Features)
    # ==============================================================
    if args.is_training == 1:
        from exp.exp_multitask_train import Exp_Classification_Train
        from exp.exp_multitask_test import Exp_MultiTask_Test

        exp = Exp_Classification_Train(args)
        for ii in range(args.itr):
            print(f'>>>>>>> start training : {setting} >>>>>>>>>>>>>>>>>>>>>>>>>>>>')
            exp.train(setting)

            print(f'>>>>>>> testing : {setting} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            tester = Exp_MultiTask_Test(exp.model, args, exp.device)
            tester.run_all(save_features=True)

            torch.cuda.empty_cache()

    # ==============================================================
    # 2️. 테스트 전용 모드 (Test only, no feature save)
    # ==============================================================
    elif args.is_training == 0 and args.extract_features == 0:
        print(f'>>>>>>> testing only : {setting} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        from exp.exp_multitask_train import Exp_Classification_Train
        from exp.exp_multitask_test import Exp_MultiTask_Test

        exp = Exp_Classification_Train(args)
        tester = Exp_MultiTask_Test(exp.model, args, exp.device)
        tester.run_all(save_features=False)

        torch.cuda.empty_cache()

    # ==============================================================
    # 3️. 피처 추출 모드
    # ==============================================================
    elif args.extract_features == 1:
        print(f'>>>>>>> extracting features : {setting} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        from exp.exp_feature_extract import FeatureExtractor
        from exp.exp_multitask_train import Exp_Classification_Train
        from data_provider.data_factory import data_provider

        exp = Exp_Classification_Train(args)
        data_set, data_loader = data_provider(args, flag='train')
        extractor = FeatureExtractor(exp.model, args, exp.device)
        save_path = os.path.join(args.checkpoints, args.model_id, f"patient_features_phase1_train.h5")
        extractor.extract(data_loader, data_set, save_path)

        torch.cuda.empty_cache()

    else:
        raise ValueError("Invalid combination of --is_training and --extract_features")