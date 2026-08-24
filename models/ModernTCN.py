import torch
from torch import nn
import torch.nn.functional as F
import math
from layers.RevIN import RevIN
from models.ModernTCN_Layer import series_decomp, Flatten_Head


class LayerNorm(nn.Module):

    def __init__(self, channels, eps=1e-6, data_format="channels_last"):
        super(LayerNorm, self).__init__()
        self.norm = nn.Layernorm(channels)

    def forward(self, x):

        B, M, D, N = x.shape
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(B * M, N, D)
        x = self.norm(
            x)
        x = x.reshape(B, M, N, D)
        x = x.permute(0, 1, 3, 2)
        return x

def get_conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
    return nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=bias)


def get_bn(channels):
    return nn.BatchNorm1d(channels)

def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1,bias=False):
    if padding is None:
        padding = kernel_size // 2
    result = nn.Sequential()
    result.add_module('conv', get_conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias))
    result.add_module('bn', get_bn(out_channels))
    return result

def fuse_bn(conv, bn):

    kernel = conv.weight
    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps
    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1)
    return kernel * t, beta - running_mean * gamma / std

class ReparamLargeKernelConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, groups,
                 small_kernel,
                 small_kernel_merged=False, nvars=7):
        super(ReparamLargeKernelConv, self).__init__()
        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        # We assume the conv does not change the feature map size, so padding = k//2. Otherwise, you may configure padding as you wish, and change the padding of small_conv accordingly.
        padding = kernel_size // 2
        if small_kernel_merged: 
            self.lkb_reparam = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=1, groups=groups, bias=True)
        else:
            self.lkb_origin = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                        stride=stride, padding=padding, dilation=1, groups=groups,bias=False) # large 커널의 Conv + BN 
            if small_kernel is not None:
                assert small_kernel <= kernel_size, 'The kernel size for re-param cannot be larger than the large kernel!'
                self.small_conv = conv_bn(in_channels=in_channels, out_channels=out_channels,
                                            kernel_size=small_kernel,
                                            stride=stride, padding=small_kernel // 2, groups=groups, dilation=1,bias=False) # small 커널의 Conv + BN


    def forward(self, inputs):

        if hasattr(self, 'lkb_reparam'):
            out = self.lkb_reparam(inputs)
        else:
            out = self.lkb_origin(inputs)
            if hasattr(self, 'small_conv'):
                out += self.small_conv(inputs)
        return out

    def PaddingTwoEdge1d(self,x,pad_length_left,pad_length_right,pad_values=0):

        D_out,D_in,ks=x.shape
        if pad_values ==0:
            pad_left = torch.zeros(D_out,D_in,pad_length_left)
            pad_right = torch.zeros(D_out,D_in,pad_length_right)
        else:
            pad_left = torch.ones(D_out, D_in, pad_length_left) * pad_values
            pad_right = torch.ones(D_out, D_in, pad_length_right) * pad_values
        x = torch.cat([pad_left,x],dims=-1)
        x = torch.cat([x,pad_right],dims=-1)
        return x

    def get_equivalent_kernel_bias(self): #두 개의 브랜치(Large Kernel + Small Kernel)를 병합하는 함수
        eq_k, eq_b = fuse_bn(self.lkb_origin.conv, self.lkb_origin.bn)
        if hasattr(self, 'small_conv'):
            small_k, small_b = fuse_bn(self.small_conv.conv, self.small_conv.bn)
            eq_b += small_b
            eq_k += self.PaddingTwoEdge1d(small_k, (self.kernel_size - self.small_kernel) // 2,
                                          (self.kernel_size - self.small_kernel) // 2, 0)
        return eq_k, eq_b

    def merge_kernel(self): # 학습 단계에서 추론 단계로 전환하는데 사용
        eq_k, eq_b = self.get_equivalent_kernel_bias()
        self.lkb_reparam = nn.Conv1d(in_channels=self.lkb_origin.conv.in_channels,
                                     out_channels=self.lkb_origin.conv.out_channels,
                                     kernel_size=self.lkb_origin.conv.kernel_size, stride=self.lkb_origin.conv.stride,
                                     padding=self.lkb_origin.conv.padding, dilation=self.lkb_origin.conv.dilation,
                                     groups=self.lkb_origin.conv.groups, bias=True)
        self.lkb_reparam.weight.data = eq_k
        self.lkb_reparam.bias.data = eq_b
        self.__delattr__('lkb_origin')
        if hasattr(self, 'small_conv'):
            self.__delattr__('small_conv')


class Block(nn.Module):
    def __init__(self, large_size, small_size, dmodel, dff, nvars, small_kernel_merged=False, drop=0.1):

        super(Block, self).__init__()
        self.dw = ReparamLargeKernelConv(in_channels=nvars * dmodel, out_channels=nvars * dmodel,
                                         kernel_size=large_size, stride=1, groups=nvars * dmodel,
                                         small_kernel=small_size, small_kernel_merged=small_kernel_merged, nvars=nvars) #  Large Kernel Conv
        self.norm = nn.BatchNorm1d(dmodel)

        # ConvFFN1: Feature Transformation
        self.ffn1pw1 = nn.Conv1d(in_channels=nvars * dmodel, out_channels=nvars * dff, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=nvars) # Pointwise # Conv1D (확장)
        self.ffn1act = nn.GELU()  # 활성화 함수 Activation (GELU)
        self.ffn1pw2 = nn.Conv1d(in_channels=nvars * dff, out_channels=nvars * dmodel, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=nvars) # 확장시켰다가 다시 원래 shape으로 되돌리기 Conv1D (축소)
        self.ffn1drop1 = nn.Dropout(drop)
        self.ffn1drop2 = nn.Dropout(drop)

        # ConvFFN2: Second Feature Transformation
        self.ffn2pw1 = nn.Conv1d(in_channels=nvars * dmodel, out_channels=nvars * dff, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=dmodel)
        self.ffn2act = nn.GELU()
        self.ffn2pw2 = nn.Conv1d(in_channels=nvars * dff, out_channels=nvars * dmodel, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=dmodel)
        self.ffn2drop1 = nn.Dropout(drop)
        self.ffn2drop2 = nn.Dropout(drop)

        self.ffn_ratio = dff//dmodel


    def forward(self, x):
        input = x  # Residual 연결을 위해 원본 입력 저장
        B, M, D, N = x.shape  # 입력 shape: (Batch, Variables, Dimensions, Time Steps)

        # 1. ReparamLargeKernelConv
        x = x.reshape(B, M * D, N)  # 입력 차원 변경: (B, M * D, N)
        x = self.dw(x)  # Large Kernel Conv 적용
        x = x.reshape(B, M, D, N)  # 원래 차원으로 복구: (B, M, D, N)

        # 2. Batch Normalization
        x = x.reshape(B * M, D, N)  # 변수 축을 배치로 합침: (B * M, D, N)
        x = self.norm(x)  # BatchNorm 적용
        x = x.reshape(B, M, D, N)  # 다시 복구: (B, M, D, N)

        # 3. 첫 번째 ConvFFN
        x = x.reshape(B, M * D, N)  # 차원 변경: (B, M * D, N)
        x = self.ffn1drop1(self.ffn1pw1(x))  # Conv1d -> Dropout
        x = self.ffn1act(x)  # GELU 활성화
        x = self.ffn1drop2(self.ffn1pw2(x))  # Conv1d -> Dropout
        x = x.reshape(B, M, D, N)  # 원래 형태로 복구: (B, M, D, N)

        # 4. 두 번째 ConvFFN
        x = x.permute(0, 2, 1, 3)  # 차원 재배열: (B, D, M, N)
        x = x.reshape(B, D * M, N)  # 차원 변경: (B, D * M, N)
        x = self.ffn2drop1(self.ffn2pw1(x))  # Conv1d -> Dropout
        x = self.ffn2act(x)  # GELU 활성화
        x = self.ffn2drop2(self.ffn2pw2(x))  # Conv1d -> Dropout
        x = x.reshape(B, D, M, N)  # 원래 형태로 복구: (B, D, M, N)
        x = x.permute(0, 2, 1, 3)  # 차원 재배열: (B, M, D, N)
        # 5. Residual 연결
        x = input + x  # Skip Connection

        return x


class Stage(nn.Module):
    def __init__(self, ffn_ratio, num_blocks, large_size, small_size, dmodel, dw_model, nvars,
                 small_kernel_merged=False, drop=0.1):

        super(Stage, self).__init__()
        d_ffn = dmodel * ffn_ratio
        blks = []
        for i in range(num_blocks):
            blk = Block(large_size=large_size, small_size=small_size, dmodel=dmodel, dff=d_ffn, nvars=nvars, small_kernel_merged=small_kernel_merged, drop=drop)
            blks.append(blk)

        self.blocks = nn.ModuleList(blks)

    def forward(self, x):

        for blk in self.blocks:
            x = blk(x)

        return x


class ModernTCN(nn.Module):  
    def __init__(self, task_name, patch_size, patch_stride, stem_ratio, downsample_ratio, ffn_ratio,
                 num_blocks, large_size, small_size, dims, dw_dims, nvars, small_kernel_merged=False,
                 backbone_dropout=0.1, head_dropout=0.1, use_multi_scale=True, revin=True, affine=True,
                 subtract_last=False, freq=None, seq_len=512, c_in=7, individual=False, target_window=96,
                 class_drop=0., class_num=10, num_epochs_per_window=1):
        """
        ModernTCN 모델 초기화 함수
        Args:
            task_name: 작업 이름 (e.g., 'classification')
            patch_size: 입력 데이터를 나눌 패치 크기 (N에 해당하는..)
            patch_stride: 패치를 나눌 때의 stride
            stem_ratio: 첫 번째 stem 레이어에서 downsampling 비율
            downsample_ratio: 각 Stage에서 downsampling 비율
            ffn_ratio: Feed Forward Network의 확장 비율
            num_blocks: 각 Stage에서 사용할 Block의 개수 리스트
            large_size: ReparamLargeKernelConv의 큰 커널 크기 리스트
            small_size: 작은 커널 크기 리스트
            dims: 각 Stage의 채널 수 리스트
            dw_dims: depthwise convolution 채널 수 리스트
            nvars: 입력 변수(채널) 개수
            backbone_dropout: Backbone 네트워크 드롭아웃 비율
            head_dropout: Head 레이어의 드롭아웃 비율
            use_multi_scale: 멀티스케일 정보를 사용할지 여부
            revin: 입력 데이터 정규화 여부 (RevIN 사용)
            affine: RevIN에서 사용할 affine 파라미터 여부
            subtract_last: RevIN에서 마지막 값을 뺄지 여부
            seq_len: 입력 시퀀스 길이
            c_in: 입력 채널 수
            target_window: 타겟 윈도우 길이
            class_drop: Classification head에서 드롭아웃 비율
            class_num: 분류 클래스 개수
        """
        super(ModernTCN, self).__init__()  # nn.Module 초기화
        self.task_name = task_name  # 작업 이름 저장 (e.g., 'classification')
        self.class_drop = class_drop  # Classification 드롭아웃 비율 설정
        self.class_num = class_num  # 분류 클래스 개수 설정
        self.dims = dims

        # 기본 파라미터 저장
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.downsample_ratio = downsample_ratio
        self.num_epochs_per_window = num_epochs_per_window

        # RevIN (Reversible Instance Normalization) 설정
        self.revin = revin
        if self.revin:
            self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)

        # Stem Layer와 Downsampling Layers 정의
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv1d(1, dims[0], kernel_size=patch_size, stride=patch_stride),  # Conv1d로 입력을 patchify + 임베딩
            nn.BatchNorm1d(dims[0])  # Batch Normalization 적용
        )
        self.downsample_layers.append(stem)  # Stem layer를 downsampling layers에 추가

        self.num_stage = len(num_blocks)  # Stage의 수는 num_blocks 리스트의 길이로 결정
        if self.num_stage > 1:  # Stage가 2개 이상이면 추가적인 다운샘플링 레이어 생성
            for i in range(self.num_stage - 1):
                downsample_layer = nn.Sequential(
                    nn.BatchNorm1d(dims[i]),  # 각 Stage에서 BatchNorm 적용
                    nn.Conv1d(dims[i], dims[i + 1], kernel_size=downsample_ratio, stride=downsample_ratio)
                )
                self.downsample_layers.append(downsample_layer)  # 다운샘플링 레이어를 추가


        # backbone
        self.num_stage = len(num_blocks)
        # Backbone 네트워크의 Stage 구성
        self.stages = nn.ModuleList()
        for stage_idx in range(self.num_stage):  # 각 Stage에 대해 반복
            layer = Stage(ffn_ratio, num_blocks[stage_idx], large_size[stage_idx], small_size[stage_idx],
                          dmodel=dims[stage_idx], dw_model=dw_dims[stage_idx], nvars=nvars,
                          small_kernel_merged=small_kernel_merged, drop=backbone_dropout)
            self.stages.append(layer)  # 각 Stage를 stages 리스트에 추가

        # Head Layer 정의
        patch_num = seq_len // patch_stride  # 전체 입력 시퀀스를 패치 단위로 나눈 개수
        self.n_vars = c_in  # 입력 변수(채널) 수 저장
        self.individual = individual  # 개별 학습 여부 설정
        d_model = dims[self.num_stage - 1]  # 마지막 Stage의 출력 채널 수

        # Multi-scale Head 여부에 따라 Head 설정
        if use_multi_scale:
            self.head_nf = d_model * patch_num  # Multi-scale 사용 시 최종 feature 개수
            self.head = Flatten_Head(self.individual, self.n_vars, self.head_nf, target_window, head_dropout)
        else:
            if patch_num % pow(downsample_ratio, (self.num_stage - 1)) == 0:
                self.head_nf = d_model * patch_num // pow(downsample_ratio, (self.num_stage - 1))
            else:
                self.head_nf = d_model * (patch_num // pow(downsample_ratio, (self.num_stage - 1)) + 1)
            self.head = Flatten_Head(self.individual, self.n_vars, self.head_nf, target_window, head_dropout)

        # Classification Task 설정
        if self.task_name == 'classification':

            self.act_class = F.gelu  # GELU 활성화 함수 사용
            self.class_dropout = nn.Dropout(self.class_drop)  # Classification 드롭아웃 설정
            self.head_class = None

    def forward_feature(self, x, te=None):
        """
        Backbone을 통과하면서 특징을 추출하는 함수
        Args:
            x: 입력 데이터 (B, M, L)
            te: Time Embedding (옵션, 사용하지 않음)
        Returns:
            Backbone을 통과한 특징 맵
        
        """
        B, M, L = x.shape  # 입력 데이터 shape: Batch, Variables, Sequence Length
        x = x.unsqueeze(-2)  # (B, M, L) → (B, M, 1, L)


        # 예: patchify + downsample

        #Downsampling 및 Stage 통과
        for i in range(self.num_stage):
            B, M, D, N = x.shape  # 각 단계의 입력 shape
            x = x.reshape(B * M, D, N)  # (B, M, D, N) → (B*M, D, N)
            if i == 0:  # 패딩 로직 : patch size와 stride가 같은 값이 아닌경우 마지막 값을 반복해서 패딩
                if self.patch_size != self.patch_stride:  # 첫 번째 Stage의 패딩 처리
                    pad_len = self.patch_size - self.patch_stride
                    pad = x[:, :, -1:].repeat(1, 1, pad_len)
                    x = torch.cat([x, pad], dim=-1)
            else:  # 다운샘플링 패딩 : 다운샘플링 ratio로 나누어떨어지지 않으면 패딩 추가 . 이것도 끝부분을 반복해서 복사
                if N % self.downsample_ratio != 0:  # Stage마다 다운샘플링 후 남은 부분 처리
                    pad_len = self.downsample_ratio - (N % self.downsample_ratio)
                    x = torch.cat([x, x[:, :, -pad_len:]], dim=-1)
            x = self.downsample_layers[i](x)  # Downsampling Layer 통과
            _, D_, N_ = x.shape  # 다운샘플링 후 Shape 확인
            x = x.reshape(B, M, D_, N_)  # 다시 원래 Shape로 복원
            x = self.stages[i](x)  # Backbone Stage를 통과
            #print(f"Stage {i} N: {N_}")

        self.current_N = N_
        return x

    def classification(self, x):
        """
        Classification 작업 수행
        Args:
            x: 입력 데이터
        Returns:
            Classification 결과
        """
        x = self.forward_feature(x, te=None)  # Backbone을 통과해 특징 추출
        x = self.act_class(x)  # 활성화 함수 적용 (GELU)
        x = self.class_dropout(x)  # 드롭아웃 적용
        x = x.reshape(x.shape[0], -1)  # 데이터를 평탄화
        x = self.head_class(x)  # 최종 Linear Layer를 통과
        return x

    def forward(self, x, te=None):
        """
        Forward 함수
        Args:
            x: 입력 데이터
            te: Time Embedding (옵션)
        Returns:
            모델의 출력 결과
        """
        x = self.classification(x)
        return x

    def structural_reparam(self):
        """ Reparameterization을 수행하여 Conv 레이어를 병합 """
        for m in self.modules():
            if hasattr(m, 'merge_kernel'):  # merge_kernel 메서드가 있는 경우
                m.merge_kernel()


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        # hyper param
        self.task_name = configs.task_name
        self.stem_ratio = configs.stem_ratio
        self.downsample_ratio = configs.downsample_ratio
        self.ffn_ratio = configs.ffn_ratio
        self.num_blocks = configs.num_blocks
        self.large_size = configs.large_size
        self.small_size = configs.small_size
        self.dims = configs.dims
        self.dw_dims = configs.dw_dims

        self.nvars = configs.enc_in
        self.small_kernel_merged = configs.small_kernel_merged
        self.drop_backbone = configs.dropout
        self.drop_head = configs.head_dropout
        self.use_multi_scale = configs.use_multi_scale
        self.revin = configs.revin
        self.affine = configs.affine
        self.subtract_last = configs.subtract_last

        self.freq = configs.freq
        self.seq_len = configs.seq_len
        self.c_in = self.nvars,
        self.individual = configs.individual
        self.target_window = configs.pred_len

        self.kernel_size = configs.kernel_size
        self.patch_size = configs.patch_size
        self.patch_stride = configs.patch_stride

        #classification
        self.class_dropout = configs.class_dropout
        self.class_num = configs.num_class


        # decomp
        self.decomposition = configs.decomposition



        self.model = ModernTCN(task_name=self.task_name,patch_size=self.patch_size, patch_stride=self.patch_stride, stem_ratio=self.stem_ratio,
                           downsample_ratio=self.downsample_ratio, ffn_ratio=self.ffn_ratio, num_blocks=self.num_blocks,
                           large_size=self.large_size, small_size=self.small_size, dims=self.dims, dw_dims=self.dw_dims,
                           nvars=self.nvars, small_kernel_merged=self.small_kernel_merged,
                           backbone_dropout=self.drop_backbone, head_dropout=self.drop_head,
                           use_multi_scale=self.use_multi_scale, revin=self.revin, affine=self.affine,
                           subtract_last=self.subtract_last, freq=self.freq, seq_len=self.seq_len, c_in=self.c_in,
                           individual=self.individual, target_window=self.target_window,
                            class_drop = self.class_dropout, class_num = self.class_num)

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec, mask=None):
        x = x.permute(0, 2, 1)
        te = None
        x = self.model(x, te)
        return x


