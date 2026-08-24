import os
import torch

from models import ModernTCN, MultiTask


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {

            'ModernTCN':ModernTCN,
            'MultiTask':MultiTask


        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args)
        if self.args.use_gpu:
            if self.args.use_multi_gpu:
                model = torch.nn.DataParallel(model)
            model = model.to(self.device)
        return model

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.devices)
            if self.args.use_multi_gpu:
                device = torch.device('cuda:0')  # 첫 번째 GPU를 기본으로 설정
                print(f'Using Multi-GPU: {self.args.devices}')
            else:
                device = torch.device(f'cuda:{self.args.gpu}')
                print(f'Using Single GPU: cuda:{self.args.gpu}')
        else:
            device = torch.device('cpu')
            print('Using CPU')
        return device


    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
