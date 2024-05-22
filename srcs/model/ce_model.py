import torch
from torch import nn
import numpy as np
from srcs.model._basic_binary_modules import STEBinary_fc, ExpBinary_fc

BinaryDict = {'STEBinary_fc': STEBinary_fc, 'ExpBinary_fc': ExpBinary_fc}


class CEBlurNet(nn.Module):
    '''
    optimizable motion blur encoder
    - weight binarize: STE-LBSIGN
    '''

    def __init__(self, sigma_range=0, test_sigma_range=0, ce_code_n=8, frame_n=8, ce_code_init=None, opt_cecode=False, binary_fc=None):
        super(CEBlurNet, self).__init__()
        self.sigma_range = sigma_range
        self.test_sigma_range = test_sigma_range
        self.frame_n = frame_n  # frame num
        self.time_idx = torch.linspace(
            0, 1, ce_code_n).unsqueeze(0).t()  # time idx
        self.upsample_factor = frame_n//ce_code_n
        self.binary_fc = BinaryDict[binary_fc]
        self.ce_code_n = ce_code_n
        self.ce_weight = nn.Parameter(torch.Tensor(ce_code_n, 1))
        if ce_code_init is None:
            nn.init.uniform_(self.ce_weight, a=-1, b=1)  # initialize
        else:
            # convert 0,1 (ce_code) -> -1,1 (ce_weight)
            ce_code_init = [-1 if ce_code_init[k] ==
                            0 else 1 for k in range(len(ce_code_init))]
            self.ce_weight.data = torch.tensor(
                ce_code_init, dtype=torch.float32).unsqueeze(0).t()
        if not opt_cecode:
            # whether optimize ce code
            self.ce_weight.requires_grad = False

        # upsample matrix for ce_code(parameters)
        self.upsample_matrix = torch.zeros(
            self.upsample_factor * ce_code_n, ce_code_n)
        for k in range(ce_code_n):
            self.upsample_matrix[k *
                                 self.upsample_factor:(k+1)*self.upsample_factor, k] = 1

    def forward(self, frames):
        device = frames.device
        ce_code = self.binary_fc(self.ce_weight)  # weights binarized
        ce_code_up = torch.matmul(
            self.upsample_matrix.to(device), ce_code)
        assert ce_code_up.data.shape[0] == frames.shape[
            1], f'frame num({frames.shape[1]}) is not equal to CeCode length({ce_code_up.shape[0]})'

        ce_code_up_ = ce_code_up.view(self.frame_n, 1, 1, 1).expand_as(frames)
        ce_blur_img = torch.sum(
            ce_code_up_*frames, axis=1)/self.frame_n

        sigma_range = self.sigma_range if self.training else self.test_sigma_range
        if isinstance(sigma_range, (int, float)):
            noise_level = sigma_range
        else:
            noise_level = np.random.uniform(*sigma_range)

        ce_blur_img_noisy = ce_blur_img + torch.tensor(noise_level, device=device) * \
            torch.randn(ce_blur_img.shape, device=device)

        return ce_blur_img_noisy, self.time_idx.to(device), ce_code, ce_blur_img

