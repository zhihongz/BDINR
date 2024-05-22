# -----------------------------------------
# 🎯 Image quality assessment metrics used in image/video reconstruction and generation tasks
# -----------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pyiqa  # pip install pyiqa
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import mean_squared_error as compare_mse


# 💡 the inputs and outputs are in 'torch tensor' format

class IQA_Metric(nn.Module):
    """image quality assessment metric calculation using [pyiqa package](https://github.com/chaofengc/IQA-PyTorch)
    Note: use `print(pyiqa.list_models())` to list all available metrics
    """

    def __init__(self, metric_name: str, calc_mean: bool = True):
        super(IQA_Metric, self).__init__()
        self.__name__ = metric_name
        self.metric = pyiqa.create_metric(metric_name=metric_name)
        self.calc_mean = calc_mean

    def forward(self, output, target):
        with torch.no_grad():
            metric_score = self.metric(output, target)
        if self.calc_mean:
            return torch.mean(metric_score)
        else:
            return metric_score


# 💡 the inputs and outputs are in 'numpy ndarray' format

def calc_psnr(output, target):
    '''
    calculate psnr
    '''
    assert output.shape[0] == target.shape[0]
    total_psnr = np.zeros(output.shape[0])
    for k, (pred, gt) in enumerate(zip(output, target)):
        if pred.ndim == 3:
            pred = pred.transpose(1, 2, 0)
            gt = gt.transpose(1, 2, 0)
        total_psnr[k] = compare_psnr(pred, gt, data_range=1)
    return np.mean(total_psnr)


def calc_ssim(output, target):
    '''
    calculate ssim
    '''
    assert output.shape[0] == target.shape[0]
    total_ssim = np.zeros(output.shape[0])
    for k, (pred, gt) in enumerate(zip(output, target)):
        if pred.ndim == 3:
            pred = pred.transpose(1, 2, 0)
            gt = gt.transpose(1, 2, 0)
        total_ssim[k] = compare_ssim(
            pred, gt, data_range=1, multichannel=True)
    return np.mean(total_ssim)


def calc_mse(output, target):
    '''
    calculate mse
    '''
    assert output.shape[0] == target.shape[0]
    total_psnr = np.zeros(output.shape[0])
    for k, (pred, gt) in enumerate(zip(output, target)):
        if pred.ndim == 3:
            pred = pred.transpose(1, 2, 0)
            gt = gt.transpose(1, 2, 0)
        total_psnr[k] = compare_mse(pred, gt)
    return np.mean(total_psnr)
