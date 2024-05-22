from cv2 import add
from torch.autograd import Variable
import torch.nn as nn
from torchvision.models import vgg16_bn
import torch
import torch.nn.functional as F
# from srcs.loss.PerceptualLoss import PerceptualLoss
from pytorch_msssim import ssim, ms_ssim  # pip install pytorch-msssim

# ===========================
# global loss info extract
# ===========================
LOSSES = {}


def add2loss(func):
    if func.__name__ in LOSSES:
        raise ValueError(f'{func.__name__} is already in the LOSSES list')
    else:
        LOSSES[func.__name__] = func
    return func

# ===========================
# weighted_loss
# ===========================


def weighted_loss(output, target, loss_conf_dict):
    """
    weighted multi-loss
    loss_conf_dict: {loss_type1: weight|[weight,{kwargs_dict}], ...}
    """
    loss_v = 0
    for k, v in loss_conf_dict.items():
        if isinstance(v, (float, int)):
            assert v >= 0, f"loss'weight {k}:{v} should be positive"
            loss_v += LOSSES[k](output, target)*v
        elif isinstance(v, list) and len(v) == 2:
            assert v[0] >= 0, f"loss'weight {k}:{v} should be positive"
            loss_v += LOSSES[k](output, target, **v[1])*v[0]
        else:
            raise ValueError(
                f"the Key({k})'s Value {v} in Dict(loss_conf_dict) should be scalar(weight) | list[weight, args] ")

    return loss_v


# ===========================
# basic_loss
# ===========================

# template
# @add2loss
# def test_loss(output, target, *args, **kwargs):
#     return _test_loss(output, target)


@add2loss
def nll_loss(output, target):
    return F.nll_loss(output, target)


@add2loss
def mse_loss(output, target):
    return F.mse_loss(output, target)


@add2loss
def l1_loss(output, target):
    return F.l1_loss(output, target)


@add2loss
def ssim_loss(output, target):
    # data range: 0-1
    ssim_loss = 1 - ssim(output, target,
                         data_range=1, size_average=True)
    return ssim_loss


@add2loss
def msssim_loss(output, target):
    # data range: 0-1
    ms_ssim_loss = 1 - ms_ssim(output, target,
                               data_range=1, size_average=True)
    return ms_ssim_loss


# @add2loss
def perceptual_loss(output, target):
    PL = PerceptualLoss()
    return PL(output, target)


@add2loss
def tv_loss(output, *args):
    # one input param, gt is not needed
    def _tensor_size(t):
        return t.size()[1]*t.size()[2]*t.size()[3]
    batch_size = output.size()[0]
    h_x = output.size()[2]
    w_x = output.size()[3]
    count_h = _tensor_size(output[:, :, 1:, :])
    count_w = _tensor_size(output[:, :, :, 1:])
    h_tv = torch.pow((output[:, :, 1:, :]-output[:, :, :h_x-1, :]), 2).sum()
    w_tv = torch.pow((output[:, :, :, 1:]-output[:, :, :, :w_x-1]), 2).sum()
    return 2*(h_tv/count_h+w_tv/count_w)/batch_size


@add2loss
def charbonnier_loss(output, target, eps=1e-3):
    diff = output - target
    loss = torch.mean(torch.sqrt((diff * diff) + eps**2))
    return loss


@add2loss
def fft_loss(output, target):
    diff = torch.fft.fft2(output) - torch.fft.fft2(target)
    loss = torch.mean(abs(diff))
    return loss


if __name__ == "__main__":
    import PerceptualLoss
    output = torch.ones(4, 3, 10, 10)
    target = torch.zeros(4, 3, 10, 10)
    loss_conf_dict = {'l1_loss': 0.5, 'mse_loss': 0.5}

    loss_v = weighted_loss(output, target, loss_conf_dict)
    print('loss: ', loss_v)
