from torch.autograd import Function
from typing import Any, NewType
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------
# Binarized module by Straight-through estimator (STE)
# Straight-Through Estimator(STE) is used to approximate the gradient of sign function.
# See:
#     Bengio, Yoshua, Nicholas Léonard, and Aaron Courville.
#     "Estimating or propagating gradients through stochastic neurons for
#     conditional computation." arXiv preprint arXiv: 1308.3432 (2013).
# --------------------------------------------

# ----- Imp 1 Start -----
class LBSign(torch.autograd.Function):
    """Return -1 if x < 0, 1 if x > 0, 0 if x==0."""
    @staticmethod
    def forward(ctx, input):
        result = torch.sign(input)
        ctx.save_for_backward(input)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_output[input > 1] = 0
        grad_output[input < -1] = 0
        return grad_output
# ----- Imp 1 End -----

# ----- Imp 2 Start -----
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.


# A type where each element is in {-1, 1}
BinaryTensor = NewType('BinaryTensor', torch.Tensor)


def binary_sign(x: torch.Tensor) -> BinaryTensor:
    """Return -1 if x < 0, 1 if x >= 0."""
    return x.sign() + (x == 0).type(torch.float)  # type: ignore


class STESign(Function):
    """
    Binarize tensor using sign function.
    """

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> BinaryTensor:  # type: ignore
        """
        Return a Sign tensor.
        Args:
            ctx: context
            x: input tensor
        Returns:
            Sign(x) = (x>=0) - (x<0)
            Output type is float tensor where each element is either -1 or 1.
        """
        ctx.save_for_backward(x)
        sign_x = binary_sign(x)
        return sign_x

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:  # type: ignore  # pragma: no cover (since this is called by C++ code) # noqa: E501
        """
        Compute gradient using STE.
        Args:
            ctx: context
            grad_output: gradient w.r.t. output of Sign
        Returns:
            Gradient w.r.t. input of the Sign function
        """
        x, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[x.gt(1)] = 0
        grad_input[x.lt(-1)] = 0
        return grad_input


# Convenience function to binarize tensors
STESign_fc = STESign.apply    # results in -1|1
def STEBinary_fc(x):
    return (STESign_fc(x)+1)/2  # results in 0|1

# ----- Imp 2 End -----

# --------------------------------------------
# exponential function approximated binarize function
# --------------------------------------------

def ExpBinary_fc(input, expn=50):
    '''
    binarize input to 0|1
    # method2: use 1/(1+torch.exp(-expn*x)) to approximate
    '''
    return 1/(1+torch.exp(-expn*input))


# --------------------------------------------
# ReCU: Reviving the Dead Weights in Binary Neural Networks
# --------------------------------------------


class BinarizeConv2d(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeConv2d, self).__init__(*kargs, **kwargs)
        self.alpha = nn.Parameter(torch.rand(
            self.weight.size(0), 1, 1), requires_grad=True)
        self.register_buffer('tau', torch.tensor(1.))

    def forward(self, input):
        a = input
        w = self.weight

        w0 = w - w.mean([1, 2, 3], keepdim=True)
        w1 = w0 / \
            (torch.sqrt(w0.var([1, 2, 3], keepdim=True) +
             1e-5) / 2 / np.sqrt(2))
        EW = torch.mean(torch.abs(w1))
        Q_tau = (- EW * torch.log(2-2*self.tau)).detach().cpu().item()
        w2 = torch.clamp(w1, -Q_tau, Q_tau)

        if self.training:
            a0 = a / torch.sqrt(a.var([1, 2, 3], keepdim=True) + 1e-5)
        else:
            a0 = a

        #* binarize
        bw = BinaryQuantize().apply(w2)
        ba = BinaryQuantize_a().apply(a0)
        #* 1bit conv
        output = F.conv2d(ba, bw, self.bias,
                          self.stride, self.padding,
                          self.dilation, self.groups)
        #* scaling factor
        output = output * self.alpha
        return output


class BinaryQuantize(Function):
    @staticmethod
    def forward(ctx, input):
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class BinaryQuantize_a(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        grad_input = (2 - torch.abs(2*input))
        grad_input = grad_input.clamp(min=0) * grad_output.clone()
        return grad_input
