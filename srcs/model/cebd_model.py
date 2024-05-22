import torch.nn as nn
from srcs.model.bd_model import BDNeRV_RC
from srcs.model.ce_model import CEBlurNet

class CEBDNet(nn.Module):
    '''
    coded exposure blur decomposition network
    '''
    def __init__(self, sigma_range=0, test_sigma_range=0, ce_code_n=8, frame_n=8, ce_code_init=None, opt_cecode=False, ce_net=None, binary_fc=None, bd_net=None):
        super(CEBDNet, self).__init__()
        self.ce_code_n = ce_code_n
        self.frame_n = frame_n
        self.bd_net = bd_net
        # coded exposure blur net
        if ce_net == 'CEBlurNet':
            self.BlurNet = CEBlurNet(
                sigma_range=sigma_range, test_sigma_range=test_sigma_range, ce_code_n=ce_code_n, frame_n=frame_n, ce_code_init=ce_code_init, opt_cecode=opt_cecode, binary_fc=binary_fc)
        else:
            raise NotImplementedError(f'No model named {ce_net}')

        # blur decomposition net
        if bd_net=='BDNeRV_RC':
            self.DeBlurNet = BDNeRV_RC()
        else:
            raise NotImplementedError(f'No model named {bd_net}')

    def forward(self, frames):
        ce_blur_img_noisy, time_idx, ce_code_up, ce_blur_img = self.BlurNet(
            frames)
        output = self.DeBlurNet(ce_blur=ce_blur_img_noisy, time_idx=time_idx, ce_code=ce_code_up)
        return output, ce_blur_img, ce_blur_img_noisy
