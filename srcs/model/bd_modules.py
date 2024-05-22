import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from srcs.model.bd_utils import NormLayer, pad2same_size, pad2size


# -------------------basic blocks ----------------
class Conv(nn.Module):
    def __init__(self, input_channels, n_feats, kernel_size, stride=1, padding=0, bias=True, bn=False, act=False):
        super(Conv, self).__init__()
        m = []
        m.append(nn.Conv2d(input_channels, n_feats,
                 kernel_size, stride, padding, bias=bias))
        if bn:
            m.append(nn.BatchNorm2d(n_feats))
        if act:
            m.append(nn.ReLU(True))
        self.body = nn.Sequential(*m)

    def forward(self, input):
        return self.body(input)


class Deconv(nn.Module):
    def __init__(self, input_channels, n_feats, kernel_size, stride=2, padding=0, output_padding=0, bias=True, act=False):
        super(Deconv, self).__init__()
        m = []
        m.append(nn.ConvTranspose2d(input_channels, n_feats, kernel_size,
                 stride=stride, padding=padding, output_padding=output_padding, bias=bias))
        if act:
            m.append(nn.ReLU(True))
        self.body = nn.Sequential(*m)

    def forward(self, input):
        return self.body(input)


class ResBlock(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, padding=0, bias=True, bn=False, act='relu', res_scale=1):
        super(ResBlock, self).__init__()
        m = []
        if act == 'relu':
            act = nn.ReLU(inplace=True)
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size,
                     padding=padding, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

# ----------------- MLP ---------------------
class Sin(nn.Module):
    def __init__(self, inplace: bool = False):
        super(Sin, self).__init__()
    
    def forward(self, input):
        return torch.sin(30 * input)  # see SIREN paper for the factor 30
    
def ActivationLayer(act_type):
    if act_type == 'relu':
        act_layer = nn.ReLU(True)
    elif act_type == 'leaky':
        act_layer = nn.LeakyReLU(inplace=True)
    elif act_type == 'leaky01':
        act_layer = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    elif act_type == 'relu6':
        act_layer = nn.ReLU6(inplace=True)
    elif act_type == 'gelu':
        act_layer = nn.GELU()
    elif act_type == 'sin':
        # act_layer = torch.sin
        act_layer = Sin()
    elif act_type == 'swish':
        act_layer = nn.SiLU(inplace=True)
    elif act_type == 'softplus':
        act_layer = nn.Softplus()
    elif act_type == 'hardswish':
        act_layer = nn.Hardswish(inplace=True)
    elif act_type == 'non':
        act_layer = nn.Identity()
    else:
        raise KeyError(f"Unknown activation function {act_type}.")
    
    return act_layer

def MLP(dim_list, act='relu', bias=True):
    act_fn = ActivationLayer(act)
    fc_list = []
    for i in range(len(dim_list) - 1):
        fc_list += [nn.Linear(dim_list[i], dim_list[i+1], bias=bias), act_fn]
    return nn.Sequential(*fc_list)

# ---------------- Transformer -----------------------

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = heads * dim_head
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0., prenorm=False):
        super(TransformerBlock, self).__init__()
        if prenorm:
            self.attn = PreNorm(dim, Attention(
                dim, heads=heads, dim_head=dim_head, dropout=dropout))
            self.ffn = PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
        else:
            self.attn = Attention(
                dim, heads=heads, dim_head=dim_head, dropout=dropout)
            self.ffn = FeedForward(dim, mlp_dim, dropout=dropout)

    def forward(self, x):
        x = self.attn(x) + x
        x = self.ffn(x) + x
        return x

# ----------------------- NERV ----------------------------------

class Conv_Up_Block(nn.Module):
    def __init__(self, **kargs):
        super().__init__()
        ngf = kargs['ngf']
        new_ngf = kargs['new_ngf']

        if ngf <= new_ngf:
            factor = 4
            self.conv1 = NeRV_CustomConv(
                ngf=ngf, new_ngf=ngf // factor, stride=kargs['stride'], bias=kargs['bias'], conv_type=kargs['conv_type'])
            self.conv2 = nn.Conv2d(
                ngf // factor, new_ngf, 3, 1, 1, bias=kargs['bias'])
        else:
            self.conv1 = nn.Conv2d(ngf, new_ngf, 3, 1, 1, bias=kargs['bias'])
            self.conv2 = NeRV_CustomConv(
                ngf=new_ngf, new_ngf=new_ngf, stride=kargs['stride'], bias=kargs['bias'], conv_type=kargs['conv_type'])
        self.norm = NormLayer(kargs['norm'], kargs['new_ngf'])
        self.act = ActivationLayer(kargs['act'])

    def forward(self, x):
        return self.act(self.norm(self.conv2(self.conv1(x))))
    
class NeRV_CustomConv(nn.Module):
    def __init__(self, **kargs):
        super(NeRV_CustomConv, self).__init__()

        ngf, new_ngf, stride = kargs['ngf'], kargs['new_ngf'], kargs['stride']
        self.conv_type = kargs['conv_type']
        if self.conv_type == 'conv':
            self.conv = nn.Conv2d(ngf, new_ngf * stride * stride, 3, 1, 1, bias=kargs['bias'])
            self.up_scale = nn.PixelShuffle(stride)
        elif self.conv_type == 'deconv':
            self.conv = nn.ConvTranspose2d(ngf, new_ngf, stride, stride)
            self.up_scale = nn.Identity()
        elif self.conv_type == 'bilinear':
            self.conv = nn.Upsample(scale_factor=stride, mode='bilinear', align_corners=True)
            self.up_scale = nn.Conv2d(ngf, new_ngf, 2*stride+1, 1, stride, bias=kargs['bias'])

    def forward(self, x):
        out = self.conv(x)
        return self.up_scale(out)

class NeRVBlock(nn.Module):
    def __init__(self, **kargs):
        super().__init__()

        self.conv = NeRV_CustomConv(ngf=kargs['ngf'], new_ngf=kargs['new_ngf'], stride=kargs['stride'], bias=kargs['bias'],
                                    conv_type=kargs['conv_type'])
        self.norm = NormLayer(kargs['norm'], kargs['new_ngf'])
        self.act = ActivationLayer(kargs['act'])

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


# condition vector & ce image feature fusion
class CFusion(nn.Module):
    def __init__(self, n_feats, act=True):
        super(CFusion, self).__init__()
        # fusion
        FusionBlock = [Conv(input_channels=n_feats, n_feats=n_feats, kernel_size=1, padding=0, act=act),
                       Conv(input_channels=n_feats, n_feats=n_feats, kernel_size=3, padding=1, act=act),
                       Conv(input_channels=n_feats, n_feats=n_feats, kernel_size=1, padding=0, act=act)]
        self.fusion = nn.Sequential(*FusionBlock)
    def forward(self, x, c):
        # x: [b, c, h, w]
        # c: [2c]
        c1 = c[:c.shape[0]//2].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        c2 = c[c.shape[0]//2:].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        xc = x*c1+c2
        xc_fusion = self.fusion(xc)
        return xc_fusion
        
# -------------------- Conditional Unet ------------------------------
class CUnet(nn.Module):
    def __init__(self, n_feats, n_resblock=4, kernel_size=3, padding=1, act=True):
        super(CUnet, self).__init__()

        # encoder1
        Encoder_first = [Conv(n_feats, n_feats*2, kernel_size, padding=padding, stride=2, act=act)]
        Encoder_first.extend([ResBlock(Conv, n_feats*2, kernel_size,padding=padding) for _ in range(n_resblock)])

        # encoder2
        Encoder_second = [Conv(n_feats*2, n_feats*4, kernel_size, padding=padding, stride=2, act=act)]
        Encoder_second.extend([ResBlock(Conv, n_feats*4, kernel_size, padding=padding) for _ in range(n_resblock)])
        
        # decoder2
        Decoder_second = [
            ResBlock(Conv, n_feats*4, kernel_size, padding=padding) for _ in range(n_resblock)]
        Decoder_second.append(Deconv(
            n_feats*4, n_feats*2, kernel_size=3, padding=1, output_padding=1, act=act))
        # decoder1
        Decoder_first = [
            ResBlock(Conv, n_feats*2, kernel_size, padding=padding) for _ in range(n_resblock)]
        Decoder_first.append(Deconv(
            n_feats*2, n_feats, kernel_size=3, padding=1, output_padding=1, act=act))

        self.encoder_first = nn.Sequential(*Encoder_first)
        self.encoder_second = nn.Sequential(*Encoder_second)
        self.decoder_second = nn.Sequential(*Decoder_second)
        self.decoder_first = nn.Sequential(*Decoder_first)
        self.cond_fusion = CFusion(n_feats*4)
        

    def forward(self, x, cond):
        n, c, h, w = x.shape
        
        # encoder
        e1 = self.encoder_first(x)
        e2 = self.encoder_second(e1)
        
        # condition fusion
        e2cond = self.cond_fusion(e2, cond)
        
        # decoder
        d2 = self.decoder_second(e2cond)
        e1, d2 = pad2same_size(e1, d2)
        d1 = self.decoder_first(e1+d2)
        x, d1 = pad2size(x, [h, w]), pad2size(d1, [h, w])
        y_in = x+d1
        return y_in
    