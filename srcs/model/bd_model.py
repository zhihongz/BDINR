from srcs.model.bd_modules import Conv, ResBlock, MLP, CUnet
import torch.nn as nn
import torch
from srcs.model.bd_utils import PositionalEncoding

class BDNeRV_RC(nn.Module):
    # recursive frame reconstruction
    def __init__(self):
        super(BDNeRV_RC, self).__init__()
        # params
        n_colors = 3
        n_resblock = 4
        n_feats = 32
        kernel_size = 3
        padding = 1

        pos_b, pos_l = 1.25, 80  # position encoding params
        mlp_dim_list = [2*pos_l, 512, n_feats*4*2] # (160, 512, 256)
        mlp_act = 'gelu'

        # main body
        self.mainbody = CUnet(n_feats=n_feats, n_resblock=n_resblock,
                              kernel_size=kernel_size, padding=padding)

        # output block
        OutBlock = [ResBlock(Conv, n_feats=n_feats, kernel_size=kernel_size, padding=padding),
                    ResBlock(Conv, n_feats=n_feats, kernel_size=kernel_size, padding=padding),
                    Conv(input_channels=n_feats, n_feats=n_colors, kernel_size=kernel_size, padding=padding)]
        self.out = nn.Sequential(*OutBlock)

        # feature block
        FeatureBlock = [Conv(input_channels=n_colors, n_feats=n_feats, kernel_size=kernel_size, padding=padding, act=True),
                        ResBlock(Conv, n_feats=n_feats,
                                 kernel_size=kernel_size, padding=padding),
                        ResBlock(Conv, n_feats=n_feats,
                                 kernel_size=kernel_size, padding=padding),
                        ResBlock(Conv, n_feats=n_feats, kernel_size=kernel_size, padding=padding)]
        self.feature = nn.Sequential(*FeatureBlock)

        # concatenation fusion block
        CatFusion = [Conv(input_channels=n_feats*2, n_feats=n_feats, kernel_size=kernel_size, padding=padding, act=True),
                        ResBlock(Conv, n_feats=n_feats,
                                 kernel_size=kernel_size, padding=padding),
                        ResBlock(Conv, n_feats=n_feats,
                                 kernel_size=kernel_size, padding=padding),
                        ResBlock(Conv, n_feats=n_feats, kernel_size=kernel_size, padding=padding)]
        self.catfusion = nn.Sequential(*CatFusion)

        # position encoding
        self.pe_t = PositionalEncoding(pe_embed_b=pos_b, pe_embed_l=pos_l)

        # mlp
        self.embed_mlp = MLP(dim_list=mlp_dim_list, act=mlp_act)

    def forward(self, ce_blur, time_idx, ce_code):
        # time index: [frame_num,1]
        # t_embed
        t_pe_ = [self.pe_t(idx)*(2*code-1)
                 for idx, code in zip(time_idx, ce_code)]  # [frame_num*[pos_l*2,1]]
        t_pe = torch.cat(t_pe_, dim=0)  # [frame_num, pos_l*2]
        t_embed = self.embed_mlp(t_pe)  # [frame_num, n_feats*4*2]
        # t_manip = self.manip_mlp(t_pe)

        # ce_blur feature
        ce_feature = self.feature(ce_blur)  # [b, c, h, w]

        # main body
        output_list = []
        for k in range(len(time_idx)):
            if k==0:
                main_feature = ce_feature
            else:
                # since k=2, cat pre-feature with ce_feature as input feature
                cat_feature = torch.cat((feat_out_k, ce_feature),dim=1)
                main_feature = self.catfusion(cat_feature)
            feat_out_k = self.mainbody(main_feature, t_embed[k])
            output_k = self.out(feat_out_k)
            output_list.append(output_k)

        output = torch.stack(output_list, dim=1)

        return output
