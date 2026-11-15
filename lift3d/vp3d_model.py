"""Our own re-implementation of VideoPose3D's TemporalModel (Pavllo et al., CVPR 2019),
written from the published architecture so we don't execute third-party code.

Parameter names match the official implementation exactly (expand_conv, expand_bn,
layers_conv.*, layers_bn.*, shrink), so the official checkpoint's `model_pos` state
dict loads with load_state_dict(strict=True). Symmetric (non-causal) and causal modes.
Reference: github.com/facebookresearch/VideoPose3D, common/model.py (CC BY-NC 4.0).
"""
import torch.nn as nn


class TemporalModel(nn.Module):
    def __init__(self, num_joints_in=17, in_features=2, num_joints_out=17,
                 filter_widths=(3, 3, 3, 3, 3), causal=False, dropout=0.25, channels=1024):
        super().__init__()
        assert all(fw % 2 == 1 for fw in filter_widths), "only odd filter widths"
        self.num_joints_in, self.in_features, self.num_joints_out = num_joints_in, in_features, num_joints_out
        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)
        self.expand_conv = nn.Conv1d(num_joints_in * in_features, channels, filter_widths[0], bias=False)
        self.expand_bn = nn.BatchNorm1d(channels, momentum=0.1)

        self.pad = [filter_widths[0] // 2]
        self.causal_shift = [filter_widths[0] // 2 if causal else 0]
        convs, bns = [], []
        dilation = filter_widths[0]
        for fw in filter_widths[1:]:
            self.pad.append((fw - 1) * dilation // 2)
            self.causal_shift.append(fw // 2 * dilation if causal else 0)
            convs += [nn.Conv1d(channels, channels, fw, dilation=dilation, bias=False),
                      nn.Conv1d(channels, channels, 1, bias=False)]
            bns += [nn.BatchNorm1d(channels, momentum=0.1), nn.BatchNorm1d(channels, momentum=0.1)]
            dilation *= fw
        self.layers_conv = nn.ModuleList(convs)
        self.layers_bn = nn.ModuleList(bns)
        self.shrink = nn.Conv1d(channels, num_joints_out * 3, 1)

    def receptive_field(self):
        return 1 + 2 * sum(self.pad)

    def forward(self, x):                                     # x: (B, T, J, 2)
        B, T = x.shape[:2]
        x = x.reshape(B, T, -1).permute(0, 2, 1)              # (B, J*2, T)
        x = self.drop(self.relu(self.expand_bn(self.expand_conv(x))))
        for i in range(len(self.pad) - 1):
            pad, shift = self.pad[i + 1], self.causal_shift[i + 1]
            res = x[:, :, pad + shift: x.shape[2] - pad + shift]
            x = self.drop(self.relu(self.layers_bn[2 * i](self.layers_conv[2 * i](x))))
            x = res + self.drop(self.relu(self.layers_bn[2 * i + 1](self.layers_conv[2 * i + 1](x))))
        x = self.shrink(x)                                    # (B, J*3, T - receptive_field + 1)
        return x.permute(0, 2, 1).reshape(B, -1, self.num_joints_out, 3)
