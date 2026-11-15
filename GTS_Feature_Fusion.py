import torch
import torch.nn as nn


def channel_shuffle(x, groups):
    """Shuffle channels across groups."""
    batch_size, num_channels, height, width = x.size()
    if num_channels % groups != 0:
        raise ValueError("The number of channels must be divisible by groups.")

    channels_per_group = num_channels // groups
    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    return x.view(batch_size, -1, height, width)


class GTS_Feature_Fusion(nn.Module):
    """Gated temporal-spatial feature fusion module."""

    def __init__(self, channels, dropoutP=0.5):
        super(GTS_Feature_Fusion, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.gate = nn.Sequential(
            nn.Linear(channels * 2, channels // 2),
            nn.ELU(),
            nn.Linear(channels // 2, channels * 2),
            nn.Sigmoid(),
            nn.Dropout(dropoutP),
        )
        self.interact = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ELU(),
        )

    def forward(self, x_time, x_space):
        b, c, _, _ = x_time.size()
        info = torch.cat(
            [self.global_pool(x_time).view(b, c), self.global_pool(x_space).view(b, c)],
            dim=1,
        )
        gates = self.gate(info).view(b, 2 * c, 1, 1)
        g_t, g_s = gates.split(c, dim=1)

        xt_c = x_time * (0.5 + 0.5 * g_t)
        xs_c = x_space * (0.5 + 0.5 * g_s)
        refined = self.interact(torch.cat((xt_c, xs_c), dim=1))

        # Interleave the temporal, interaction, and spatial feature groups.
        out = torch.cat((xt_c, refined, xs_c), dim=1)
        return channel_shuffle(out, groups=3)
