import torch
import torch.nn as nn

class ContinusParalleConv(nn.Module):
    # 一个连续的卷积模块，包含BatchNorm 在前 和 在后 两种模式
    def __init__(self, in_channels, out_channels, pre_Batch_Norm=True):
        super(ContinusParalleConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if pre_Batch_Norm:
            self.Conv_forward = nn.Sequential(
                nn.BatchNorm2d(self.in_channels),
                nn.ReLU(),
                nn.Conv2d(self.in_channels, self.out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1))

        else:
            self.Conv_forward = nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1),
                nn.BatchNorm2d(self.out_channels),
                nn.ReLU())

    def forward(self, x):
        x = self.Conv_forward(x)
        return x
class FFParser(nn.Module):
    def __init__(self, dim, h=128, w=65):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(dim, h, w // 2 + 1, 2, dtype=torch.float32) * 0.02)
        self.w = w
        self.h = h

    def forward(self, x, spatial_size=None):
        B, C, H, W = x.shape
        assert H == W, "height and width are not equal"
        if spatial_size is None:
            a = b = H
        else:
            a, b = spatial_size

        x = x.to(torch.float32)
        x = torch.fft.rfft2(x, dim=(2, 3), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight).to(x.device)
        x = x * weight
        x = torch.fft.irfft2(x, s=(H, W), dim=(2, 3), norm='ortho')

        x = x.reshape(B, C, H, W)

        return x


class SelfAttention(nn.Module):
    """自注意力模块"""

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k1):
        B, C, H, W = q.shape

        # 对Q和K进行FFT变换
        q_fft = FFParser(C, H, W)(q)
        k_fft = FFParser(C, H, W)(k1)

        # 将输入展平
        q_fft = q_fft.view(B, C, H * W).permute(0, 2, 1)  # B, N, C
        k_fft = k_fft.view(B, C, H * W).permute(0, 2, 1)  # B, N, C
        v = k1.view(B, C, H * W).permute(0, 2, 1)  # B, N, C
        complex_weight = nn.Parameter(torch.randn(C, H, H, 2, dtype=torch.float32) * 0.02)

        # 计算注意力得分
        attn = (q_fft * torch.conj(k_fft)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 计算注意力加权输出
        v_fft = torch.fft.fft(v, dim=1)
        x_fft = (attn * v_fft)

        # 对加权输出进行IFFT变换
        x_ifft = torch.fft.ifft(x_fft, dim=1).real

        # 调整形状
        x = x_ifft.transpose(1, 2).reshape(B, -1, self.dim)
        x = x.view(B, H * W, C).permute(0, 2, 1).view(B, C, H, W)

        return x

class UnetPlusPlusDecoder(nn.Module):
    def __init__(self, num_classes, deep_supervision=True, Attention=True, au_branch=True):
        super(UnetPlusPlusDecoder, self).__init__()
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        self.filters = [64, 128, 256, 512, 1024]
        self.Attention2_2 = SelfAttention(256, 2)
        self.Attention = Attention
        self.au_branch = au_branch

        self.stage_0 = ContinusParalleConv(3, 64, pre_Batch_Norm=False)
        self.stage_1 = ContinusParalleConv(64, 128, pre_Batch_Norm=False)
        self.stage_2 = ContinusParalleConv(128, 256, pre_Batch_Norm=False)
        self.stage_3 = ContinusParalleConv(256, 512, pre_Batch_Norm=False)
        self.stage_4 = ContinusParalleConv(512, 1024, pre_Batch_Norm=False)

        self.pool = nn.MaxPool2d(2)

        # 病灶分割部分
        self.CONV_lesion_3_1 = ContinusParalleConv(512 * 2, 512, pre_Batch_Norm=True)

        self.CONV_lesion_2_2 = ContinusParalleConv(256 * 3, 256, pre_Batch_Norm=True)
        self.CONV_lesion_2_1 = ContinusParalleConv(256 * 2, 256, pre_Batch_Norm=True)

        self.CONV_lesion_1_1 = ContinusParalleConv(128 * 2, 128, pre_Batch_Norm=True)
        self.CONV_lesion_1_2 = ContinusParalleConv(128 * 3, 128, pre_Batch_Norm=True)
        self.CONV_lesion_1_3 = ContinusParalleConv(128 * 4, 128, pre_Batch_Norm=True)

        self.CONV_lesion_0_1 = ContinusParalleConv(64 * 2, 64, pre_Batch_Norm=True)
        self.CONV_lesion_0_2 = ContinusParalleConv(64 * 3, 64, pre_Batch_Norm=True)
        self.CONV_lesion_0_3 = ContinusParalleConv(64 * 4, 64, pre_Batch_Norm=True)
        # self.CONV_lesion_0_4 = ContinusParalleConv(64 * 5, 64, pre_Batch_Norm=True)
        if not Attention:
            self.CONV_lesion_0_4 = ContinusParalleConv(64*5, 64, pre_Batch_Norm = True)
        else:
            self.CONV_lesion_0_4 = ContinusParalleConv(64 * 6, 64, pre_Batch_Norm=True)

        self.upsample_lesion_3_1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1)

        self.upsample_lesion_2_1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.upsample_lesion_2_2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1)

        self.upsample_lesion_1_1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.upsample_lesion_1_2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.upsample_lesion_1_3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)

        self.upsample_lesion_0_1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.upsample_lesion_0_2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.upsample_lesion_0_3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.upsample_lesion_0_4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)

        # 分割头
        self.final_super_lesion_0_1 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, self.num_classes, 3, padding=1),
        )
        self.final_super_lesion_0_2 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, self.num_classes, 3, padding=1),
        )
        self.final_super_lesion_0_3 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, self.num_classes, 3, padding=1),
        )
        self.final_super_lesion_0_4 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, self.num_classes, 3, padding=1),
        )

        if self.au_branch==True:

            # 叶片分割部分
            self.CONV_leaf_3_1 = ContinusParalleConv(512 * 2, 512, pre_Batch_Norm=True)

            self.CONV_leaf_2_2 = ContinusParalleConv(256 * 3, 256, pre_Batch_Norm=True)
            self.CONV_leaf_2_1 = ContinusParalleConv(256 * 2, 256, pre_Batch_Norm=True)

            self.CONV_leaf_1_1 = ContinusParalleConv(128 * 2, 128, pre_Batch_Norm=True)
            self.CONV_leaf_1_2 = ContinusParalleConv(128 * 3, 128, pre_Batch_Norm=True)
            self.CONV_leaf_1_3 = ContinusParalleConv(128 * 4, 128, pre_Batch_Norm=True)

            self.CONV_leaf_0_1 = ContinusParalleConv(64 * 2, 64, pre_Batch_Norm=True)
            self.CONV_leaf_0_2 = ContinusParalleConv(64 * 3, 64, pre_Batch_Norm=True)
            self.CONV_leaf_0_3 = ContinusParalleConv(64 * 4, 64, pre_Batch_Norm=True)
            # self.CONV_leaf_0_4 = ContinusParalleConv(64 * 5, 64, pre_Batch_Norm=True)
            self.CONV_leaf_0_4 = ContinusParalleConv(64*5, 64, pre_Batch_Norm = True)

            self.upsample_leaf_3_1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2,
                                                        padding=1)

            self.upsample_leaf_2_1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2,
                                                        padding=1)
            self.upsample_leaf_2_2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2,
                                                        padding=1)

            self.upsample_leaf_1_1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2,
                                                        padding=1)
            self.upsample_leaf_1_2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2,
                                                        padding=1)
            self.upsample_leaf_1_3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2,
                                                        padding=1)

            self.upsample_leaf_0_1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2,
                                                        padding=1)
            self.upsample_leaf_0_2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2,
                                                        padding=1)
            self.upsample_leaf_0_3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2,
                                                        padding=1)
            self.upsample_leaf_0_4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2,
                                                        padding=1)

            # 分割头
            self.final_super_leaf_0_1 = nn.Sequential(
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, self.num_classes, 3, padding=1),
            )
            self.final_super_leaf_0_2 = nn.Sequential(
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, self.num_classes, 3, padding=1),
            )
            self.final_super_leaf_0_3 = nn.Sequential(
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, self.num_classes, 3, padding=1),
            )
            self.final_super_leaf_0_4 = nn.Sequential(
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, self.num_classes, 3, padding=1),
            )

    def forward(self, x):
        x_0_0 = self.stage_0(x)
        x_1_0 = self.stage_1(self.pool(x_0_0))
        x_2_0 = self.stage_2(self.pool(x_1_0))
        x_3_0 = self.stage_3(self.pool(x_2_0))
        x_4_0 = self.stage_4(self.pool(x_3_0))

        # 病灶部分
        x_0_1_lesion = torch.cat([self.upsample_lesion_0_1(x_1_0), x_0_0], 1)
        x_0_1_lesion = self.CONV_lesion_0_1(x_0_1_lesion)

        x_1_1_lesion = torch.cat([self.upsample_lesion_1_1(x_2_0), x_1_0], 1)
        x_1_1_lesion = self.CONV_lesion_1_1(x_1_1_lesion)

        x_2_1_lesion = torch.cat([self.upsample_lesion_2_1(x_3_0), x_2_0], 1)
        x_2_1_lesion = self.CONV_lesion_2_1(x_2_1_lesion)

        x_3_1_lesion = torch.cat([self.upsample_lesion_3_1(x_4_0), x_3_0], 1)
        x_3_1_lesion = self.CONV_lesion_3_1(x_3_1_lesion)

        x_2_2_lesion = torch.cat([self.upsample_lesion_2_2(x_3_1_lesion), x_2_0, x_2_1_lesion], 1)
        x_2_2_lesion = self.CONV_lesion_2_2(x_2_2_lesion)

        x_1_2_lesion = torch.cat([self.upsample_lesion_1_2(x_2_1_lesion), x_1_0, x_1_1_lesion], 1)
        x_1_2_lesion = self.CONV_lesion_1_2(x_1_2_lesion)

        x_1_3_lesion = torch.cat([self.upsample_lesion_1_3(x_2_2_lesion), x_1_0, x_1_1_lesion, x_1_2_lesion], 1)
        x_1_3_lesion = self.CONV_lesion_1_3(x_1_3_lesion)

        x_0_2_lesion = torch.cat([self.upsample_lesion_0_2(x_1_1_lesion), x_0_0, x_0_1_lesion], 1)
        x_0_2_lesion = self.CONV_lesion_0_2(x_0_2_lesion)

        x_0_3_lesion = torch.cat([self.upsample_lesion_0_3(x_1_2_lesion), x_0_0, x_0_1_lesion, x_0_2_lesion], 1)
        x_0_3_lesion = self.CONV_lesion_0_3(x_0_3_lesion)

        # x_0_4_lesion = torch.cat([self.upsample_lesion_0_4(x_1_3_lesion), x_0_0, x_0_1_lesion, x_0_2_lesion, x_0_3_lesion], 1)
        if self.Attention:
            x_0_4_0_lesion = self.Attention2_2(self.upsample_lesion_0_4(x_1_3_lesion), x_0_0)
            x_0_4_0_lesion = self.Attention2_2(x_0_4_0_lesion, self.upsample_lesion_0_4(x_1_3_lesion))
            x_0_4_lesion = torch.cat(
                [self.upsample_lesion_0_4(x_1_3_lesion), x_0_0, x_0_1_lesion, x_0_2_lesion, x_0_3_lesion,
                 x_0_4_0_lesion], 1)
        else:
            x_0_4_lesion = torch.cat(
                [self.upsample_lesion_0_4(x_1_3_lesion), x_0_0, x_0_1_lesion, x_0_2_lesion, x_0_3_lesion], 1)
        x_0_4_lesion = self.CONV_lesion_0_4(x_0_4_lesion)

        if self.au_branch == True:

            # 叶片部分
            x_0_1_leaf = torch.cat([self.upsample_leaf_0_1(x_1_0), x_0_0], 1)
            x_0_1_leaf = self.CONV_leaf_0_1(x_0_1_leaf)

            x_1_1_leaf = torch.cat([self.upsample_leaf_1_1(x_2_0), x_1_0], 1)
            x_1_1_leaf = self.CONV_leaf_1_1(x_1_1_leaf)

            x_2_1_leaf = torch.cat([self.upsample_leaf_2_1(x_3_0), x_2_0], 1)
            x_2_1_leaf = self.CONV_leaf_2_1(x_2_1_leaf)

            x_3_1_leaf = torch.cat([self.upsample_leaf_3_1(x_4_0), x_3_0], 1)
            x_3_1_leaf = self.CONV_leaf_3_1(x_3_1_leaf)

            x_2_2_leaf = torch.cat([self.upsample_leaf_2_2(x_3_1_leaf), x_2_0, x_2_1_leaf], 1)
            x_2_2_leaf = self.CONV_leaf_2_2(x_2_2_leaf)

            x_1_2_leaf = torch.cat([self.upsample_leaf_1_2(x_2_1_leaf), x_1_0, x_1_1_leaf], 1)
            x_1_2_leaf = self.CONV_leaf_1_2(x_1_2_leaf)

            x_1_3_leaf = torch.cat([self.upsample_leaf_1_3(x_2_2_leaf), x_1_0, x_1_1_leaf, x_1_2_leaf], 1)
            x_1_3_leaf = self.CONV_leaf_1_3(x_1_3_leaf)

            x_0_2_leaf = torch.cat([self.upsample_leaf_0_2(x_1_1_leaf), x_0_0, x_0_1_leaf], 1)
            x_0_2_leaf = self.CONV_leaf_0_2(x_0_2_leaf)

            x_0_3_leaf = torch.cat([self.upsample_leaf_0_3(x_1_2_leaf), x_0_0, x_0_1_leaf, x_0_2_leaf], 1)
            x_0_3_leaf = self.CONV_leaf_0_3(x_0_3_leaf)

            x_0_4_leaf = torch.cat([self.upsample_leaf_0_4(x_1_3_leaf), x_0_0, x_0_1_leaf, x_0_2_leaf, x_0_3_leaf], 1)
            x_0_4_leaf = self.CONV_leaf_0_4(x_0_4_leaf)

        if self.au_branch == True:
            if self.deep_supervision:
                out_put1 = self.final_super_lesion_0_1(x_0_1_lesion)
                out_put2 = self.final_super_lesion_0_2(x_0_2_lesion)
                out_put3 = self.final_super_lesion_0_3(x_0_3_lesion)
                out_put4 = self.final_super_lesion_0_4(x_0_4_lesion)
                out_put_leaf1 = self.final_super_leaf_0_1(x_0_1_leaf)
                out_put_leaf2 = self.final_super_leaf_0_2(x_0_2_leaf)
                out_put_leaf3 = self.final_super_leaf_0_3(x_0_3_leaf)
                out_put_leaf4 = self.final_super_leaf_0_4(x_0_4_leaf)
                # print(
                #     f"small_feature:{x_4_0.shape}, large_feature_1:{out_put4.shape}, large_feature_2:{out_put_leaf4.shape}")

                return [out_put1, out_put2, out_put3, out_put4, out_put_leaf1, out_put_leaf2, out_put_leaf3, out_put_leaf4]
            else:
                return self.final_super_lesion_0_4(x_0_4_lesion), self.final_super_leaf_0_4(x_0_4_leaf)

        else:
            if self.deep_supervision:
                out_put1 = self.final_super_lesion_0_1(x_0_1_lesion)
                out_put2 = self.final_super_lesion_0_2(x_0_2_lesion)
                out_put3 = self.final_super_lesion_0_3(x_0_3_lesion)
                out_put4 = self.final_super_lesion_0_4(x_0_4_lesion)
                return [out_put1, out_put2, out_put3, out_put4]
            else:
                return self.final_super_lesion_0_4(x_0_4_lesion)

class UnetPlusPlusOrigin(nn.Module):
    def __init__(self, num_classes, deep_supervision=False):
        super(UnetPlusPlusOrigin, self).__init__()
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        self.filters = [64, 128, 256, 512, 1024]

        self.CONV3_1 = ContinusParalleConv(512 * 2, 512, pre_Batch_Norm=True)

        self.CONV2_2 = ContinusParalleConv(256 * 3, 256, pre_Batch_Norm=True)
        self.CONV2_1 = ContinusParalleConv(256 * 2, 256, pre_Batch_Norm=True)

        self.CONV1_1 = ContinusParalleConv(128 * 2, 128, pre_Batch_Norm=True)
        self.CONV1_2 = ContinusParalleConv(128 * 3, 128, pre_Batch_Norm=True)
        self.CONV1_3 = ContinusParalleConv(128 * 4, 128, pre_Batch_Norm=True)

        self.CONV0_1 = ContinusParalleConv(64 * 2, 64, pre_Batch_Norm=True)
        self.CONV0_2 = ContinusParalleConv(64 * 3, 64, pre_Batch_Norm=True)
        self.CONV0_3 = ContinusParalleConv(64 * 4, 64, pre_Batch_Norm=True)
        self.CONV0_4 = ContinusParalleConv(64 * 5, 64, pre_Batch_Norm=True)

        self.stage_0 = ContinusParalleConv(3, 64, pre_Batch_Norm=False)
        self.stage_1 = ContinusParalleConv(64, 128, pre_Batch_Norm=False)
        self.stage_2 = ContinusParalleConv(128, 256, pre_Batch_Norm=False)
        self.stage_3 = ContinusParalleConv(256, 512, pre_Batch_Norm=False)
        self.stage_4 = ContinusParalleConv(512, 1024, pre_Batch_Norm=False)

        self.pool = nn.MaxPool2d(2)

        self.upsample_3_1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1)

        self.upsample_2_1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.upsample_2_2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1)

        self.upsample_1_1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.upsample_1_2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.upsample_1_3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)

        self.upsample_0_1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.upsample_0_2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.upsample_0_3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.upsample_0_4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)

        # 分割头
        self.final_super_0_1 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, self.num_classes, 3, padding=1),
        )
        self.final_super_0_2 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, self.num_classes, 3, padding=1),
        )
        self.final_super_0_3 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, self.num_classes, 3, padding=1),
        )
        self.final_super_0_4 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, self.num_classes, 3, padding=1),
        )

    def forward(self, x):
        x_0_0 = self.stage_0(x)
        x_1_0 = self.stage_1(self.pool(x_0_0))
        x_2_0 = self.stage_2(self.pool(x_1_0))
        x_3_0 = self.stage_3(self.pool(x_2_0))
        x_4_0 = self.stage_4(self.pool(x_3_0))
        # print(f"x_0_0:{x_0_0.shape},x_1_0:{x_1_0.shape},x_2_0:{x_2_0.shape},x_3_0:{x_3_0.shape},x_4_0:{x_4_0.shape}")
        #
        x_0_1 = torch.cat([self.upsample_0_1(x_1_0), x_0_0], 1)
        x_0_1 = self.CONV0_1(x_0_1)

        x_1_1 = torch.cat([self.upsample_1_1(x_2_0), x_1_0], 1)
        x_1_1 = self.CONV1_1(x_1_1)

        x_2_1 = torch.cat([self.upsample_2_1(x_3_0), x_2_0], 1)
        x_2_1 = self.CONV2_1(x_2_1)

        x_3_1 = torch.cat([self.upsample_3_1(x_4_0), x_3_0], 1)
        x_3_1 = self.CONV3_1(x_3_1)

        x_2_2 = torch.cat([self.upsample_2_2(x_3_1), x_2_0, x_2_1], 1)
        x_2_2 = self.CONV2_2(x_2_2)

        x_1_2 = torch.cat([self.upsample_1_2(x_2_1), x_1_0, x_1_1], 1)
        x_1_2 = self.CONV1_2(x_1_2)

        x_1_3 = torch.cat([self.upsample_1_3(x_2_2), x_1_0, x_1_1, x_1_2], 1)
        x_1_3 = self.CONV1_3(x_1_3)

        x_0_2 = torch.cat([self.upsample_0_2(x_1_1), x_0_0, x_0_1], 1)
        x_0_2 = self.CONV0_2(x_0_2)

        x_0_3 = torch.cat([self.upsample_0_3(x_1_2), x_0_0, x_0_1, x_0_2], 1)
        x_0_3 = self.CONV0_3(x_0_3)

        x_0_4 = torch.cat([self.upsample_0_4(x_1_3), x_0_0, x_0_1, x_0_2, x_0_3], 1)
        x_0_4 = self.CONV0_4(x_0_4)

        if self.deep_supervision:
            out_put1 = self.final_super_0_1(x_0_1)
            out_put2 = self.final_super_0_2(x_0_2)
            out_put3 = self.final_super_0_3(x_0_3)
            out_put4 = self.final_super_0_4(x_0_4)
            return [out_put1, out_put2, out_put3, out_put4]
        else:
            return self.final_super_0_4(x_0_4)