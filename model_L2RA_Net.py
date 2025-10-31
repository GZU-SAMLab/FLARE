import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # 第一层全连接层
        self.relu = nn.ReLU()  # 激活函数 ReLU
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # 输出层

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
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
    def __init__(self, num_classes, deep_supervision=False, Attention=True):
        super(UnetPlusPlusDecoder, self).__init__()
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        self.filters = [64, 128, 256, 512, 1024]
        self.Attention2_2 = SelfAttention(256, 2)
        self.Attention = Attention

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

        if self.deep_supervision:
            out_put4 = self.final_super_lesion_0_4(x_0_4_lesion)

            return x_4_0, out_put4
        else:
            return x_4_0, self.final_super_lesion_0_4(x_0_4_lesion)

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
            out_put4 = self.final_super_0_4(x_0_4)
            return x_4_0, out_put4
        else:
            return x_4_0, self.final_super_0_4(x_0_4)

class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class CL(nn.Module):
    def __init__(self):
        super(CL, self).__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, fused_features, text_features, label_group_text_features, classifier, class_label, temperature=0.07):
        # 归一化特征
        image_features = fused_features / fused_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # 确保两个张量的数据类型一致（例如都为 float32）
        image_features = image_features.to(dtype=torch.float32)
        text_features = text_features.to(dtype=torch.float32)

        # 正样本：计算点积相似度（得到每个样本的正样本分数，维度为batch_size）
        logits_positive = (image_features * text_features).sum(dim=1)  # 计算点积，得到相似度

        # 确保 class_label 是整数类型，并按 batch_size 获取对应类别的概率
        class_label = class_label.long()  # 强制转换为长整型
        probability_positive = torch.gather(classifier, 1, class_label.view(-1, 1))  # 获取每个样本的类别概率
        positive_scores = logits_positive * probability_positive.squeeze()  # 乘以正样本的概率

        # 创建掩码，排除 class_label 对应的列
        mask = torch.ones_like(classifier, dtype=torch.bool)
        mask[torch.arange(mask.size(0)), class_label] = False  # 将 class_label 对应的列置为 False
        negative_classifier_probabilities = torch.masked_select(classifier, mask).view(classifier.size(0), -1)

        # 与label_group_text_features 的形状相一致，先类别再batch
        probability_negative = negative_classifier_probabilities.t()

        # 计算负样本分数（对所有负样本的分数求和）,分别计算每个负样本类别 的分数
        negative_scores1 = 0
        for i, neg_text_features in enumerate(label_group_text_features):
            # 归一化负样本特征
            neg_text_features = neg_text_features / neg_text_features.norm(dim=1, keepdim=True)
            neg_text_features = neg_text_features.to(dtype=torch.float32) # [B,1024]

            # 对每个负样本计算点积相似度
            logits_negative = (image_features * neg_text_features).sum(dim=1)  # 计算点积，得到相似度[B]

            # 将负样本分数加到负样本分数总和
            negative_scores = logits_negative * probability_negative[i].squeeze()
            negative_scores1 += negative_scores

        return -torch.log(positive_scores / (negative_scores1 + positive_scores))  # 返回正负样本分数比

class global_SelfAttn(nn.Module):
    def __init__(self, channels=1024, H=14, W=14):
        super(global_SelfAttn, self).__init__()

        # 可学习的投影矩阵 K, V
        self.proj_matrix = nn.Parameter(torch.randn(channels, channels))

        # 可学习的查询矩阵 Q
        self.query_matrix = nn.Parameter(torch.randn(1, channels, H * W))  # [1, C, H*W] 可学习的查询矩阵

        self.scale = channels ** 0.5  # 缩放因子
        self.attn_drop = nn.Dropout(0.5)

    def forward(self, x):
        B, C, H, W = x.shape  # B: batch size, C: channels, H: height, W: width
        # print(f"x:{x.shape}")

        # 可学习查询矩阵 Q，通过expand扩展到 [B, H*W, C] 的形状
        q = self.query_matrix.expand(B, C, H * W).permute(0, 2, 1)  # [B, H*W, C] 扩展为批次大小 B

        # 计算 K 和 V，通过输入 x 和投影矩阵计算
        k = torch.matmul(x.view(B, C, H * W).permute(0, 2, 1), self.proj_matrix)  # [B, N, C]
        # print(f"q:{q.shape},k:{q.shape},v:{v.shape}")

        # 计算交叉注意力权重
        attn_scores = torch.bmm(q, k.permute(0, 2, 1))  # [B, N, N]
        attn_scores = attn_scores / self.scale  # 缩放
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, N, N]
        attn_weights = self.attn_drop(attn_weights)  # 应用 Dropout

        # 使用注意力权重加权 V
        output = torch.bmm(attn_weights, x.view(B, C, H * W).permute(0, 2, 1))  # [B, N, C]

        # 检查维度
        # print(f"Output shape before permute and view: {output.shape}")  # 打印输出的形状，检查是否为 [B, H*W, C]

        # 将输出重新排列为 [B, C, H, W]
        output = output.permute(0, 2, 1).view(B, C, H, W)  # 恢复形状 [B, C, H, W]

        return output

class MI_CroSelfAttn(nn.Module):
    def __init__(self):
        super(MI_CroSelfAttn, self).__init__()
        self.proj_matrix = nn.Parameter(torch.randn(1024, 1024))  # 可学习的投影矩阵，形状为 [C, C]
        self.scale = 1024 ** 0.5  # 缩放因子
        self.attn_drop = nn.Dropout(0.5)
    def forward(self, Q, K):
        """
        Q: 输入的第一个特征图，形状为 [B, C, H1, W1]
        K: 输入的第二个特征图，形状为 [B, C, H2, W2]
        """
        B, C, H, W = K.shape  # 获取 Q 的形状

        # 使用可学习的投影矩阵对 Q 和 K 进行投影
        q = torch.matmul(Q.view(B, C, H * W).permute(0, 2, 1), self.proj_matrix)  # [B, N, C]
        k = torch.matmul(K.view(B, C, H * W).permute(0, 2, 1), self.proj_matrix)  # [B, N, C]
        q1 = torch.matmul(K.view(B, C, H * W).permute(0, 2, 1), self.proj_matrix)  # [B, N, C]

        # 计算交叉注意力权重
        attn_scores = torch.bmm(q, k.permute(0, 2, 1))  # [B, N, N]
        attn_scores = attn_scores / self.scale  # 缩放
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, N, N]
        attn_weights = self.attn_drop(attn_weights)  # 应用 Dropout

        # 使用注意力权重加权 Q
        output1 = torch.bmm(attn_weights, K.view(B, C, H * W).permute(0, 2, 1))  # [B, N, C]
        output1 = output1.permute(0, 2, 1).view(B, C, H, W)  # 恢复形状 [B, C, H, W]

        # 计算交叉注意力权重
        attn_scores1 = torch.bmm(q1, k.permute(0, 2, 1))  # [B, N, N]
        attn_scores1 = attn_scores1 / self.scale  # 缩放
        attn_weights1 = F.softmax(attn_scores1, dim=-1)  # [B, N, N]
        attn_weights1 = self.attn_drop(attn_weights1)  # 应用 Dropout

        # 使用注意力权重加权 Q
        output2 = torch.bmm(attn_weights1, K.view(B, C, H * W).permute(0, 2, 1))  # [B, N, C]
        output2 = output2.permute(0, 2, 1).view(B, C, H, W)  # 恢复形状 [B, C, H, W]

        # 将两个输出沿通道维度拼接
        output = torch.cat([output1, output2], dim=1)  # [B, 2*C, H, W]

        return output

class ClassifierCroSelfAttn3(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ClassifierCroSelfAttn3, self).__init__()

        # 下采样模块：将大的特征下采样到与小的特征相同的大小
        self.downsample_large_1 = nn.Conv2d(2, 1024, kernel_size=3, stride=16, padding=1)  # large_feature_1 下采样
        self.downsample_large_2 = nn.Conv2d(2, 1024, kernel_size=3, stride=16, padding=1)  # large_feature_2 下采样

        # 特征融合后经过的卷积和激活
        self.conv1 = nn.Conv2d(input_dim * 4, output_dim, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(1024)
        self.relu = nn.ReLU(inplace=True)
        self.res_block = ResidualBlock(1024, 1024)  # 残差块
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 自适应池化
        self.mlp = MLP(input_dim=1024, hidden_dim=512, output_dim=5)
        self.contrastive_loss = CL()

        self.MI_CroSelfAttn = MI_CroSelfAttn()
        self.global_SelfAttn = global_SelfAttn()

    def forward(self, lesion_features, leaf_features, text_features=None, label_group_text_features=None, class_label=None, train=True):
        global_leaf, leaf_feature = leaf_features
        global_lesion, lesion_feature = lesion_features
        # print(f"small_feature: {small_feature.shape}, large_feature_1: {large_feature_1.shape}, large_feature_2: {large_feature_2.shape}")

        # 下采样大的特征图到与小的特征图一致的大小
        downsampled_large_1 = self.downsample_large_1(lesion_feature)  # [B, 1024, 14, 14]
        downsampled_large_2 = self.downsample_large_2(leaf_feature)  # [B, 1024, 14, 14]

        fused_feature = self.MI_CroSelfAttn(downsampled_large_1, downsampled_large_2)

        global_feature = self.global_SelfAttn(global_lesion)
        global_feature1 = self.global_SelfAttn(global_leaf)

        output = torch.cat([fused_feature, global_feature, global_feature1], dim=1)  # [B, 2*C, H, W]

        # 通过卷积、批标准化和 ReLU
        x = self.conv1(output)
        x = self.bn1(x)
        x = self.relu(x)

        # 通过残差块
        x = self.res_block(x)

        # 自适应池化，生成固定尺寸的特征图
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # 将特征图展平为一维向量
        classifier = self.mlp(x)

        if train == True:
            cur_loss = self.contrastive_loss(x, text_features, label_group_text_features, F.softmax(classifier, dim=1),
                                             class_label)
            return classifier, cur_loss
        else:
            return classifier


class ClassifierCroSelfAttn3_1(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ClassifierCroSelfAttn3_1, self).__init__()

        # 下采样模块：将大的特征下采样到与小的特征相同的大小
        self.downsample_large_1 = nn.Conv2d(2, 1024, kernel_size=3, stride=16, padding=1)  # large_feature_1 下采样
        self.downsample_large_2 = nn.Conv2d(2, 1024, kernel_size=3, stride=16, padding=1)  # large_feature_2 下采样

        # 特征融合后经过的卷积和激活
        self.conv1 = nn.Conv2d(input_dim * 4, output_dim, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(1024)
        self.relu = nn.ReLU(inplace=True)
        self.res_block = ResidualBlock(1024, 1024)  # 残差块
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 自适应池化
        self.mlp = MLP(input_dim=1024, hidden_dim=512, output_dim=5)
        self.contrastive_loss = CL()

        self.MI_CroSelfAttn = MI_CroSelfAttn()
        self.global_SelfAttn = global_SelfAttn()

        self.UnetPlusPlusDecoder = UnetPlusPlusDecoder(num_classes=2, Attention=True)
        self.UnetPlusPlusOrigin = UnetPlusPlusOrigin(num_classes=2)

    def forward(self, images, text_features=None, label_group_text_features=None, class_label=None, train=True):
        leaf_features = self.UnetPlusPlusOrigin(images)
        lesion_features = self.UnetPlusPlusDecoder(images)

        global_leaf, leaf_feature = leaf_features
        global_lesion, lesion_feature = lesion_features
        # print(f"small_feature: {small_feature.shape}, large_feature_1: {large_feature_1.shape}, large_feature_2: {large_feature_2.shape}")

        # 下采样大的特征图到与小的特征图一致的大小
        downsampled_large_1 = self.downsample_large_1(lesion_feature)  # [B, 1024, 14, 14]
        downsampled_large_2 = self.downsample_large_2(leaf_feature)  # [B, 1024, 14, 14]

        fused_feature = self.MI_CroSelfAttn(downsampled_large_1, downsampled_large_2)

        global_feature = self.global_SelfAttn(global_lesion)
        global_feature1 = self.global_SelfAttn(global_leaf)

        output = torch.cat([fused_feature, global_feature, global_feature1], dim=1)  # [B, 2*C, H, W]

        # 通过卷积、批标准化和 ReLU
        x = self.conv1(output)
        x = self.bn1(x)
        x = self.relu(x)

        # 通过残差块
        x = self.res_block(x)

        # 自适应池化，生成固定尺寸的特征图
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # 将特征图展平为一维向量
        classifier = self.mlp(x)

        if train == True:
            cur_loss = self.contrastive_loss(x, text_features, label_group_text_features, F.softmax(classifier, dim=1),
                                             class_label)
            return classifier, cur_loss
        else:
            return classifier

class ClassifierCroSelfAttn3_1Vision(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ClassifierCroSelfAttn3_1Vision, self).__init__()

        # 下采样模块：将大的特征下采样到与小的特征相同的大小
        self.downsample_large_1 = nn.Conv2d(2, 1024, kernel_size=3, stride=16, padding=1)  # large_feature_1 下采样
        self.downsample_large_2 = nn.Conv2d(2, 1024, kernel_size=3, stride=16, padding=1)  # large_feature_2 下采样

        # 特征融合后经过的卷积和激活
        self.conv1 = nn.Conv2d(input_dim * 4, output_dim, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(1024)
        self.relu = nn.ReLU(inplace=True)
        self.res_block = ResidualBlock(1024, 1024)  # 残差块
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 自适应池化
        self.mlp = MLP(input_dim=1024, hidden_dim=512, output_dim=5)
        self.contrastive_loss = CL()

        self.MI_CroSelfAttn = MI_CroSelfAttn()
        self.global_SelfAttn = global_SelfAttn()

        self.UnetPlusPlusDecoder = UnetPlusPlusDecoder(num_classes=2, Attention=True)
        self.UnetPlusPlusOrigin = UnetPlusPlusOrigin(num_classes=2)

    def forward(self, images, text_features=None, label_group_text_features=None, class_label=None, train=True):
        leaf_features = self.UnetPlusPlusOrigin(images)
        lesion_features = self.UnetPlusPlusDecoder(images)

        global_leaf, leaf_feature = leaf_features
        global_lesion, lesion_feature = lesion_features
        # print(f"small_feature: {small_feature.shape}, large_feature_1: {large_feature_1.shape}, large_feature_2: {large_feature_2.shape}")

        # 下采样大的特征图到与小的特征图一致的大小
        downsampled_large_1 = self.downsample_large_1(lesion_feature)  # [B, 1024, 14, 14]
        downsampled_large_2 = self.downsample_large_2(leaf_feature)  # [B, 1024, 14, 14]

        fused_feature = self.MI_CroSelfAttn(downsampled_large_1, downsampled_large_2)

        global_feature = self.global_SelfAttn(global_lesion)
        global_feature1 = self.global_SelfAttn(global_leaf)

        output = torch.cat([fused_feature, global_feature, global_feature1], dim=1)  # [B, 2*C, H, W]

        # 通过卷积、批标准化和 ReLU
        x = self.conv1(output)
        x = self.bn1(x)
        x = self.relu(x)

        # 通过残差块
        x = self.res_block(x)

        # 自适应池化，生成固定尺寸的特征图
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # 将特征图展平为一维向量
        classifier = self.mlp(x)

        if train == True:
            cur_loss = self.contrastive_loss(x, text_features, label_group_text_features, F.softmax(classifier, dim=1),
                                             class_label)
            return classifier, cur_loss
        else:
            return classifier, output, leaf_feature, lesion_feature
