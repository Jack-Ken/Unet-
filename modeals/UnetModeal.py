import torch.nn as nn
import torch as t


class DownsampleLayer(nn.Module):
    """
    定义下采样忽的网络层
    """
    def __init__(self, in_chanel, out_chanel):
        super(DownsampleLayer, self).__init__()
        self.conv_relu = nn.Sequential(
            nn.Conv2d(in_channels=in_chanel, out_channels=out_chanel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_chanel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_chanel, out_channels=out_chanel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_chanel),
            nn.ReLU(inplace=True)
        )
        self.dowmsample = nn.Sequential(
            # 利用步长实现向下池化
            # nn.MaxPool2d(2)
            nn.Conv2d(in_channels=out_chanel, out_channels=out_chanel, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_chanel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """

        :param
            x: 输入的图像数据
        :return:
            out:输出到深层
            out_2:输出到下一层
        """
        out = self.conv_relu(x)
        out_2 = self.dowmsample(out)
        return out, out_2


class UpsampleLayer(nn.Module):
    """
    定义上采样的网络层
    """
    def __init__(self, in_chanel, out_chanel):
        super(UpsampleLayer, self).__init__()
        self.up_conv_relu = nn.Sequential(
            nn.Conv2d(in_channels=in_chanel, out_channels=out_chanel*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_chanel * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_chanel*2, out_channels=out_chanel*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_chanel * 2),
            nn.ReLU(inplace=True)
        )
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels=out_chanel*2, out_channels=out_chanel, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_chanel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, out):
        """

        :param
            x:图片数据
            out: 对应上采样层的浅层特征
        :return:
             cat_out：上采样层的千层特征和下采样层的抽象特征拼接
        """
        x = self.up_conv_relu(x)
        x = self.upsample(x)
        cat_out = t.cat([x, out], dim=1)
        return cat_out


class Unet(nn.Module):
    """
    定义Unet网络结构
    """
    def __init__(self):
        super(Unet, self).__init__()
        # 定义下采样层的网络层结构
        self.d1 = DownsampleLayer(1, 64)
        self.d2 = DownsampleLayer(64, 128)
        self.d3 = DownsampleLayer(128, 256)
        self.d4 = DownsampleLayer(256, 512)

        #定义上采样层的网络层结构
        self.u1 = UpsampleLayer(512, 512)
        self.u2 = UpsampleLayer(1024, 256)
        self.u3 = UpsampleLayer(512, 128)
        self.u4 = UpsampleLayer(256, 64)
        # 输出
        self.o = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=1),
        )

    def forward(self, x):
        out_1, out1 = self.d1(x)
        out_2, out2 = self.d2(out1)
        out_3, out3 = self.d3(out2)
        out_4, out4 = self.d4(out3)

        out5 = self.u1(out4, out_4)
        out6 = self.u2(out5, out_3)
        out7 = self.u3(out6, out_2)
        out8 = self.u4(out7, out_1)

        out = self.o(out8)
        return out



