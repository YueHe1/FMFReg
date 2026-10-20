import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones.voxelmorph.torch import layers
from models.backbones.layers import encoder
import torch.utils.checkpoint
from einops import repeat

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kSize=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=True,
                 mode='nearest'):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kSize, stride=stride, padding=padding, bias=bias),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=kSize, stride=stride, padding=padding, bias=bias),
            nn.InstanceNorm3d(out_channels)
        )

        self.downsample = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=bias),
            nn.InstanceNorm3d(out_channels)
        )

        self.leakyrelu = nn.LeakyReLU()

        self.up = nn.Upsample(scale_factor=(2, 2, 1), mode=mode)

    def forward(self, x):
        residual = self.downsample(x)

        out = self.conv1(x)
        out = self.conv2(out)
        out += residual

        out = self.leakyrelu(out)
        out = self.up(out)
        return out


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kSize=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=True):
        super().__init__()

        self.conv_pred = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // 2, kernel_size=kSize, stride=stride, padding=padding, bias=bias),
            nn.InstanceNorm3d(in_channels // 2),
            nn.LeakyReLU()
        )
        self.segmentation_head = nn.Conv3d(in_channels // 2, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv_pred(x)
        return self.segmentation_head(x)


class ConvInsBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, in_channels, out_channels, kernal_size=3, stride=1, padding=1, alpha=0.1):
        super().__init__()

        self.main = nn.Conv3d(in_channels, out_channels, kernal_size, stride, padding)
        self.norm = nn.InstanceNorm3d(out_channels)
        self.activation = nn.LeakyReLU(alpha)

    def forward(self, x):
        out = self.main(x)
        out = self.norm(out)
        out = self.activation(out)
        return out


class Encoder(nn.Module):
    """
    Main model
    """

    def __init__(self, in_channel=1, first_out_channel=16):
        super(Encoder, self).__init__()

        c = first_out_channel
        self.conv0 = ConvInsBlock(in_channel, c, 3, 1)

        self.conv1 = nn.Sequential(
            nn.Conv3d(c, 2 * c, kernel_size=3, stride=2, padding=1),  # 80
            ResBlock(2 * c)
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(2 * c, 4 * c, kernel_size=3, stride=2, padding=1),  # 40
            ResBlock(4 * c)
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(4 * c, 8 * c, kernel_size=3, stride=2, padding=1),  # 20
            ResBlock(8 * c)
        )

    def forward(self, x):
        out0 = self.conv0(x)  # 1
        out1 = self.conv1(out0)  # 1/2
        out2 = self.conv2(out1)  # 1/4
        out3 = self.conv3(out2)  # 1/8

        return [out0, out1, out2, out3]


class dispWarp(nn.Module):

    def __init__(self, in_cs, ks=1, is_int=1):

        super(dispWarp, self).__init__()

        self.is_int = is_int

        self.disp_field_fea = nn.Sequential(
            nn.Conv3d(2 * in_cs, 2 * in_cs, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(2 * in_cs, (ks * 2 + 1) ** 3, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

        self.get_flow = nn.Conv3d((ks * 2 + 1) ** 3, 3, 3, 1, 1)
        self.up_tri = torch.nn.Upsample(scale_factor=(2, 2, 1), mode='trilinear', align_corners=True)

    def disp_field(self, x, y):

        feas = self.disp_field_fea(torch.cat((y + x, y - x), dim=1))
        flow = self.get_flow(feas)

        return flow

    def forward(self, x, y, transformer, up_flow, integrate):

        if up_flow is not None:
            x = transformer(x, up_flow)

        flow = self.disp_field(x, y)
        preint_flow = flow
        if self.is_int:
            flow = integrate(flow)

        if up_flow is not None:
            flow = flow + transformer(up_flow, flow)

        up_flow = self.up_tri(flow)  # * 2
        up_flow = up_flow * torch.tensor((2, 2, 1)).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(
            (up_flow.shape[0], 1, up_flow.shape[2], up_flow.shape[3], up_flow.shape[4])).cuda()

        return preint_flow, flow, up_flow


class ResBlock(nn.Module):
    """
    VoxRes module
    """

    def __init__(self, channel, alpha=0.1):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.InstanceNorm3d(channel),
            nn.LeakyReLU(alpha),
            nn.Conv3d(channel, channel, kernel_size=3, padding=1)
        )
        self.actout = nn.Sequential(
            nn.InstanceNorm3d(channel),
            nn.LeakyReLU(alpha),
        )

    def forward(self, x):
        out = self.block(x) + x
        return self.actout(out)

class BilinearInterpolation(nn.Module):
    def __init__(self, scale_factor):
        super(BilinearInterpolation, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        batch_size, channels, height, width, depth = x.size()
        new_height = int(height * self.scale_factor[0])
        new_width = int(width * self.scale_factor[1])
        new_depth = int(depth * self.scale_factor[2])

        interpolated = F.interpolate(x, size=(new_height, new_width, depth), mode='trilinear')
        return interpolated


class Dinov3IR(nn.Module):

    def __init__(self,
        img_size='(128,128,16)', # (128,128,16) for ACDC
        start_channel='32',
        lk_size= '5',
        cv_ks = '1',
        layer_num = '4',
        is_int = '1',
        is_hadamard = '1',
    ):
        super(Dinov3IR, self).__init__()

        self.img_size = eval(img_size)
        self.start_channel = int(start_channel)
        self.lk_size = int(lk_size)
        self.cv_ks = int(cv_ks)
        self.layer_num = int(layer_num)
        self.is_int = int(is_int)
        self.is_hadamard = int(is_hadamard)

        self.interpolate = BilinearInterpolation(scale_factor=(0.5, 0.5, 1))
        self.num_classes = 4

        print(
            "img_size: {}, start_channel: {}, lk_size: {}, cv_ks: {}, is_int: {}, is_hadamard: {}, layer_num: {}".format( \
                self.img_size, self.start_channel, self.lk_size, self.cv_ks, self.is_int, self.is_hadamard,
                self.layer_num + 1))

        N_s = self.start_channel
        self.simple_encoder = nn.Sequential(
            encoder(769, N_s, 3, 1, 1),
            encoder(N_s, 2 * N_s, 3, 1, 1),
            encoder(2 * N_s, N_s, 3, 1, 1),
        )

        ss = self.img_size
        self.transformers = nn.ModuleList(
            [layers.SpatialTransformer((ss[0] // 2 ** i, ss[1] // 2 ** i, ss[2])) for i in range(5)])
        self.integrates = nn.ModuleList([layers.VecInt((ss[0] // 2 ** i, ss[1] // 2 ** i, ss[2]), 7) for i in range(5)])
        self.disp_warp_4 = dispWarp(N_s, self.cv_ks, self.is_int)
        self.disp_warp_3 = dispWarp(N_s, self.cv_ks, self.is_int)
        self.disp_warp_2 = dispWarp(N_s, self.cv_ks, self.is_int)
        self.disp_warp_1 = dispWarp(N_s, self.cv_ks, self.is_int)
        self.disp_warp_0 = dispWarp(N_s, self.cv_ks, self.is_int)


        self.do_ds = False
        self.decoder5 = BasicBlock(in_channels=256, out_channels=128)

        self.decoder4 = BasicBlock(in_channels=128, out_channels=64)

        self.decoder3 = BasicBlock(in_channels=64, out_channels=32)

        self.decoder2 = BasicBlock(in_channels=32, out_channels=16)

        self.out1 = SegmentationHead(in_channels=16, out_channels=self.num_classes)
        if self.do_ds:
            self.out3 = SegmentationHead(in_channels=64, out_channels=self.num_classes)
            self.out2 = SegmentationHead(in_channels=32, out_channels=self.num_classes)

    def forward(self, x, y, source_dino, target_dino, registration=False):

        source = torch.cat([x, source_dino], 1)
        target = torch.cat([y, target_dino], 1)

        feas = self.simple_encoder(torch.cat([source, target], 0))
        x_feas, y_feas = torch.chunk(feas, 2, dim=0)

        x_0, y_0 = x_feas, y_feas

        x_1 = self.interpolate(x_0)
        y_1 = self.interpolate(y_0)

        x_2 = self.interpolate(x_1)
        y_2 = self.interpolate(y_1)

        x_3 = self.interpolate(x_2)
        y_3 = self.interpolate(y_2)

        x_4 = self.interpolate(x_3)
        y_4 = self.interpolate(y_3)

        int_flow_4, pos_flow_4, up_flow_4 = self.disp_warp_4(x_4, y_4, self.transformers[4], None,
                                                             self.integrates[4])
        int_flow_3, pos_flow_3, up_flow_3 = self.disp_warp_3(x_3, y_3, self.transformers[3], up_flow_4,
                                                             self.integrates[3])
        int_flow_2, pos_flow_2, up_flow_2 = self.disp_warp_2(x_2, y_2, self.transformers[2], up_flow_3,
                                                             self.integrates[2])
        int_flow_1, pos_flow_1, up_flow_1 = self.disp_warp_1(x_1, y_1, self.transformers[1], up_flow_2,
                                                             self.integrates[1])
        int_flow_0, pos_flow_0, _ = self.disp_warp_0(x_0, y_0, self.transformers[0], up_flow_1, self.integrates[0])

        int_flows = [int_flow_0, int_flow_1, int_flow_2, int_flow_3, int_flow_4]
        pos_flows = [pos_flow_0, pos_flow_1, pos_flow_2, pos_flow_3, pos_flow_4]

        if not registration:
            return int_flows, pos_flows#, source_dino, target_dino
        else:
            return pos_flows[0]