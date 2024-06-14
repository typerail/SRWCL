import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Noise_Layer import Identity, ScreenShooting

from modules import infomax_gan_base
from modules.layers import SNConv2d, SNLinear
from modules.resblocks import DBlockOptimized, DBlock, GBlock

class ConvBNRelu(nn.Module):
    def __init__(self, channels_in, channels_out, stride=1):
        super(ConvBNRelu, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 3, stride, padding=1),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)


class SingleConv(nn.Module):
    def __init__(self, inchannel, outchannel, s):
        super(SingleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=s, padding=1, bias=True),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, s):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=s, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if s != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=s, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Discriminator(nn.Module):
    def __init__(self, num_channels):
        super().__init__()

        self.activation = nn.ReLU(True)
        self.local_feat_blocks = nn.Sequential(
            DBlockOptimized(3, num_channels),
            DBlock(num_channels, num_channels, downsample=True),
            DBlock(num_channels, num_channels, downsample=False)
        )

        self.global_feat_blocks = nn.Sequential(
            DBlock(num_channels, num_channels, downsample=False)
        )

        self.linear = SNLinear(num_channels, 1)
        nn.init.xavier_uniform_(self.linear.weight.data, 1.0)

        self.local_nrkhs_a = SNConv2d(num_channels, num_channels, 1, 1, 0)
        self.local_nrkhs_b = SNConv2d(num_channels, 64, 1, 1, 0)
        self.local_nrkhs_sc = SNConv2d(num_channels, 64, 1, 1, 0)

        nn.init.xavier_uniform_(self.local_nrkhs_a.weight.data, 1.0)
        nn.init.xavier_uniform_(self.local_nrkhs_b.weight.data, 1.0)
        nn.init.xavier_uniform_(self.local_nrkhs_sc.weight.data, 1.0)

        # Critic network layers for global features
        self.global_nrkhs_a = SNLinear(num_channels, num_channels)
        self.global_nrkhs_b = SNLinear(num_channels, 64)
        self.global_nrkhs_sc = SNLinear(num_channels, 64)

        nn.init.xavier_uniform_(self.global_nrkhs_a.weight.data, 1.0)
        nn.init.xavier_uniform_(self.global_nrkhs_b.weight.data, 1.0)
        nn.init.xavier_uniform_(self.global_nrkhs_sc.weight.data, 1.0)


        # super(Discriminator, self).__init__()
        # self.discriminator = nn.Sequential(
        #     ConvBNRelu(3, num_channels),
        #     ConvBNRelu(num_channels, num_channels),
        #     ConvBNRelu(num_channels, num_channels),
        #     nn.AdaptiveAvgPool2d(output_size=(1, 1))
        # )
        # self.linear = nn.Linear(num_channels, 1)

    def forward(self, x):
        h = x
        local_feat = self.local_feat_blocks(h)  # (N, C, H, W)
        global_feat = self.global_feat_blocks(local_feat)
        global_feat = self.activation(global_feat)
        global_feat = torch.sum(global_feat, dim=(2, 3))
        # GAN task output
        output = self.linear(global_feat)

        return output, local_feat, global_feat
        # return output

        # D = self.discriminator(x)
        # D.squeeze_(3).squeeze_(2)
        # D = self.linear(D)
        # return D

#test
class TestConv(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(TestConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class Testup_conv(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(Testup_conv, self).__init__()
        self.up = nn.Sequential(
            #nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(inchannel, inchannel, kernel_size=2, stride=2),
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


#test

class DoubleConv(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            #nn.ConvTranspose2d(inchannel, inchannel, kernel_size=2, stride=2),
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class up_expanded_message(nn.Module):
    def __init__(self, inchannels, flag):
        super(up_expanded_message, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(inchannels, inchannels, kernel_size=flag, stride=flag)
        )
    def forward(self, x):
        x = self.up(x)
        return x



class U_Net_Encoder_Diffusion(nn.Module):
    def __init__(self, inchannel=3, outchannel=3):

        super(U_Net_Encoder_Diffusion, self).__init__()



        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Globalpool = nn.MaxPool2d(kernel_size=4, stride=4)

        #test
        # self.testConv1 = TestConv(inchannel, 16)
        # self.testMaxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.testConv2 = TestConv(16, 32)
        # self.testConv3 = TestConv(32, 64)
        # self.testGlobalpool = nn.MaxPool2d(kernel_size=4, stride=4)
        # self.testlinear = nn.Linear(30, 256)
        # self.testConv_message = TestConv(1, 64)
        # self.testUp4 = Testup_conv(64 * 3, 64)
        # self.testConv7 = TestConv(64 * 3, 64)
        #test

        self.Conv1 = DoubleConv(inchannel, 16)
        self.Conv2 = DoubleConv(16, 32)
        self.Conv3 = DoubleConv(32, 64)

        self.Up4 = up_conv(64 * 3, 64)
        self.Conv7 = DoubleConv(64 * 3, 64)

        self.Up3 = up_conv(64, 32)
        self.Conv8 = DoubleConv(32 * 2 + 64, 32)

        self.Up2 = up_conv(32, 16)
        self.Conv9 = DoubleConv(16 * 2 + 64, 16)

        self.Conv_1x1 = nn.Conv2d(16, outchannel, kernel_size=1, stride=1, padding=0)
        self.linear = nn.Linear(30, 256)
        self.Conv_message = DoubleConv(1, 64)
        # self.up_message1 = up_expanded_message(1, 2)
        # self.up_message2 = up_expanded_message(1, 4)
        # self.up_message3 = up_expanded_message(1, 8)


    def forward(self, x, watermark):
        # p1 = self.testConv1(x)
        # p2 = self.testMaxpool(p1)
        # p2 = self.testConv2(p2)
        # p3 = self.testMaxpool(p2)
        # p3 = self.testConv3(p3)
        # p4 = self.testMaxpool(p3)
        # p6 = self.testGlobalpool(p4)
        # p7 = p6.repeat(1, 1, 4, 4)
        # message = self.testlinear(watermark)
        # message = message.view(-1, 1, 16, 16)
        # message = self.testConv_message(message)
        # p4 = torch.cat((p4, p7, message), dim=1)
        #
        # f4 = self.testUp4(p4)
        #
        # message = self.testlinear(watermark)
        # message = message.view(-1, 1, 16, 16)
        # message = torch.nn.functional.interpolate(message, size=(f4.shape[2], f4.shape[3]),
        #                                                    mode='bilinear')
        # message = self.testConv_message(message)
        # f4 = torch.cat((p3, f4, message), dim=1)
        # f4 = self.testConv7(f4)


        # print(message)

        x1 = self.Conv1(x)
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        x4 = self.Maxpool(x3)
        x6 = self.Globalpool(x4)
        x7 = x6.repeat(1, 1, 4, 4)
        expanded_message = self.linear(watermark)
        expanded_message = expanded_message.view(-1, 1, 16, 16)
        expanded_message = self.Conv_message(expanded_message)
        x4 = torch.cat((x4, x7, expanded_message), dim=1)

        d4 = self.Up4(x4)
        expanded_message = self.linear(watermark)
        expanded_message = expanded_message.view(-1, 1, 16, 16)
        # expanded_message = self.up_message1(expanded_message)
        expanded_message = torch.nn.functional.interpolate(expanded_message, size=(d4.shape[2], d4.shape[3]),
                                                           mode='bilinear')
        expanded_message = self.Conv_message(expanded_message)
        d4 = torch.cat((x3, d4, expanded_message), dim=1)
        d4 = self.Conv7(d4)

        d3 = self.Up3(d4)
        expanded_message = self.linear(watermark)
        expanded_message = expanded_message.view(-1, 1, 16, 16)
        #expanded_message = self.up_message2(expanded_message)
        expanded_message = torch.nn.functional.interpolate(expanded_message, size=(d3.shape[2], d3.shape[3]),
                                                           mode='bilinear')
        expanded_message = self.Conv_message(expanded_message)
        d3 = torch.cat((x2, d3, expanded_message), dim=1)
        d3 = self.Conv8(d3)

        d2 = self.Up2(d3)
        expanded_message = self.linear(watermark)
        expanded_message = expanded_message.view(-1, 1, 16, 16)
        # expanded_message = self.up_message3(expanded_message)
        expanded_message = torch.nn.functional.interpolate(expanded_message, size=(d2.shape[2], d2.shape[3]),
                                                           mode='bilinear')
        expanded_message = self.Conv_message(expanded_message)
        d2 = torch.cat((x1, d2, expanded_message), dim=1)
        d2 = self.Conv9(d2)

        out = self.Conv_1x1(d2)

        return out


class Extractor(nn.Module):
    def __init__(self, inchannel=64):
        super(Extractor, self).__init__()
        self.layer1 = SingleConv(inchannel, 64, 1)
        self.layer2 = nn.Sequential(ResidualBlock(64, 64, 1), ResidualBlock(64, 64, 2))
        self.layer3 = nn.Sequential(ResidualBlock(64, 64, 1), ResidualBlock(64, 64, 2))
        self.layer4 = nn.Sequential(ResidualBlock(64, 64, 1), ResidualBlock(64, 64, 2))
        self.layer5 = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.linear = nn.Linear(256, 30)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out.squeeze_(1)
        out = out.view(-1, 1, 256)
        out = self.linear(out)
        out.squeeze_(1)
        return out


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.extractor = Extractor()
        self.layer1 = nn.Sequential(
            SingleConv(3, 64, 1),
            SingleConv(64, 64, 1),
            SingleConv(64, 64, 1),
            ResidualBlock(64, 64, 1),
            ResidualBlock(64, 64, 1),
            ResidualBlock(64, 64, 1),
        )

    def forward(self, x):
        x1 = self.layer1(x)
        Message = self.extractor(x1)
        return Message


class Encoder_Decoder(nn.Module):
    def __init__(self, distortion):
        super(Encoder_Decoder, self).__init__()
        self.Encoder = U_Net_Encoder_Diffusion()
        self.Decoder = Decoder()
        self.distortion = distortion
        if distortion == 'Identity':
            self.Noiser = Identity()
        elif distortion == 'ScreenShooting':
            self.Noiser = ScreenShooting()

    def forward(self, x, m):
        Encoded_image = self.Encoder(x, m)
        Noised_image = self.Noiser(Encoded_image)
        Decoded_message = self.Decoder(Noised_image.float())
        return Encoded_image, Noised_image, Decoded_message
