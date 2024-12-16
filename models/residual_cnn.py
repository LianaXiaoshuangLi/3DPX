import torch
import torch.nn as nn

class Basic(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super(Basic, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out

class Residual_0(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Residual_0, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn_1 = nn.BatchNorm2d(out_channels)
        self.basic1_2 = Basic(in_channels, out_channels // 2, kernel_size=1, stride=1, padding=0)
        self.basic2_2 = Basic(out_channels // 2, out_channels // 2, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv2d(out_channels // 2, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn_2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        branch1 = self.conv_1(x)
        branch1 = self.bn_1(branch1)
        branch2 = self.basic1_2(x)
        branch2 = self.basic2_2(branch2)
        branch2 = self.conv_2(branch2)
        branch2 = self.bn_2(branch2)
        out = self.relu(branch1 + branch2)
        return out

class Residual_1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Residual_1, self).__init__()
        self.basic1_2 = Basic(in_channels, out_channels//2, kernel_size=1, stride=1, padding=0)
        self.basic2_2 = Basic(out_channels//2, out_channels//2, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv2d(out_channels//2, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn_2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        branch1 = x
        branch2 = self.basic1_2(x)
        branch2 = self.basic2_2(branch2)
        branch2 = self.conv_2(branch2)
        branch2 = self.bn_2(branch2)
        out = self.relu(branch1 + branch2)
        return out


class Encoder_0(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder_0, self).__init__()
        self.residual1 = Residual_1(in_channels, out_channels//2)
        self.residual2 = Residual_1(out_channels//2, out_channels//2)
        self.residual3 = Residual_0(out_channels//2, out_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.residual1(x)
        out = self.residual2(out)
        out = self.residual3(out)
        residual = out
        out = self.maxpool(out)
        return out, residual

class Encoder_1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder_1, self).__init__()
        self.residual1 = Residual_1(in_channels, out_channels)
        self.residual2 = Residual_1(out_channels, out_channels)
        self.residual3 = Residual_1(out_channels, out_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.residual1(x)
        out = self.residual2(out)
        out = self.residual3(out)
        residual = out
        out = self.maxpool(out)
        return out, residual

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.residual = Residual_0(in_channels=in_channels, out_channels=out_channels)
        self.deconv = nn.ConvTranspose2d(in_channels=out_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x, residual=None):
        if residual is not None:
            x = x + residual
        out = self.residual(x)
        out = self.deconv(out)
        return out


class ResidualCNN(nn.Module):
    def __init__(self, in_channels=1, out_channels=250):
        super(ResidualCNN, self).__init__()
        self.basic1 = Basic(in_channels=in_channels, out_channels=64, kernel_size=7, padding=3, stride=2)
        # self.basic1 = Basic(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1, stride=2)
        self.residual1 = Residual_0(in_channels=64, out_channels=128)
        self.encoder1 = Encoder_0(in_channels=128, out_channels=256)
        self.encoder2 = Encoder_1(in_channels=256, out_channels=256)
        self.encoder3 = Encoder_1(in_channels=256, out_channels=256)
        self.encoder4 = Encoder_1(in_channels=256, out_channels=256)

        self.residual2 = Residual_1(in_channels=256, out_channels=256)
        self.residual3 = Residual_1(in_channels=256, out_channels=256)
        self.residual4 = Residual_1(in_channels=256, out_channels=256)
        self.residual5 = Residual_1(in_channels=256, out_channels=256)

        self.decoder4 = Decoder(in_channels=256, out_channels=256)
        self.decoder3 = Decoder(in_channels=256, out_channels=256)
        self.decoder2 = Decoder(in_channels=256, out_channels=256)
        self.decoder1 = Decoder(in_channels=256, out_channels=256)
        self.decoder0 = Decoder(in_channels=256, out_channels=256)

        self.basic2 = Basic(in_channels=257, out_channels=256, kernel_size=1, padding=0, stride=1)
        self.final = nn.Conv2d(in_channels=256,out_channels=out_channels, kernel_size=1, padding=0, stride=1)

        self.loss_names = ['G_Recons', 'G_Proj']
        # self.old_lr = opt.lr

    def projection_loss(self, x, y):
        ''' orthogonal projections along each dimension of the generated 3D image
            x: (w, h, d) 3D image
        '''
        x_side0 = torch.mean(x, dim=0)
        x_side1 = torch.mean(x, dim=1)
        x_side2 = torch.mean(x, dim=2)
        y_side0 = torch.mean(y, dim=0)
        y_side1 = torch.mean(y, dim=1)
        y_side2 = torch.mean(y, dim=2)

        loss0 = torch.mean(torch.pow(x_side0 - y_side0, 2))
        loss1 = torch.mean(torch.pow(x_side1 - y_side1, 2))
        loss2 = torch.mean(torch.pow(x_side2 - y_side2, 2))

        return loss0 + loss1 + loss2

    def reconstruction_loss(self, x, y):
        loss = torch.sum(torch.pow(x - y, 2))
        return loss

    def forward(self, x, flattened):
        input = x
        x = self.basic1(x)
        x = self.residual1(x)
        x, residual1 = self.encoder1(x)
        x, residual2 = self.encoder2(x)
        x, residual3 = self.encoder3(x)
        x, residual4 = self.encoder4(x)

        x = self.residual2(x)
        x = self.residual3(x)
        x = self.residual4(x)
        x = self.residual5(x)

        x = self.decoder4(x)
        x = self.decoder3(x, residual4)
        x = self.decoder2(x, residual3)
        x = self.decoder1(x, residual2)
        x = self.decoder0(x, residual1)

        # x = x + residual1
        x = torch.cat((x, input), dim=1)
        x = self.basic2(x)
        out = self.final(x)
        return [self.reconstruction_loss(out, flattened), self.projection_loss(out, flattened)], out
        # return out

    def inference(self, x):
        input = x
        x = self.basic1(x)
        x = self.residual1(x)
        x, residual1 = self.encoder1(x)
        x, residual2 = self.encoder2(x)
        x, residual3 = self.encoder3(x)
        x, residual4 = self.encoder4(x)

        x = self.residual2(x)
        x = self.residual3(x)
        x = self.residual4(x)
        x = self.residual5(x)

        x = self.decoder4(x)
        x = self.decoder3(x, residual4)
        x = self.decoder2(x, residual3)
        x = self.decoder1(x, residual2)
        x = self.decoder0(x, residual1)

        # x = x + residual1
        x = torch.cat((x, input), dim=1)
        x = self.basic2(x)
        out = self.final(x)
        # return [self.reconstruction_loss(out, flattened), self.projection_loss(out, flattened)], out
        return out

