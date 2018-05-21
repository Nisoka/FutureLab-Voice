import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.autograd import Variable
import torch.nn.functional as F
import math

__all__ = ['AngleLoss', 'SpeakerResNet', 'spearkerresnet24', 'spearkerresnet14']


def myphi(x, m):
    x = x * m
    return 1 - x ** 2 / math.factorial(2) + x ** 4 / math.factorial(4) - x ** 6 / math.factorial(6) + \
           x ** 8 / math.factorial(8) - x ** 9 / math.factorial(9)


class AngleLinear(nn.Module):
    def __init__(self, in_features, out_features, m=4, phiflag=True):
        super(AngleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.phiflag = phiflag
        self.m = m
        self.mlambda = [
            lambda x: x ** 0,
            lambda x: x ** 1,
            lambda x: 2 * x ** 2 - 1,
            lambda x: 4 * x ** 3 - 3 * x,
            lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
            lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x
        ]

    def forward(self, input):
        x = input  # size=(B,F)    F is feature len
        w = self.weight  # size=(F,Classnum) F=in_features Classnum=out_features

        ww = w.renorm(2, 1, 1e-5).mul(1e5)
        xlen = x.pow(2).sum(1).pow(0.5)  # size=B
        wlen = ww.pow(2).sum(0).pow(0.5)  # size=Classnum

        cos_theta = x.mm(ww)  # size=(B,Classnum)
        cos_theta = cos_theta / xlen.view(-1, 1) / wlen.view(1, -1)
        cos_theta = cos_theta.clamp(-1, 1)

        if self.phiflag:
            cos_m_theta = self.mlambda[self.m](cos_theta)
            theta = Variable(cos_theta.data.acos())
            k = (self.m * theta / 3.14159265).floor()
            n_one = k * 0.0 - 1
            phi_theta = (n_one ** k) * cos_m_theta - 2 * k
        else:
            theta = cos_theta.acos()
            phi_theta = myphi(theta, self.m)
            phi_theta = phi_theta.clamp(-1 * self.m, 1)

        cos_theta = cos_theta * xlen.view(-1, 1)
        phi_theta = phi_theta * xlen.view(-1, 1)
        output = (cos_theta, phi_theta)
        return output  # size=(B,Classnum,2)


class AngleLoss(nn.Module):
    def __init__(self, gamma=0):
        super(AngleLoss, self).__init__()
        self.gamma = gamma
        self.it = 0
        self.LambdaMin = 5.0
        self.LambdaMax = 1500.0
        self.lamb = 1500.0

    def forward(self, input, target):
        self.it += 1
        cos_theta, phi_theta = input
        target = target.view(-1, 1)  # size=(B,1)

        index = cos_theta.data * 0.0  # size=(B,Classnum)
        index.scatter_(1, target.data.view(-1, 1), 1)
        index = index.byte()
        index = Variable(index)

        self.lamb = max(self.LambdaMin, self.LambdaMax / (1 + 0.1 * self.it))
        output = cos_theta * 1.0  # size=(B,Classnum)
        output[index] -= cos_theta[index] * (1.0 + 0) / (1 + self.lamb)
        output[index] += phi_theta[index] * (1.0 + 0) / (1 + self.lamb)

        logpt = F.log_softmax(output, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        loss = -1 * (1 - pt) ** self.gamma * logpt
        loss = loss.mean()

        return loss


class TDNNRedisualBlock(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.activation1 = nn.PReLU()
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)
        self.activation2 = nn.PReLU()

    def forward(self, input):
        x = self.conv1(input)
        x = self.activation1(x)
        x = self.conv2(x)
        x += input
        x = self.activation2(x)
        return x


class MaxFeatureMap(nn.Module):
    def forward(self, input):
        split_size = int(input.shape[1] / 2)
        chunka, chunkb = torch.split(input, split_size, 1)
        aggregate = torch.max(chunka, chunkb)
        return aggregate


class SpeakerResNet(nn.Module):
    def __init__(self, input_channel=23, M=10, num_classes=5, softmax=False, dropout=0.0):
        super().__init__()
        self.softmax = softmax
        self.frame1 = nn.Conv1d(input_channel, 128, 3)
        self.maxpool = nn.MaxPool2d((2, 2))
        self.residual_blocks = nn.Sequential(*[TDNNRedisualBlock() for _ in range(M)])
        self.frame2 = nn.Conv1d(64, 1024, 1)
        self.mfm = MaxFeatureMap()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 512)
        AngleLinear(256, num_classes)
        if softmax:
            self.fc3 = nn.Linear(256, num_classes)
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        x = self.frame1(input)
        x = self.maxpool(x)
        x = self.residual_blocks(x)
        x = self.frame2(x)
        x = self.maxpool(x)
        mean = torch.mean(x, 2)
        std = torch.std(x, 2)
        x = torch.cat((mean, std), dim=1)
        if self.dropout: x = self.dropout(x)
        x = self.fc1(x)
        x = self.mfm(x)
        # if self.dropout: x = self.dropout(x)
        # x = self.fc2(x)
        # x = self.mfm(x)
        if self.softmax:
            if self.dropout: x = self.dropout(x)
            x = self.fc3(x)
        else:
            x = self.asoftmax(x)
        return x


def spearkerresnet24(args, **kwargs):
    return SpeakerResNet(num_classes=args.output_classes, softmax=args.softmax, dropout=args.dropout)


def spearkerresnet14(args, **kwargs):
    return SpeakerResNet(num_classes=args.output_classes, M=5, softmax=args.softmax, dropout=args.dropout)


def main():
    model = SpeakerResNet()
    x = torch.randn(7, 23, 120)
    x.requires_grad = True
    y = model(x)
    criterion = AngleLoss()
    loss = criterion(y, torch.randint(0, 4, (7,), dtype=torch.int64))
    print(loss)
    loss.backward()


if __name__ == '__main__':
    main()
