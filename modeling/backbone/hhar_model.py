import torch
from torch.autograd import Variable
import torch.nn as nn


class ConvHHAR2D(nn.Module):

    def __init__(self, indim=6, flatten=True):
        super(ConvHHAR2D, self).__init__()
        indim = indim
        trunk = [
            nn.Conv2d(indim, 32, kernel_size=(1, 3)),
            nn.ReLU(True),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 64, kernel_size=(1, 3)),
            nn.ReLU(True),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, kernel_size=(1, 3)),
            nn.MaxPool2d((1, 2)),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 256, kernel_size=(1, 3)),
            nn.ReLU(True),
            nn.BatchNorm2d(256),

            nn.Conv2d(256, 512, kernel_size=(1, 3)),
            nn.MaxPool2d((1, 2)),
            nn.ReLU(True),
            nn.BatchNorm2d(512),

            nn.Flatten()]
        self.trunk = nn.Sequential(*trunk)
        self.final_feat_dim = 30720

    def forward(self, x):
        out = self.trunk(x)
        return out


class ConvHHAR(nn.Module):

    def __init__(self, indim=6, num_classes=7):
        super(ConvHHAR, self).__init__()
        indim = indim
        trunk = [
            nn.Conv1d(indim, 32, kernel_size=3),
            nn.ReLU(True),
            nn.BatchNorm1d(32),

            nn.Conv1d(32, 64, kernel_size=3),
            nn.ReLU(True),
            nn.BatchNorm1d(64),

            nn.Conv1d(64, 128, kernel_size=3),
            nn.MaxPool1d(2),
            nn.BatchNorm1d(128),

            nn.Conv1d(128, 256, kernel_size=3),
            nn.ReLU(True),
            nn.BatchNorm1d(256),

            nn.Conv1d(256, 512, kernel_size=3),
            nn.MaxPool1d(2),
            nn.ReLU(True),
            nn.BatchNorm1d(512),

            nn.Flatten()]
        self.trunk = nn.Sequential(*trunk)
        self.final_feat_dim = 30720

        dense = [
            nn.Linear(self.final_feat_dim, 1024),
            nn.Linear(1024, 256),
            nn.Linear(256, num_classes)

        ]
        self.dense = nn.Sequential(*dense)

    def forward(self, x):
        out = self.trunk(x)
        out = self.dense(out)
        return out


class DeepSense(nn.Module):
    def __init__(self):
        super(DeepSense, self).__init__()
        self.acc_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(1, 18), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Dropout(0.2),
        )
        self.acc_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 3), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Dropout(0.2),
        )
        self.acc_conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 3), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Dropout(0.2),
        )
        self.gyro_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(1, 18), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Dropout(0.2),
        )
        self.gyro_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 3), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Dropout(0.2),
        )
        self.gyro_conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 3), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Dropout(0.2),
        )

        self.sensor_conv1 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(1, 2, 8), stride=1, padding=(0, 0, 0)),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            nn.Dropout(0.2),
        )
        self.sensor_conv2 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(1, 2, 6), stride=1, padding=(0, 0, 0)),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            nn.Dropout(0.2),
        )
        self.sensor_conv3 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(1, 2, 4,), stride=1, padding=(0, 0, 0)),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            nn.Dropout(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(52224, 6)
        )

    def forward(self, x):
        acc_input, gyro_input = x.chunk(2, dim=3)
        x1 = self.acc_conv1(acc_input)
        x1 = self.acc_conv2(x1)
        x1 = self.acc_conv3(x1)

        x2 = self.gyro_conv1(gyro_input)
        x2 = self.gyro_conv2(x2)
        x2 = self.gyro_conv3(x2)

        x1 = x1.view(x1.size()[0], x1.size()[1], 1, x1.size()[2], x1.size()[3])
        x2 = x2.view(x2.size()[0], x2.size()[1], 1, x2.size()[2], x2.size()[3])

        sensor_input = torch.cat((x1, x2), dim=2)
        x3 = self.sensor_conv1(sensor_input)
        x3 = self.sensor_conv2(x3)
        x3 = self.sensor_conv3(x3)

        # 加一层GRU
        x3 = torch.flatten(x3, 1)
        x3 = self.fc(x3)

        return x3


if __name__ == "__main__":
    input = torch.randn(8, 6, 256)
    model = ConvHHAR()
    output = model(input)
    print(output.size())
