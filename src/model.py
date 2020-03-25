import torch.nn as nn
import torch


class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.leaky_reLU = nn.LeakyReLU(0.2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1, return_indices=True)
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=1)
        self.tanh = nn.Tanh()


        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.ConvTranspose2d(in_channels=1024, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.ConvTranspose2d(in_channels=512, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv9 = nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv10 =nn.ConvTranspose2d(in_channels=128, out_channels=2, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.leaky_reLU(self.conv1(x))
        out1 = x
        size1 = x.size()
        x, indices1 = self.pool(x)

        x = self.leaky_reLU(self.conv2(x))
        out2 = x
        size2 = x.size()
        x, indices2 = self.pool(x)

        x = self.leaky_reLU(self.conv3(x))
        out3 = x
        size3 = x.size()
        x, indices3 = self.pool(x)

        x = self.leaky_reLU(self.conv4(x))
        out4 = x
        size4 = x.size()
        x, indices4 = self.pool(x)

        x = self.leaky_reLU(self.conv5(x))

        x = self.leaky_reLU(self.conv6(x))

        x = self.unpool(x, indices4, output_size=size4)
        x = self.leaky_reLU(self.conv7(torch.cat((x, out4), 1)))

        x = self.unpool(x, indices3, output_size=size3)
        x = self.leaky_reLU(self.conv8(torch.cat((x, out3), 1)))

        x = self.unpool(x, indices2, output_size=size2)
        x = self.leaky_reLU(self.conv9(torch.cat((x, out2), 1)))

        x = self.unpool(x, indices1, output_size=size1)
        x = self.tanh(self.conv10(torch.cat((x, out1), 1)))

        return x

