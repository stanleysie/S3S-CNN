import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class MER(nn.Module):
    def __init__(self, in_channels=1, num_classes=6):
        super(MER, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels=8, kernel_size=3, stride=2, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.batch_norm = nn.BatchNorm2d
        self.fc = nn.Linear(in_features=24 * 16 * 16, out_features=num_classes)

        self.conv_layer1 = nn.Sequential(
            self.conv1, self.batch_norm(8), self.relu, self.max_pool, self.dropout
        )

        self.conv_layer2 = nn.Sequential(
            self.conv1, self.batch_norm(8), self.relu, self.max_pool, self.dropout
        )

        self.conv_layer3 = nn.Sequential(
            self.conv1, self.batch_norm(8), self.relu, self.max_pool, self.dropout
        )
    
    def forward(self, x):
        ops, h, v = x
        x1 = self.conv_layer1(ops)
        x2 = self.conv_layer2(h)
        x3 = self.conv_layer3(v)

        x = torch.cat((x1, x2, x3), dim=1)
        x_avg = self.avg_pool(x)

        fig, ax = plt.subplots(ncols=3, nrows=3, figsize=(16, 8))
        ax[0, 0].imshow(ops[0, 0, :, :].cpu().detach().numpy(), cmap='gray')
        ax[0, 1].imshow(h[0, 0, :, :].cpu().detach().numpy(), cmap='gray')
        ax[0, 2].imshow(v[0, 0, :, :].cpu().detach().numpy(), cmap='gray')
        ax[1, 0].imshow(x1[0, 1, :, :].cpu().detach().numpy(), cmap='gray')
        ax[1, 1].imshow(x2[0, 1, :, :].cpu().detach().numpy(), cmap='gray')
        ax[1, 2].imshow(x3[0, 1, :, :].cpu().detach().numpy(), cmap='gray')
        ax[2, 0].imshow(x[0, 1, :, :].cpu().detach().numpy(), cmap='gray')
        ax[2, 1].imshow(x_avg[0, 1, :, :].cpu().detach().numpy(), cmap='gray')
        plt.show()

        x_avg = x_avg.view(x_avg.size(0), -1)
        x_fc = self.fc(x_avg)

        return x_fc