import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.CondNet = nn.Sequential(
            nn.Conv2d(3, 128, 4, 4), nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, 1), nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, 1), nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, 1), nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 32, 1)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.lastOut = nn.Linear(32, 3)  # 32是ConvNet最后输出通道数

    def forward(self, x):
        out = self.CondNet(x)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.lastOut(out)
        return out

# 测试代码，构造一个假输入
if __name__ == '__main__':
    model = Classifier()
    dummy_input = torch.randn(4, 3, 64, 64)  # batch=4, RGB 64x64图像
    output = model(dummy_input)
    print('Output shape:', output.shape)
