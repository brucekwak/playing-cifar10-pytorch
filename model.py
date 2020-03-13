# import
import torch 
import torch.nn as nn
import torchvision


# [Reference] https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/convolutional_neural_network/main.py
class SimpleConvNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SimpleConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_dim, 16, kernel_size=5, stride=1, padding=2), # in_channels, out_channels
            nn.BatchNorm2d(16),  # num_features
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.fc = nn.Linear(8*8*32, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out
    
