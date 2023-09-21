from torch import nn 

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()

        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1)
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, padding = 1)

        self.conv4 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, padding = 1)
        self.conv5 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, padding = 1)

        self.conv6 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, padding = 1)
        self.conv7 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, padding = 1)
        self.conv8 = nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3, padding = 1)

        self.conv9 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, padding = 1)
        self.conv10 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, padding = 1)
        self.conv11 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, padding = 1)
        self.conv12 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, padding = 1)
        self.conv13 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, padding = 1)

        self.fc1 = nn.Linear(in_features = 25088, out_features = 4096)
        self.fc2 = nn.Linear(in_features = 4096, out_features = 4096)
        self.fc3 = nn.Linear(in_features = 4096, out_features = 10)

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2)

    def forward(self, x):

        x = self.conv1(x) 
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv5(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.relu(x)
        x = self.conv7(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv8(x)
        x = self.relu(x)
        x = self.conv9(x)
        x = self.relu(x)
        x = self.conv10(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv11(x)
        x = self.relu(x)
        x = self.conv12(x)
        x = self.relu(x)
        x = self.conv13(x)
        x = self.maxpool(x)

        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        
        return x