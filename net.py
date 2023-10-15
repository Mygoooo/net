import torch
import torch.nn as nn

class my_network(nn.Module):
    def __init__(self):
        super(my_network, self).__init__()

        # 第一組卷積層
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)

        # 第二組卷積層
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)

        # 第三組卷積層
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(256)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)

        # 第四組卷積層
        self.conv9 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn9 = nn.BatchNorm2d(512)
        self.conv10 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn10 = nn.BatchNorm2d(512)
        self.conv11 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(512)
        self.conv12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(2, 2)

        # 第五組卷積層
        self.conv13 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn13 = nn.BatchNorm2d(512)
        self.conv14 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn14 = nn.BatchNorm2d(512)
        self.conv15 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn15 = nn.BatchNorm2d(512)
        self.conv16 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn16 = nn.BatchNorm2d(512)
        self.pool5 = nn.MaxPool2d(2, 2)

        # 全連接層
        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 12)
        self.dropout = nn.Dropout(0.5)  # 添加了50%的dropout

    def forward(self, x):
        # 第一組卷積層的前向傳播
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)

        # 第二組卷積層的前向傳播
        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)

        # 第三組卷積層的前向傳播
        x = torch.relu(self.bn5(self.conv5(x)))
        x = torch.relu(self.bn6(self.conv6(x)))
        x = torch.relu(self.bn7(self.conv7(x)))
        x = torch.relu(self.bn8(self.conv8(x)))
        x = self.pool3(x)

        # 第四組卷積層的前向傳播
        x = torch.relu(self.bn9(self.conv9(x)))
        x = torch.relu(self.bn10(self.conv10(x)))
        x = torch.relu(self.bn11(self.conv11(x)))
        x = torch.relu(self.bn12(self.conv12(x)))
        x = self.pool4(x)

        # 第五組卷積層的前向傳播
        x = torch.relu(self.bn13(self.conv13(x)))
        x = torch.relu(self.bn14(self.conv14(x)))
        x = torch.relu(self.bn15(self.conv15(x)))
        x = torch.relu(self.bn16(self.conv16(x)))
        x = self.pool5(x)

        x = x.view(-1, 512 * 7 * 7)

        # 全連接層的前向傳播，包括dropout
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x

model = my_network()
