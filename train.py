# import your model from net.py
from net import my_network

'''
    You can add any other package, class and function if you need.
    You should read the .jpg from "./dataset/train/" and save your weight to "./w_{student_id}.pth"
'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from PIL import Image
from torchvision.transforms import ToTensor, Normalize
from torchvision import transforms
import os
import shutil

import pandas as pd

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []

        # 从 CSV 文件读取标签
        df = pd.read_csv(csv_file)

        for index, row in df.iterrows():
            img_name = row['name']
            label = row['label']

            img_path = os.path.join(root_dir, img_name)
            self.images.append(img_path)
            self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        img = Image.open(img_path).convert("RGB")

        # 如果有转换函数，应用转换
        if self.transform:
            img = self.transform(img)

        return img, label
    

def train():
    student_id = "110652045"
    num_epochs = 25
    learning_rate = 0.00001
    validation_split = 0.2

    # 載入模型
    model = my_network()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # 計算數據集的平均值和標準差
    transform = ToTensor()
    dataset = CustomDataset(root_dir="./dataset/train", csv_file="./dataset/train/train.csv", transform=transform)


    # 添加更多的数据增强
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # 将正規化的轉換添加到數據集
    train_dataset = CustomDataset(root_dir="./dataset/train", csv_file="./dataset/train/train.csv", transform=transform_train)
    val_dataset = CustomDataset(root_dir="./dataset/train", csv_file="./dataset/train/train.csv", transform=transform_val)

    # 切割驗證集
    dataset_size = len(train_dataset)
    val_size = int(validation_split * dataset_size)
    train_size = dataset_size - val_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    # 使用 SubsetRandomSampler 创建 DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 定義損失函數和優化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 訓練迴圈
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {running_loss / len(train_dataloader)}")

        # 在验证集上评估
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            accuracy = correct / total
            print(f"Epoch {epoch + 1}/{num_epochs}, Correct Rate: {accuracy}")

    # 保存權重
    torch.save(model.state_dict(), f"./w_{student_id}.pth")

if __name__ == "__main__":
    train()