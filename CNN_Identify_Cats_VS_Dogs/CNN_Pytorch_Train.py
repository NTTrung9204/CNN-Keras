import torch
import os
from CatDogDataset import CatDogDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from CNN_Pytorch_Model import CNN_Attention
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from CNN_DenseNet121_Torch_Model import DenseNet121
import sys

if __name__ == "__main__":
    TRAIN_PATH_DOG = "dataset/train/dog/"
    TRAIN_PATH_CAT = "dataset/train/cat/"
    EPOCHS = 5
    BATCH_SIZE = 64

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = []

    [dataset.append(TRAIN_PATH_DOG + image_name) for image_name in os.listdir(TRAIN_PATH_DOG)]
    [dataset.append(TRAIN_PATH_CAT + image_name) for image_name in os.listdir(TRAIN_PATH_CAT)]

    # transforms = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    # ])

    transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])

    CD_Dataset = CatDogDataset(dataset=dataset, transform=transforms)

    CD_Loader = DataLoader(CD_Dataset, BATCH_SIZE, shuffle=True, num_workers=2)

    # model = CNN_Attention()
    model = DenseNet121(growth_rate=32, num_blocks=[6, 12, 24, 16], num_classes=2)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    losses = []
    for epoch in range(EPOCHS):
        for sub_epoch, (images, labels) in enumerate(CD_Loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            prediction = model(images)

            loss = criterion(prediction, labels)
            loss.backward()

            optimizer.step()

            losses.append(loss.item())

            sys.stdout.write(f"\r[{epoch + 1}|{EPOCHS}], [{sub_epoch + 1:4d}|{len(CD_Loader)}], Loss: {losses[-1]:.4f}")
        print()

    plt.plot(losses)
    plt.xlabel("Epochs")
    plt.ylabel("Losses")
    plt.title("Training")

    plt.show()

    torch.save(model.state_dict(), 'model_weights.pth')