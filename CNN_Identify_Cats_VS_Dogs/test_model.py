import torch
import os
from CatDogDataset import CatDogDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from CNN_Pytorch_Model import CNN_Attention
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from CNN_DenseNet121_Torch_Model import DenseNet121

def evaluate(model, valid_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in valid_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)

            # Predicted class labels
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy

if __name__ == "__main__":
    TEST_PATH_DOG = "dataset/test/dog/"
    TEST_PATH_CAT = "dataset/test/cat/"
    EPOCHS = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = []

    [dataset.append(TEST_PATH_DOG + image_name) for image_name in os.listdir(TEST_PATH_DOG)]
    [dataset.append(TEST_PATH_CAT + image_name) for image_name in os.listdir(TEST_PATH_CAT)]

    print(len(dataset))

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

    CD_Loader = DataLoader(CD_Dataset, 128, shuffle=True, num_workers=2)

    model = CNN_Attention()
    # model = DenseNet121(growth_rate=32, num_blocks=[6, 12, 24, 16], num_classes=2)
    model.to(device)
    model.load_state_dict(torch.load("model_weights.pth"))

    print(evaluate(model, CD_Loader, device))
