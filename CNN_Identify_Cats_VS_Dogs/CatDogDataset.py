from torch.utils.data import Dataset
from PIL import Image

class CatDogDataset(Dataset):
    def __init__(self, dataset, transform=None):
        super(CatDogDataset, self).__init__()

        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        image_path = self.dataset[index]

        image = Image.open(image_path)
        image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = 0 if "dog" in image_path else 1

        return image, label
