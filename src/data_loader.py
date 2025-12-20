from datasets import load_dataset
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

class FER2013Dataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform
    
    def __getitem__(self, idx):
        image = self.dataset[idx]
        labels = image['cls']
        pixel_values = self.transform(image['jpg'])
        return {
            'pixel_values': pixel_values,
            'labels': labels
        }
    
    def __len__(self):
        return len(self.dataset)

def get_loader(batch_size=64):
    ds = load_dataset("clip-benchmark/wds_fer2013")
    print("Successfully loaded dataset")
    train_transform = transforms.Compose([
        transforms.Resize((48,48)),
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.5,),(0.5,))
    ])
    test_transform = transforms.Compose([
        transforms.Resize((48,48)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,),(0.5,))
    ])
    print("Successfully created transforms")
    train_dataset = FER2013Dataset(ds['train'], transform=train_transform)
    test_dataset = FER2013Dataset(ds['test'], transform=test_transform)
    print("Successfully split datasets")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    print("Successfully created dataloaders")
    return (train_loader, test_loader)


