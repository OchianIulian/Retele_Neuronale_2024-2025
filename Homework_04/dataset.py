from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Normalize, Compose
from torch.utils.data import DataLoader

def get_dataloaders(batch_size=64):
    # Define transformations
    train_transforms = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
    test_transforms = Compose([ToTensor(), Normalize((0.5,), (0.5,))])

    # Load datasets
    train_dataset = MNIST(root='data', train=True, transform=train_transforms, download=True)
    test_dataset = MNIST(root='data', train=False, transform=test_transforms, download=True)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
