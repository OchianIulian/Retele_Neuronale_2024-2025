import torchvision.transforms as transforms
import torchvision.datasets as datasets

def get_dataset(name, data_dir):
    if name == 'MNIST':
        transform = transforms.Compose([transforms.ToTensor()])
        return datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    elif name == 'CIFAR-10':
        transform = transforms.Compose([transforms.ToTensor()])
        return datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
    elif name == 'CIFAR-100':
        transform = transforms.Compose([transforms.ToTensor()])
        return datasets.CIFAR100(data_dir, train=True, download=True, transform=transform)
    else:
        raise ValueError("Unsupported dataset")