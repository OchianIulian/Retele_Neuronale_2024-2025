import timm

from preact_resnet import PreActResNet18


def get_model(name):
    if name == 'resnet18_cifar10':
        return timm.create_model('resnet18', pretrained=False, num_classes=10)
    elif name == 'PreActResNet18':
        # Assuming PreActResNet18 is defined elsewhere
        return PreActResNet18()
    elif name == 'MLP':
        # Define MLP model
        pass
    elif name == 'LeNet':
        # Define LeNet model
        pass
    else:
        raise ValueError("Unsupported model")