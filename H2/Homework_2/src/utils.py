import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import wandb

def get_dataloader(dataset, batch_size, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def get_optimizer(name, model, lr):
    if name == 'SGD':
        return torch.optim.SGD(model.parameters(), lr=lr)
    elif name == 'Adam':
        return torch.optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError("Unsupported optimizer")

def get_scheduler(name, optimizer):
    if name == 'StepLR':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    elif name == 'ReduceLROnPlateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    else:
        return None

class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

def init_logging(log_dir, use_wandb=False, project_name=None):
    writer = SummaryWriter(log_dir)
    if use_wandb:
        wandb.init(project=project_name)
    return writer

def log_metrics(writer, epoch, metrics, use_wandb=False):
    for key, value in metrics.items():
        writer.add_scalar(key, value, epoch)
        if use_wandb:
            wandb.log({key: value})