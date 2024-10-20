import argparse
import torch
from dataset import get_dataset
from model import get_model
from utils import get_dataloader, get_optimizer, get_scheduler, EarlyStopping, init_logging, log_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    parser = argparse.ArgumentParser(description="Generic Training Pipeline")
    parser.add_argument('--dataset', type=str, required=True, choices=['MNIST', 'CIFAR-10', 'CIFAR-100'])
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--optimizer', type=str, required=True)
    parser.add_argument('--scheduler', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--project_name', type=str, default=None)
    args = parser.parse_args()

    dataset = get_dataset(args.dataset, './data')
    dataloader = get_dataloader(dataset, args.batch_size)
    model = get_model(args.model).to(device)
    optimizer = get_optimizer(args.optimizer, model, args.lr)
    scheduler = get_scheduler(args.scheduler, optimizer)
    writer = init_logging(args.log_dir, args.use_wandb, args.project_name)

    early_stopping = EarlyStopping()

    for epoch in range(args.epochs):
        model.train()
        for batch in dataloader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = torch.nn.functional.cross_entropy(outputs, targets)
            loss.backward()
            optimizer.step()

        if scheduler:
            scheduler.step()

        # Log metrics
        metrics = {'loss': loss.item()}
        log_metrics(writer, epoch, metrics, args.use_wandb)

        # Early stopping
        early_stopping(loss.item())
        if early_stopping.early_stop:
            print("Early stopping")
            break

if __name__ == '__main__':
    main()