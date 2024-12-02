import torch
from torch import optim
from torch.nn import CrossEntropyLoss
from dataset import get_dataloaders
from simple_mlp import SimpleMLP
from train_eval import train_model, evaluate_model

def run_pipeline():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get DataLoaders
    train_loader, test_loader = get_dataloaders(batch_size=16)

    # Initialize model, criterion, optimizer
    model = SimpleMLP().to(device)
    criterion = CrossEntropyLoss()#entropia = diferența dintre probabilitățile prezise și etichetele reale
    optimizer = optim.Adam(model.parameters(), lr=0.001) #Ajusteaza weights si biases pe baza entropiei

    num_epochs = 40
    for epoch in range(num_epochs):
        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate_model(model, test_loader, criterion, device)
        print(
            f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    return model, test_loader, device
