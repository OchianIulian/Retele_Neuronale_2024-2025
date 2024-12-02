import torch

def train_model(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0 #Va stoca suma pierderilor (loss) pentru toate batch-urile din epoca curentă.
    correct = 0 #Numără predicțiile corecte (clasificări corecte) pentru a calcula acuratețea
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item() #se adauga entropia/pierderea batchului curent
        _, preds = torch.max(outputs, 1) #Se iau procentajele maxime de preziceri pentru fiecare label
        correct += (preds == labels).sum().item() #se aduna predictiile corecte

    accuracy = correct / len(dataloader.dataset) #se calculeaza acuratetea
    return total_loss / len(dataloader), accuracy #se returneaza eroarea medie si acuratetea


def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()

    accuracy = correct / len(dataloader.dataset)
    return total_loss / len(dataloader), accuracy
