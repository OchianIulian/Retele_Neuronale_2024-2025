import pandas as pd
import torch
from pipeline import run_pipeline

def generate_submission(model, dataloader, device):
    model.eval()
    results = []
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            results.extend(preds.cpu().numpy())

    submission = pd.DataFrame({
        "ID": list(range(len(results))),
        "target": results
    })
    submission.to_csv('submission.csv', index=False)
    print("Submission file saved as 'submission.csv'")


if __name__ == "__main__":
    model, test_loader, device = run_pipeline()
    generate_submission(model, test_loader, device)
