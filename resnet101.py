import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import wandb
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, recall_score
from torchvision import models
from nih_loader import nih_loader

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--model', type=str, default='resnet101')
    args = parser.parse_args()

    wandb.init(project="NIH_Chest_X-ray",
               entity="hails",
               config=args.__dict__,
               name=f"ChestXray_{args.model}_lr:{args.lr}_Batch:{args.batch_size}"
               )

    train_loader, test_loader = nih_loader(batch_size=args.batch_size, num_workers=4, resize=True)

    if args.model == 'resnet50':
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 14)
        model = model.to(device)
    elif args.model == 'resnet101':
        model = models.resnet101(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 14)
        model = model.to(device)
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print('start training')

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{args.epochs}], Loss: {avg_loss:.4f}')
        wandb.log({'Train Loss': avg_loss}, step=epoch + 1)

        test_model(model, test_loader, epoch, criterion)

def test_model(model, test_loader, epoch, criterion):
    model.eval()
    true_labels = []
    predicted_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = torch.sigmoid(outputs) > 0.5
            true_labels.append(labels.cpu().numpy())
            predicted_labels.append(preds.cpu().numpy())

    true_labels = np.concatenate(true_labels)
    predicted_labels = np.concatenate(predicted_labels)

    accuracy = accuracy_score(true_labels.flatten(), predicted_labels.flatten())
    sensitivity = recall_score(true_labels.flatten(), predicted_labels.flatten(), average='macro')
    specificity = recall_score(true_labels.flatten(), predicted_labels.flatten(), average='macro', pos_label=0)

    avg_loss = total_loss / len(test_loader)
    print(f'Test Loss: {avg_loss:.4f}, '
          f'Accuracy: {accuracy:.4f}, '
          f'Sensitivity: {sensitivity:.4f}, '
          f'Specificity: {specificity:.4f}')
    wandb.log({
        'Test Loss': avg_loss,
        'Test Accuracy': accuracy,
        'Test Sensitivity': sensitivity,
        'Test Specificity': specificity
    }, step=epoch + 1)

if __name__ == '__main__':
    main()