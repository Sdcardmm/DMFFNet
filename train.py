import os
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import image_paths, label_list
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataloader import MyDataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, confusion_matrix
from model.DMFFNet import DMFFNet
import torch
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = MyDataset(image_paths=image_paths, label_paths=label_list, transform=transform)
batch_size = 8
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'device on {device}')

loss_fn = nn.CrossEntropyLoss()

model = DMFFNet(in_ch=3, out_ch=2)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=5, min_lr=1e-6)

def train_model(train_loader, model, loss_fn, optimizer, scheduler, num_epochs=60):
    model.train()
    save_dir = 'model_weights/medical_trans'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for epoch in range(num_epochs):
        total_loss = 0.0
        total_correct = 0
        all_labels = []
        all_outputs = []
        all_predicted = []

        for data in train_loader:
            imgs, labels = data
            imgs = imgs.to(device)
            labels = labels.to(device).long()

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(probabilities, 1)

            total_loss += loss.item() * imgs.size(0)
            total_correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().detach().numpy())
            all_outputs.extend(probabilities[:, 1].cpu().detach().numpy())
            all_predicted.extend(predicted.cpu().detach().numpy())

        epoch_loss = total_loss / len(train_loader.dataset)
        epoch_acc = total_correct / len(train_loader.dataset)
        epoch_auc = roc_auc_score(all_labels, all_outputs)

        scheduler.step(epoch_loss)

        print(f'Epoch {epoch + 1}/{num_epochs} | '
              f'Loss: {epoch_loss:.4f} | '
              f'Acc: {epoch_acc * 100:.2f}% | '
              f'AUC: {epoch_auc:.4f}')

        model_path = os.path.join(save_dir, f'model_epoch_{epoch + 1}.pth')
        torch.save(model.state_dict(), model_path)
        print(f'Saved model to {model_path}')

    cm = confusion_matrix(all_labels, all_predicted)

    print(f'Confusion Matrix:\n {cm}')

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['Actual 0', 'Actual 1'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.show()

    return cm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_model(train_loader, model, loss_fn, optimizer, scheduler, num_epochs=100)