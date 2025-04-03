import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torchvision.models as models
import snntorch as snn
from snntorch import surrogate
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score

# Set device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Data transformations
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Datasets
train_dataset = datasets.ImageFolder(root='/Users/lukassspazo/Year 3 Python/Deep Learning for Images/dataset/train', transform=train_transform)
test_dataset = datasets.ImageFolder(root='/Users/lukassspazo/Year 3 Python/Deep Learning for Images/dataset/test', transform=val_transform)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=True)

# Model definition
class VGG16_SNN(nn.Module):
    def __init__(self, time_steps=100, num_classes=2):
        super(VGG16_SNN, self).__init__()
        self.time_steps = time_steps
        vgg16 = models.vgg16(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg16.features.children()))
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.fc1 = nn.Linear(512 * 7 * 7, 256)
        self.lif1 = snn.Leaky(beta=0.9, spike_grad=surrogate.fast_sigmoid())
        self.fc2 = nn.Linear(256, num_classes)
        self.lif2 = snn.Leaky(beta=0.9, spike_grad=surrogate.fast_sigmoid())

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        spk_out = 0
        for _ in range(self.time_steps):
            x0 = self.fc1(x)
            spk1, mem1 = self.lif1(x0, mem1)
            x0 = self.fc2(spk1)
            spk2, mem2 = self.lif2(x0, mem2)
            spk_out += spk2
        return spk_out / self.time_steps

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
model = VGG16_SNN().to(device)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

# Tracking metrics
track_loss = []
track_accuracy = []
train_recall_hist = []
train_preci_hist = []
train_f1_hist = []
test_loss_hist = []
test_accuracy_hist = []

# Training loop
epochs = 100
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    model.train()
    correct = 0
    total_loss = 0
    prediction = np.array([])
    sum_batch_label = np.array([])

    for batch_features, batch_labels in train_loader:
        sum_batch_label = np.append(sum_batch_label, batch_labels)
        batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)

        outputs = model(batch_features)
        loss = criterion(outputs, batch_labels)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        correct += (preds == batch_labels).sum().item()
        preds = preds.cpu().numpy()
        prediction = np.append(prediction, preds)

    accuracy = correct / len(sum_batch_label)
    precision = precision_score(sum_batch_label, prediction, zero_division=0)
    recall = recall_score(sum_batch_label, prediction, zero_division=0)
    f1 = f1_score(sum_batch_label, prediction, zero_division=0)
    print(f"Train Loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

    track_loss.append(total_loss)
    track_accuracy.append(accuracy)
    train_recall_hist.append(recall)
    train_preci_hist.append(precision)
    train_f1_hist.append(f1)

    # Evaluation phase
    model.eval()
    correct = 0
    total_loss = 0
    prediction = np.array([])
    sum_batch_label = np.array([])

    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            sum_batch_label = np.append(sum_batch_label, batch_labels)
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)

            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            total_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            correct += (preds == batch_labels).sum().item()
            preds = preds.cpu().numpy()
            prediction = np.append(prediction, preds)

    accuracy = correct / len(sum_batch_label)
    precision = precision_score(sum_batch_label, prediction, zero_division=0)
    recall = recall_score(sum_batch_label, prediction, zero_division=0)
    f1 = f1_score(sum_batch_label, prediction, zero_division=0)
    print(f"Test Loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

    test_loss_hist.append(total_loss)
    test_accuracy_hist.append(accuracy)

# Visualization
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Training Loss
axs[0, 0].plot(track_loss, label='Training Loss')
axs[0, 0].set_title("Training Loss")
axs[0, 0].set_xlabel("Epoch")
axs[0, 0].set_ylabel("Loss")
axs[0, 0].legend()

# Training Accuracy
axs[0, 1].plot(track_accuracy, label='Training Accuracy')
axs[0, 1].set_title("Training Accuracy")
axs[0, 1].set_xlabel("Epoch")
axs[0, 1].set_ylabel("Accuracy")
axs[0, 1].legend()

# Testing Loss
axs[1, 0].plot(test_loss_hist, label='Testing Loss')
axs[1, 0].set_title("Testing Loss")
axs[1, 0].set_xlabel("Epoch")
axs[1, 0].set_ylabel("Loss")
axs[1, 0].legend()

# Testing Accuracy
axs[1, 1].plot(test_accuracy_hist, label='Testing Accuracy')
axs[1, 1].set_title("Testing Accuracy")
axs[1, 1].set_xlabel("Epoch")
axs[1, 1].set_ylabel("Accuracy")
axs[1, 1].legend()

plt.tight_layout()
plt.show()
