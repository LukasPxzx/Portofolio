import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_score, recall_score, f1_score
import snntorch as snn
from snntorch import surrogate
from img2vec_pytorch import Img2Vec
from PIL import Image

# Img2VecExtraction Class
class Img2VecExtraction:
    def __init__(self, categories):
        self.categories = categories
        self.train_features = []
        self.test_features = []
        self.train_lables = []
        self.test_lables = []
        self.img2vec = Img2Vec(model="resnet50")
        self.train_dir = "/Users/lukassspazo/Year 3 Python/Deep Learning for Images/dataset/train"  
        self.test_dir = "/Users/lukassspazo/Year 3 Python/Deep Learning for Images/dataset/test"    

    def _is_grey_scale_image(self, image):
        return image.mode == 'L'

    def _to_three_channel_image(self, image):
        return image.convert('RGB')

    def _store_features_and_lables(self, label, data_path, feature_array, labels_array):
        for image in os.listdir(data_path):
            train_image_path = os.path.join(data_path, image)
            train_img = Image.open(train_image_path)
            if self._is_grey_scale_image(train_img):
                train_img = self._to_three_channel_image(train_img)
            train_img_features = self.img2vec.get_vec(train_img)
            feature_array.append(train_img_features)
            labels_array.append(label)

    def class_label_str2int(self):
        int_train_labels = []
        int_test_labels = []
        
        for label in self.train_lables:
            int_train_labels.append(self.categories.index(label))
        
        for label in self.test_lables:
            int_test_labels.append(self.categories.index(label))

        return int_train_labels, int_test_labels

    def feature_extraction(self):
        for category in self.categories:
            train_data_path = os.path.join(self.train_dir, category)
            test_data_path = os.path.join(self.test_dir, category)
            
            self._store_features_and_lables(category, train_data_path, self.train_features, self.train_lables)
            self._store_features_and_lables(category, test_data_path, self.test_features, self.test_lables)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Feature extraction
categories = ['cats', 'dogs']
img_extr = Img2VecExtraction(categories)
img_extr.feature_extraction()

int_train_labels, int_test_labels = img_extr.class_label_str2int()
feature_vectors = torch.tensor(img_extr.train_features)
labels = torch.tensor(int_train_labels)
test_features = torch.tensor(img_extr.test_features)
test_labels = torch.tensor(int_test_labels)

def get_labels(data_loader):
    all_labels = []
    for _, labels in data_loader:
        all_labels.append(labels)
    return torch.cat(all_labels).numpy()

# SNN Model
class Img2Vec_SNN(nn.Module):
    def __init__(self, input_size=2048, time_steps=100, num_classes=2):
        super(Img2Vec_SNN, self).__init__()
        self.time_steps = time_steps
        self.fc1 = nn.Linear(input_size, 256)
        self.lif1 = snn.Leaky(beta=0.9, spike_grad=surrogate.fast_sigmoid())
        self.fc2 = nn.Linear(256, num_classes)
        self.lif2 = snn.Leaky(beta=0.9, spike_grad=surrogate.fast_sigmoid())

    def forward(self, x):
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

# Data loaders
train_loader = DataLoader(TensorDataset(feature_vectors, labels), batch_size=64, shuffle=True, drop_last=True)
test_loader = DataLoader(TensorDataset(test_features, test_labels), batch_size=64, shuffle=True, drop_last=True)

train_label = get_labels(train_loader)
test_label = get_labels(test_loader)

# Model training
model = Img2Vec_SNN(input_size=2048, num_classes=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
epochs = 100
track_loss = []
track_accuracy = []
all_preds = []


for epoch in range(epochs):
    model.train()
    correct, total_loss = 0, 0
    all_preds = []  # Reset predictions for this epoch
    for batch_features, batch_labels in train_loader:
        batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
        outputs = model(batch_features)
        loss = criterion(outputs, batch_labels)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Get predictions for the current batch
        _, preds = torch.max(outputs, 1)
        
        # Append predictions to the list
        all_preds.extend(preds.cpu().numpy())  # Append predictions to the list
        correct += (preds == batch_labels).sum().item()

    accuracy = correct / len(train_loader.dataset)

    # Check lengths before calculating metrics
    print(f"Length of train_label: {len(train_label)}")
    print(f"Length of all_preds: {len(all_preds)}")

    # Use all_preds instead of preds for calculating metrics
    precision = precision_score(train_label, all_preds, average='weighted', zero_division=0)
    recall = recall_score(train_label, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(train_label, all_preds, average='weighted', zero_division=0)
    
    print(f"Epoch {epoch + 1}/{epochs}: Loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    track_loss.append(total_loss)
    track_accuracy.append(accuracy)
# Plotting results
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(track_loss, label='Training Loss')
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(track_accuracy, label='Training Accuracy')
plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()
