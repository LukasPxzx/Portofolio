import os
import numpy as np
from skimage.feature import hog
from sklearn.cluster import KMeans
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from scipy.stats import mode
from PIL import Image

# Define dataset path
data_dir = "/Users/lukassspazo/Year 3 Python/Deep Learning for Images/dataset"
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")

# Function to extract HOG features
def extract_hog_features(folder):
    descriptors_list = []
    labels = []
    for subfolder in os.listdir(folder):
        subfolder_path = os.path.join(folder, subfolder)
        if os.path.isdir(subfolder_path):
            label = 0 if subfolder == "cats" else 1  # Assign 0 for cats, 1 for dogs
            for filename in tqdm(os.listdir(subfolder_path), desc=f"Extracting HOG from {subfolder}"):
                img_path = os.path.join(subfolder_path, filename)
                img = Image.open(img_path).convert("L")  # Convert to grayscale
                img = img.resize((64, 64))  # Resize for consistency
                img_np = np.array(img)  # Convert to numpy array
                features = hog(img_np, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
                descriptors_list.append(features)
                labels.append(label)
    return descriptors_list, labels

# Extract HOG features
train_features, train_labels = extract_hog_features(train_dir)
test_features, test_labels = extract_hog_features(test_dir)

# Combine all extracted features and labels for clustering
all_features = np.array(train_features + test_features)
all_labels = np.array(train_labels + test_labels)

# Apply K-means clustering
num_clusters = 2
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
kmeans.fit(all_features)

# Output clustering results
labels = kmeans.labels_
print("Cluster distribution:", np.bincount(labels))

# Map clusters to original labels by finding the best alignment
mapped_labels = np.zeros_like(labels)
for i in range(num_clusters):
    mask = (labels == i)
    mapped_labels[mask] = mode(all_labels[mask])[0]

# Compute accuracy
accuracy = accuracy_score(all_labels, mapped_labels)
print(f"Best accuracy after mapping clusters: {accuracy:.2f}")
