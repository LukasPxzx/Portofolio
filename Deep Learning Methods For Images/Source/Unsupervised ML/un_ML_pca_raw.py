import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from PIL import Image
from scipy.stats import mode

# Define dataset path
data_dir = "/Users/lukassspazo/Year 3 Python/Deep Learning for Images/dataset"
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")

# Function to load images and labels
def load_images_from_folder(folder, img_size=(64, 64)):
    images = []
    labels = []
    for subfolder in os.listdir(folder):
        subfolder_path = os.path.join(folder, subfolder)
        if os.path.isdir(subfolder_path):
            label = 0 if subfolder == "cats" else 1  # Assign 0 for cats, 1 for dogs
            for filename in tqdm(os.listdir(subfolder_path), desc=f"Loading {subfolder}"):
                img_path = os.path.join(subfolder_path, filename)
                img = Image.open(img_path).convert("RGB")  # Open image and convert to RGB
                img = img.resize(img_size)  # Resize image
                img_np = np.array(img)  # Convert to numpy array
                images.append(img_np.flatten())  # Flatten image to 1D array
                labels.append(label)
    return images, labels

# Load images and labels
train_images, train_labels = load_images_from_folder(train_dir)
test_images, test_labels = load_images_from_folder(test_dir)

# Combine all images and labels for clustering
all_images = np.array(train_images + test_images)
all_labels = np.array(train_labels + test_labels)

# Reduce dimensionality using PCA for better clustering
pca = PCA(n_components=50)
image_features = pca.fit_transform(all_images)

# Apply K-means clustering
num_clusters = 2
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
kmeans.fit(image_features)

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