import os
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from scipy.stats import mode
from skimage import io, color
from skimage.feature import ORB

# Define dataset path
data_dir = "/Users/lukassspazo/Year 3 Python/Deep Learning for Images/dataset"
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")

# Function to extract ORB features
def extract_orb_features(folder):
    orb = ORB()
    descriptors_list = []
    labels = []
    for subfolder in os.listdir(folder):
        subfolder_path = os.path.join(folder, subfolder)
        if os.path.isdir(subfolder_path):
            label = 0 if subfolder == "cats" else 1  # Assign 0 for cats, 1 for dogs
            for filename in tqdm(os.listdir(subfolder_path), desc=f"Extracting ORB from {subfolder}"):
                img_path = os.path.join(subfolder_path, filename)
                img = io.imread(img_path)

                # Check if the image is grayscale and convert it to RGB if necessary
                if img.ndim == 2:  # Grayscale image
                    img_gray = img  # Already grayscale
                else:
                    img_gray = color.rgb2gray(img)  # Convert to grayscale

                orb.detect_and_extract(img_gray)  # Detect keypoints and extract descriptors
                descriptors = orb.descriptors
                if descriptors is not None:
                    descriptors_list.append(descriptors.flatten())  # Flatten descriptors
                    labels.append(label)
    return descriptors_list, labels

# Extract ORB features
train_features, train_labels = extract_orb_features(train_dir)
test_features, test_labels = extract_orb_features(test_dir)

# Combine all extracted features and labels for clustering
all_features = np.array(train_features + test_features, dtype=object)
all_labels = np.array(train_labels + test_labels)

# Ensure all features have the same dimensionality by padding/truncating
feature_dim = max(len(f) for f in all_features)
all_features = np.array([np.pad(f, (0, feature_dim - len(f)), mode='constant') if len(f) < feature_dim else f[:feature_dim] for f in all_features])

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