import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from skimage import feature
from img2vec_pytorch import Img2Vec
from torchvision import models, transforms
from PIL import Image
import torch

# Base class for feature extraction
class _FeatureExtraction:
    def __init__(self, categories):
        self.categories = categories
        self.train_features = []
        self.train_lables = []
        self.test_features = []
        self.test_lables = []
        self.train_dir = '/Users/lukassspazo/Year 3 Python/Deep Learning for Images/dataset/train'
        self.test_dir = '/Users/lukassspazo/Year 3 Python/Deep Learning for Images/dataset/test'

from skimage.feature import ORB

class SIFTExtraction(_FeatureExtraction):
    def __init__(self, categories):
        super().__init__(categories)
        self.sift = ORB()  # Using ORB instead of SIFT

    def _store_feature(self, data_path, curr_label, features, labels):
        for img in os.listdir(data_path):
            image_path = os.path.join(data_path, img)
            image = Image.open(image_path).convert('L')  # Convert to grayscale
            image = image.resize((64, 64))  # Resize image
            image_np = np.array(image)

            # Detect keypoints and extract descriptors
            self.sift.detect_and_extract(image_np)
            keypoints = self.sift.keypoints
            descriptors = self.sift.descriptors

            # Ensure descriptors are not empty
            if descriptors is not None and descriptors.shape[0] > 0:
                features.append(descriptors.flatten())  # Flatten to ensure consistent shape
                labels.append(curr_label)
            else:
                print(f"No descriptors found for image: {img}")  # Log missing descriptors

    def feature_extraction(self):
        for category in self.categories:
            train_data_path = os.path.join(self.train_dir, category)
            test_data_path = os.path.join(self.test_dir, category)
            self._store_feature(train_data_path, category, self.train_features, self.train_lables)
            self._store_feature(test_data_path, category, self.test_features, self.test_lables)

class LBPExtraction(_FeatureExtraction):
    def __init__(self, categories):
        super().__init__(categories)
    
    def _store_features_and_labels(self, label, data_path, features, labels):
        radius = 1
        n_points = 8 * radius
        for img in os.listdir(data_path):
            image_path = os.path.join(data_path, img)
            img = Image.open(image_path).convert('L')  # Read as grayscale
            img = img.resize((64, 64))  # Resize image
            img_np = np.array(img)
            lbp = feature.local_binary_pattern(img_np, n_points, radius, method='uniform')
            hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
            hist = hist.astype("float") / hist.sum()  # Normalize
            features.append(hist)
            labels.append(label)

    def feature_extraction(self):
        for category in self.categories:
            train_data_path = os.path.join(self.train_dir, category)
            test_data_path = os.path.join(self.test_dir, category)
            self._store_features_and_labels(category, train_data_path, self.train_features, self.train_lables)
            self._store_features_and_labels(category, test_data_path, self.test_features, self.test_lables)

class HOGExtraction(_FeatureExtraction):
    def _store_features_and_labels(self, label, data_path, features, labels):
        for img in os.listdir(data_path):
            image_path = os.path.join(data_path, img)
            img = Image.open(image_path).convert('L')  # Read as grayscale
            img = img.resize((64, 64))  # Resize image
            img_np = np.array(img)
            img_hog = feature.hog(img_np, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False, feature_vector=True)
            features.append(img_hog)
            labels.append(label)

    def feature_extraction(self):
        for category in self.categories:
            train_data_path = os.path.join(self.train_dir, category)
            test_data_path = os.path.join(self.test_dir, category)
            self._store_features_and_labels(category, train_data_path, self.train_features, self.train_lables)
            self._store_features_and_labels(category, test_data_path, self.test_features, self.test_lables)

class VGG16Extraction(_FeatureExtraction):
    def __init__(self, categories):
        super().__init__(categories)
        self.model = models.vgg16(pretrained=True)
        self.model.classifier = torch.nn.Identity()  # Remove the classification layer
        self.model.eval()  # Set the model to evaluation mode

    def _preprocess_image(self, img_path):
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')  # Convert to RGB
        img = transform(img).unsqueeze(0)  # Add batch dimension
        return img

    def _extract_features(self, img_path):
        img_tensor = self._preprocess_image(img_path)
        with torch.no_grad():
            features = self.model(img_tensor)
        return features.numpy().flatten()  # Flatten the features

    def _store_features_and_labels(self, label, data_path, features, labels):
        for img in os.listdir(data_path):
            image_path = os.path.join(data_path, img)
            extracted_features = self._extract_features(image_path)
            features.append(extracted_features)
            labels.append(label)

    def feature_extraction(self):
        for category in self.categories:
            train_data_path = os.path.join(self.train_dir, category)
            test_data_path = os.path.join(self.test_dir, category)
            self._store_features_and_labels(category, train_data_path, self.train_features, self.train_lables)
            self._store_features_and_labels(category, test_data_path, self.test_features, self.test_lables)

class Img2VecExtraction(_FeatureExtraction):
    def __init__(self, categories):
        super().__init__(categories)
        self.img2vec = Img2Vec(model="resnet50")

    def _is_grey_scale_image(self, image):
        return image.mode == 'L'
    
    def _to_three_channel_image(self, image):
        return image.convert('RGB')

    def _store_features_and_labels(self, label, data_path, feature_array, labels_array):
        for img in os.listdir(data_path):
            train_image_path = os.path.join(data_path, img)
            train_img = Image.open(train_image_path)
            if self._is_grey_scale_image(train_img):
                train_img = self._to_three_channel_image(train_img)
            train_img_features = self.img2vec.get_vec(train_img)
            feature_array.append(train_img_features)
            labels_array.append(label)

    def feature_extraction(self):
        for category in self.categories:
            train_data_path = os.path.join(self.train_dir, category)
            test_data_path = os.path.join(self.test_dir, category)
            self._store_features_and_labels(category, train_data_path, self.train_features, self.train_lables)
            self._store_features_and_labels(category, test_data_path, self.test_features, self.test_lables)

# Function to pad features to ensure consistent shape
def pad_features(features, max_length):
    padded_features = []
    for feature in features:
        if feature.size < max_length:
            padded = np.pad(feature, (0, max_length - feature.size), mode='constant')
            padded_features.append(padded)
        else:
            padded_features.append(feature)
    return np.array(padded_features)

# Main classification logic
categories = ['cats', 'dogs']
extractors = {
    "Img2Vec": Img2VecExtraction(categories),
    "SIFT": SIFTExtraction(categories),
    "HOG": HOGExtraction(categories),
    "LBP": LBPExtraction(categories),
    "VGG16": VGG16Extraction(categories),
}

# Extract features for all methods
for name, extractor in extractors.items():
    print(f"Extracting features using {name}...")
    extractor.feature_extraction()

# Random Forest classification function
def random_forest(train_features, train_labels, test_features, test_labels):
    metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }
    tree_numbers = [10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    
    for n_trees in tree_numbers:
        model = RandomForestClassifier(n_estimators=n_trees, random_state=42)
        model.fit(train_features, train_labels)
        y_pred = model.predict(test_features)

        metrics['accuracy'].append(accuracy_score(test_labels, y_pred))
        metrics['f1'].append(f1_score(test_labels, y_pred, pos_label='cats', average='binary'))
        metrics['precision'].append(precision_score(test_labels, y_pred, pos_label='cats', average='binary'))
        metrics['recall'].append(recall_score(test_labels, y_pred, pos_label='cats', average='binary'))

        print(f"Tree Number: {n_trees}, Accuracy: {metrics['accuracy'][-1]:.4f}, "
              f"Precision: {metrics['precision'][-1]:.4f}, Recall: {metrics['recall'][-1]:.4f}, "
              f"F1 Score: {metrics['f1'][-1]:.4f}")

    return metrics

# Evaluate and test each feature extraction method
metrics_dict = {}
for name, extractor in extractors.items():
    print(f"\nEvaluating model trained using {name}...")
    max_length = max(len(f) for f in extractor.train_features if f.size > 0)  # Find max length for padding
    train_features_padded = pad_features(extractor.train_features, max_length)
    test_features_padded = pad_features(extractor.test_features, max_length)

    metrics = random_forest(
        train_features_padded,
        extractor.train_lables,
        test_features_padded,
        extractor.test_lables
    )
    metrics_dict[name] = metrics

    # Final model training and testing
    model = RandomForestClassifier()
    model.fit(train_features_padded, extractor.train_lables)
    y_pred = model.predict(test_features_padded)

    # Print classification report
    print(f"Model trained with {name} - Accuracy: {accuracy_score(extractor.test_lables, y_pred):.4f}")
    print(classification_report(extractor.test_lables, y_pred))

# Plot Accuracy and F1 Score
def plot_metrics(metrics_dict):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    x_custom = [10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

    # Plot accuracy
    for method, metrics in metrics_dict.items():
        axs[0].plot(x_custom, metrics['accuracy'], marker='o', label=f'{method} Accuracy')
    axs[0].set_title("Accuracy vs No. of Trees")
    axs[0].set_xlabel("No. of Trees")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend()

    # Plot F1 Score
    for method, metrics in metrics_dict.items():
        axs[1].plot(x_custom, metrics['f1'], marker='o', label=f'{method} F1 Score')
    axs[1].set_title("F1 Score vs No. of Trees")
    axs[1].set_xlabel("No. of Trees")
    axs[1].set_ylabel("F1 Score")
    axs[1].legend()

    plt.tight_layout()
    plt.show()

# Call the plotting function
plot_metrics(metrics_dict)
