# Image Classification Models

This repository contains various image classification models that utilize different feature extraction techniques to classify images of cats and dogs. Each model demonstrates unique methodologies for extracting features and applying machine learning algorithms for classification.

## Overview

The primary goal of these models is to classify images into two categories: cats and dogs. The models differ mainly in their feature extraction methods, which significantly impact their performance.

## Feature Extraction Methods

1. **HOG (Histogram of Oriented Gradients)**: Extracts features based on the distribution of gradients in localized portions of an image.

2. **SIFT (Scale-Invariant Feature Transform)**: Identifies keypoints and descriptors that are invariant to scale and rotation.

3. **ORB (Oriented FAST and Rotated BRIEF)**: A fast alternative to SIFT for detecting keypoints and extracting descriptors.

4. **LBP (Local Binary Patterns)**: Captures texture information by comparing each pixel with its neighbors.

5. **Color Histograms**: Analyzes the frequency distribution of colors in an image.

6. **VGG16**: Utilizes a pre-trained deep learning model to extract high-level features from images.

7. **Img2Vec**: Leverages a deep learning model to generate vector representations of images.

8. **PCA (Principal Component Analysis)**: Reduces dimensionality of the dataset for clustering.

9. **K-Means Clustering**: Groups similar feature vectors into clusters for classification.

10. **Autoencoder**: A neural network architecture that learns to compress and reconstruct data.

## Models Included

- **Model 1**: HOG feature extraction with K-Means clustering
- **Model 2**: SIFT feature extraction with K-Means clustering
- **Model 3**: ORB feature extraction with K-Means clustering
- **Model 4**: LBP feature extraction with K-Means clustering
- **Model 5**: Color Histogram feature extraction with K-Means clustering
- **Model 6**: VGG16 feature extraction with classification
- **Model 7**: Img2Vec feature extraction with classification
- **Model 8**: PCA feature reduction with K-Means clustering
- **Model 9**: Autoencoder for feature extraction and classification

## Requirements

- Python 3.x
- Libraries:
    - NumPy
    - scikit-learn
    - TensorFlow or PyTorch (depending on the model)
    - OpenCV (if necessary)
    - Matplotlib
    - PIL (Pillow)

## Usage

1. Clone the repository:
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the desired model script. For example:
    ```bash
    python model1.py
    ```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Scikit-learn](https://scikit-learn.org/) for machine learning algorithms.
- [TensorFlow](https://www.tensorflow.org/) and [PyTorch](https://pytorch.org/) for deep learning frameworks.
- [OpenCV](https://opencv.org/) for image processing functionalities.

## Contact

For any inquiries or contributions, please contact [your_email@example.com].
