import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow.keras.callbacks import ReduceLROnPlateau
import time

# Path Definition
base_dir = '/Users/lukassspazo/Year 3 Python/AIoT/Tutorials/AutoencoderTRY/CatsDogsDataset'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

# Image data generator for loading and augmenting images
datagen = ImageDataGenerator(rescale=1.0/255)

# Loading datasets
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='input',
    shuffle=True
)

test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='input',
    shuffle=False
)

# Build Autoencoder Model
def build_autoencoder():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(128, 128, 3)))
    model.add(layers.MaxPooling2D((2, 2), padding='same'))
    model.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2), padding='same'))
    model.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(layers.UpSampling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.UpSampling2D((2, 2)))
    model.add(layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same'))  # Output layer
    return model

learning_rate = 0.001
autoencoder = build_autoencoder()
autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='binary_crossentropy')

def calculate_metrics(original, reconstructed):
    # Calculate Mean Squared Error for each image
    errors = np.mean((original - reconstructed) ** 2, axis=(1, 2, 3))  # MSE for each image
    
    # Calculate mean and standard deviation of errors
    mean_error = np.mean(errors)
    std_error = np.std(errors)

    # Classify images based on errors using the original threshold
    y_true = (errors > (mean_error + std_error)).astype(int)
    y_pred = (errors > (mean_error + std_error)).astype(int)

    accuracy = np.mean(y_true)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return accuracy, precision, recall, f1

# Training Callback to Calculate Metrics
class MetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data):
        super(MetricsCallback, self).__init__()
        self.validation_data = validation_data
        self.precisions = []
        self.recalls = []
        self.f1_scores = []

    def on_epoch_end(self, epoch, logs=None):
        # Get a batch of validation images
        images = self.validation_data
        reconstructed = self.model.predict(images)
        accuracy, precision, recall, f1 = calculate_metrics(images, reconstructed)
        self.precisions.append(precision)
        self.recalls.append(recall)
        self.f1_scores.append(f1)

# Train the Autoencoder
steps_per_epoch = int(np.ceil(train_generator.samples / train_generator.batch_size))
validation_steps = int(np.ceil(test_generator.samples / test_generator.batch_size))
lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

start_time_autoencoder = time.time()

metrics_callback = MetricsCallback(validation_data=next(test_generator)[0])
history = autoencoder.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=50,
    validation_data=test_generator,
    validation_steps=validation_steps,
    callbacks=[lr_reducer, metrics_callback]
)
end_time_autoencoder = time.time()
autoencoder_training_time = end_time_autoencoder - start_time_autoencoder

# Display Reconstructed Images and Metrics
def display_reconstructed_images(model, data_generator):
    n = 10  
    images = next(data_generator)[0]  
    reconstructed = model.predict(images)

    plt.figure(figsize=(20, 4))
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(images[i])
        plt.axis('off')

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(reconstructed[i])
        plt.axis('off')

    plt.show()

    accuracy, precision, recall, f1 = calculate_metrics(images, reconstructed)
    print(f"Autoencoder Metrics:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

    return accuracy, precision, recall, f1

autoencoder_metrics = display_reconstructed_images(autoencoder, test_generator)

# Build Classifier Model
def build_classifier(autoencoder):
    model = models.Sequential()
    model.add(autoencoder.layers[0])  # Conv2D Layer
    model.add(autoencoder.layers[1])  # MaxPooling2D Layer
    model.add(autoencoder.layers[2])  # Conv2D Layer
    model.add(autoencoder.layers[3])  # MaxPooling2D Layer
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))  # FC layer
    model.add(layers.Dropout(0.5))  # Dropout
    model.add(layers.Dense(1, activation='sigmoid'))  # Output layer
    return model

# Create and Compile Classifier
classifier = build_classifier(autoencoder)
classifier.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
                   loss='binary_crossentropy', 
                   metrics=['accuracy'])

# Prepare Data Generators for Classification
train_generator_classification = datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=64,
    class_mode='binary',  
    shuffle=True
)

test_generator_classification = datagen.flow_from_directory(
    test_dir,
    target_size=(128, 128),
    batch_size=64,
    class_mode='binary',  
    shuffle=False
)

steps_per_epoch_classification = int(np.ceil(train_generator_classification.samples / train_generator_classification.batch_size))
validation_steps_classification = int(np.ceil(test_generator_classification.samples / test_generator_classification.batch_size))

# Training Callback to Calculate Classifier Metrics
class ClassifierMetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data):
        super(ClassifierMetricsCallback, self).__init__()
        self.validation_data = validation_data
        self.precisions = []
        self.recalls = []
        self.f1_scores = []

    def on_epoch_end(self, epoch, logs=None):
        y_true = []
        y_pred = []
        for _ in range(validation_steps_classification):
            x_batch, y_batch = next(self.validation_data)
            preds = self.model.predict(x_batch)
            y_true.extend(y_batch)
            y_pred.extend((preds > 0.5).astype(int))

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        self.precisions.append(precision)
        self.recalls.append(recall)
        self.f1_scores.append(f1)

# Train Classifier
start_time_classifier = time.time()

classifier_metrics_callback = ClassifierMetricsCallback(validation_data=test_generator_classification)
history_classification = classifier.fit(
    train_generator_classification,
    steps_per_epoch=steps_per_epoch_classification,
    epochs=50,
    validation_data=test_generator_classification,
    validation_steps=validation_steps_classification,
    callbacks=[lr_reducer, classifier_metrics_callback] 
)
end_time_classifier = time.time()
classifier_training_time = end_time_classifier - start_time_classifier

# Evaluate Classifier and Display Metrics
def evaluate_classifier(classifier, test_generator):
    loss, accuracy = classifier.evaluate(test_generator)
    print(f"Classifier Test Loss: {loss:.4f}")
    print(f"Classifier Test Accuracy: {accuracy:.4f}")

    # Calculate additional metrics
    y_true = []
    y_pred = []
    for _ in range(validation_steps_classification):
        x_batch, y_batch = next(test_generator)
        preds = classifier.predict(x_batch)
        y_true.extend(y_batch)
        y_pred.extend((preds > 0.5).astype(int))

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"Classifier Metrics:")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

    return accuracy, precision, recall, f1

classifier_metrics = evaluate_classifier(classifier, test_generator_classification)

# Plotting Metrics for Autoencoder and Classifier
plt.figure(figsize=(15, 15))

# Autoencoder Loss
plt.subplot(4, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Autoencoder Loss per Epoch')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.grid()

# Classifier Loss
plt.subplot(4, 2, 2)
plt.plot(history_classification.history['loss'], label='Training Loss')
plt.plot(history_classification.history['val_loss'], label='Validation Loss')
plt.title('Classifier Loss per Epoch')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.grid()

# Autoencoder Metrics
plt.subplot(4, 2, 3)
plt.plot(metrics_callback.precisions, label='Precision')
plt.plot(metrics_callback.recalls, label='Recall')
plt.plot(metrics_callback.f1_scores, label='F1 Score')
plt.title('Autoencoder Metrics per Epoch')
plt.ylabel('Score')
plt.xlabel('Epoch')
plt.legend()
plt.grid()

# Classifier Metrics
plt.subplot(4, 2, 4)
plt.plot(classifier_metrics_callback.precisions, label='Precision')
plt.plot(classifier_metrics_callback.recalls, label='Recall')
plt.plot(classifier_metrics_callback.f1_scores, label='F1 Score')
plt.title('Classifier Metrics per Epoch')
plt.ylabel('Score')
plt.xlabel('Epoch')
plt.legend()
plt.grid()

# Show Running Times
total_training_time = autoencoder_training_time + classifier_training_time
plt.figtext(0.5, 0.01, f'Autoencoder Training Time: {autoencoder_training_time:.2f} seconds\n'
                       f'Classifier Training Time: {classifier_training_time:.2f} seconds\n'
                       f'Total Training Time: {total_training_time:.2f} seconds',
            ha='center', fontsize=12)

plt.tight_layout()
plt.show()

# Save Autoencoder Loss Plot
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Autoencoder Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.savefig('autoencoder_loss_per_epoch.png')
plt.close()

# Save Classifier Metrics Plot
plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.plot(history_classification.history['loss'], label='Training Loss')
plt.plot(history_classification.history['val_loss'], label='Validation Loss')
plt.title('Classifier Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(classifier_metrics_callback.precisions, label='Precision')
plt.plot(classifier_metrics_callback.recalls, label='Recall')
plt.plot(classifier_metrics_callback.f1_scores, label='F1 Score')
plt.title('Classifier Metrics per Epoch')
plt.ylabel('Score')
plt.xlabel('Epoch')
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig('classifier_metrics_per_epoch.png')
plt.close()