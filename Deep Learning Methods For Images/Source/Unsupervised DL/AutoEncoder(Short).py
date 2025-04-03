import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow.keras.callbacks import ReduceLROnPlateau

####### Setup #######

# Define paths to the dataset (update paths as needed)
base_dir = '../../dataset'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

# Image data generator for loading and augmenting images
datagen = ImageDataGenerator(rescale=1.0/255)

# Load training and testing datasets with target size of 28x28
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(28, 28),
    batch_size=32,
    class_mode='input',
    shuffle=True
)

test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(28, 28),
    batch_size=32,
    class_mode='input',
    shuffle=False
)

####### Build Autoencoder Model #######

def build_autoencoder():
    model = models.Sequential()
    
    # Encoder
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 3)))
    model.add(layers.MaxPooling2D((2, 2), padding='same'))

    model.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2), padding='same'))

    # Decoder
    model.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(layers.UpSampling2D((2, 2)))

    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.UpSampling2D((2, 2)))

    model.add(layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same'))  # Output layer

    return model

learning_rate = 0.001

autoencoder = build_autoencoder()
autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='binary_crossentropy')

####### Train the autoencoder #######

# Calculate steps_per_epoch and validation_steps
steps_per_epoch = int(np.ceil(train_generator.samples / train_generator.batch_size))
validation_steps = int(np.ceil(test_generator.samples / test_generator.batch_size))

# Learning Rate Scheduler
lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

# Train the autoencoder and capture history
history = autoencoder.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=50,
    validation_data=test_generator,
    validation_steps=validation_steps,
    callbacks=[lr_reducer]  # Add the learning rate scheduler callback
)

####### Plot Loss Graph #######

def plot_loss(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')  # Include validation loss
    plt.title('Model Loss Over Epochs')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid()
    plt.show()

# Call the plot_loss function to display the loss graph
plot_loss(history)

####### Metrics Calculation #######

def calculate_metrics(original, reconstructed, threshold=0.01):
    # Calculate reconstruction errors
    errors = np.mean((original - reconstructed) ** 2, axis=(1, 2, 3))  # MSE for each image
    y_true = (errors < threshold).astype(int)
    y_pred = (errors < threshold).astype(int)  # Predicted labels based on threshold

    # Calculate metrics
    accuracy = np.mean(y_true)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return accuracy, precision, recall, f1, errors

####### Display #######

def display_reconstructed_images(model, data_generator):
    n = 10  # Number of images to display
    images = next(data_generator)[0]  # Get a batch of images
    reconstructed = model.predict(images)

    plt.figure(figsize=(20, 4))
    for i in range(n):
        # Display original images
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(images[i])
        plt.axis('off')

        # Display reconstructed images
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(reconstructed[i])
        plt.axis('off')

    plt.show()

    # Calculate and display metrics
    accuracy, precision, recall, f1, errors = calculate_metrics(images, reconstructed)
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"Reconstruction Errors: {errors}")

# Display the results
display_reconstructed_images(autoencoder, test_generator)
