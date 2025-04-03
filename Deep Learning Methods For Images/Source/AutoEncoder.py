import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization , Flatten, Dense, Reshape, Conv2DTranspose, Activation
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.losses import MeanSquaredError


# Step 1: Setup  

def load_cats_dogs_dataset(base_dir):
    # Define paths to the dataset
    train_dir = os.path.join(base_dir, 'train')
    test_dir = os.path.join(base_dir, 'test')

# Step 2: Data Preprocessing  
# Initialize ImageDataGenerator for training data  

    # Image data generator for loading images
    datagen = ImageDataGenerator(rescale=1./255)  # Normalize pixel values to [0, 1]

 # Load the training data  
    train_generator = datagen.flow_from_directory(
        train_dir,
        target_size=(28, 28), 
        color_mode='grayscale',  
        class_mode='binary',  # Assuming binary classification (cats vs dogs)
        batch_size=32,
        shuffle=True,
        seed=42
    )

# Load the testing data 
    test_generator = datagen.flow_from_directory(
        test_dir,
        target_size=(28, 28),
        color_mode='grayscale',
        class_mode='binary',
        batch_size=32,
        shuffle=False,
        seed=42
    )

    # Convert generators to numpy arrays
    X_train, Y_train = next(train_generator)  # Get one batch of training data
    X_test, Y_test = next(test_generator)    # Get one batch of test data

    return X_train, Y_train, X_test, Y_test


# Step 3: Define Autoencoder Architecture 
#Stablish the Constructor of the Autoencoder

class Autoencoder:
    def __init__(self,
                 input_shape,
                 convolution_filters,
                 convolution_kernels,
                 convolution_strides,
                 latent_space_dims):
        self.input_shape = input_shape # [28, 28, 1]
        self.convolution_filters = convolution_filters # [2, 4, 8]
        self.convolution_kernels = convolution_kernels # [3, 5, 3]
        self.convolution_strides = convolution_strides # [1, 2, 2]  2 Basically halfs which means w eare downsampling the data at that point in the architecture
        self.latent_space_dims = latent_space_dims # 2

        self.encoder = None
        self.decoder = None
        self.model = None
        

        self._num_convolution_layers = len(convolution_filters)  # Number of Convolutional layers
        self._shape_before_bottleneck = None
        self._model_input = None

        self._build()
    
    def summary(self): #Summary of Autoencoder Architecture
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()

    def _build(self):  #Builidng the encoder/decoder architecture
        self._build_encoder()
        self._build_decoder()
        self._build_autoencoder()

#### ENCODER ####

    def _build_encoder(self): #Basically building a CNN for encoding
        encoder_input = self._add_encoder_input()
        conv_layers = self._add_convolution_layers(encoder_input)
        bottleneck = self._add_bottleneck(conv_layers)
        self._model_input = encoder_input # Model Input for the Autoencoder
        self.encoder = Model(encoder_input, bottleneck, name ="encoder")

    def _add_encoder_input(self): # We want to create an input layer, using the shape of the previously 
        return Input(shape= self.input_shape, name = "encoder_input")
    
    def _add_convolution_layers(self, encoder_input):
        # Creates all convolutional blocks in the encoder
        x = encoder_input 
        for layer_index in range(self._num_convolution_layers): #We get as many steps as the number of conv layers
            x = self._add_convolution_layer(layer_index, x)
        return x
    
    def _add_convolution_layer(self, layer_index, x):
    # Adds a convolution block to a graph of layers;
    # Consisting of: 2D Convolution + ReLU + batch normalization.
        layer_number = layer_index + 1
        convolution_layer = Conv2D(filters=self.convolution_filters[layer_index],  # Fixed here
                               kernel_size=self.convolution_kernels[layer_index],
                               strides=self.convolution_strides[layer_index],
                               padding="same",
                               name=f"encoder_convolution_layer_{layer_number}"
                               )
        x = convolution_layer(x)  # Applying keras layer to graph of layers by using (x)
        x = ReLU(name=f"encoder_ReLU_{layer_number}")(x)
        x = BatchNormalization(name=f"encoder_BN_{layer_number}")(x)
        return x  # Don't forget to return x

    def _add_bottleneck(self, x):
        # Flatten Data and add Bottleneck ( Dense layer )
        self._shape_before_bottleneck = K.int_shape(x)[1:] # 4 dimensional array, for instance [ batch_size, width, height, num_channels ]. ç
        # we slice the batch_size out since we don´t need it here
        x = Flatten()(x)
        x = Dense(self.latent_space_dims, name= "encoder_output")(x)
        return x
    
#### DECODER ####

    def _build_decoder(self):
        decoder_input = self._add_decoder_input()
        dense_layer = self._add_dense_layer(decoder_input)
        reshape_layer = self._add_reshape_layer(dense_layer)
        convolution_transponse_layers = self._add_convolution_transponse_layers(reshape_layer)
        decoder_output = self._add_decoder_output(convolution_transponse_layers)
        self.decoder = Model(decoder_input, decoder_output, name="decoder")

    def _add_decoder_input(self):
        return Input(shape=(self.latent_space_dims,), name="decoder_input")

    def _add_dense_layer(self, decoder_input):
        num_neurons = np.prod(self._shape_before_bottleneck)  # [4, 4, 32] --> 4x4x32 = 512 Neurons
        dense_layer = Dense(num_neurons, name="decoder_dense")(decoder_input)  # Remember to apply the instanciated dense layer to the graph 
        return dense_layer

    def _add_reshape_layer(self, dense_layer):
        reshape_layer = Reshape(self._shape_before_bottleneck)(dense_layer)
        return reshape_layer

    def _add_convolution_transponse_layers(self, x):  # Add Convolutional Transpose Blocks = SUM[CNN + ReLU + BN]. 
        # Loop through all conv layers in reverse order
        for layer_index in reversed(range(self._num_convolution_layers)):
            # Objective is to [0, 1, 2] ----> [2, 1, 0] 
            # Beyond this, we want to get rid of the first index, that's why we use the range(1, num)
            x = self._add_convolution_transponse_layer(layer_index, x)  
        return x  # Make sure that this is out of the loop to avoid early stopping 

    def _add_convolution_transponse_layer(self, layer_index, x):
        layer_number = self._num_convolution_layers - layer_index  # Different NUMS, similar names 
        convolution_transponse_layer = Conv2DTranspose(
            filters=self.convolution_filters[layer_index],
            kernel_size=self.convolution_kernels[layer_index],
            strides=self.convolution_strides[layer_index],
            padding="same",
            name=f"decoder_convolution_transponse_layer_{layer_number}"
        )
        x = convolution_transponse_layer(x)
        x = ReLU(name=f"decoder_relu{layer_number}")(x)              
        x = BatchNormalization(name=f"decoder_BN_{layer_number}")(x)  # Building the Convolutional Transpose Block
        return x

    def _add_decoder_output(self, x):
        convolution_transponse_layer = Conv2DTranspose(
            filters=1,  # Final output channels
            kernel_size=self.convolution_kernels[0], 
            strides=self.convolution_strides[0],
            padding="same",
            name=f"decoder_output_{self._num_convolution_layers}"
        )
        x = convolution_transponse_layer(x)
        output_layer = Activation("sigmoid", name="sigmoid_layer")(x)  # Remember to apply the layer to the graph by using (x)
        return output_layer
    
#### AUTOENCODER ####

    def _build_autoencoder(self):
        model_input = self._model_input
        model_output = self.decoder(self.encoder(model_input))
        self.model = Model(model_input, model_output, name = "autoencoder")


# Step 4: Compile the Model  
# Compile the model with optimizer and loss function  

    def compile(self, learning_rate= 0.0001):
        optimizer = Adam(learning_rate=learning_rate)
        mse_loss = MeanSquaredError()
        self.model.compile(optimizer=optimizer, loss = mse_loss)    # Since the Autoencoder is a Keras model, we can naturally use the .compile method

# Step 5: Train the Autoencoder  
# Fit the model on training data  
    def train(self, X_train,batch_size, num_epochs):
        self.model.fit(X_train,
                       X_train, 
                       batch_size= batch_size, 
                       epochs= num_epochs,
                       shuffle = True
                       )
        
if __name__ ==  "__main__":
    autoencoder = Autoencoder(
        input_shape= (28 ,28 ,1),
        convolution_filters= (32, 64, 64, 64),
        convolution_kernels=(3, 3, 3, 3),
        convolution_strides= (1, 2, 2, 1),
        latent_space_dims= 2
        
    )
    autoencoder.summary()
