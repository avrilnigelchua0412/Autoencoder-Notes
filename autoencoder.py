import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, LeakyReLU, Flatten, Dense, Reshape, Conv2DTranspose, ReLU, Activation, Lambda, Layer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError, Loss
from tensorflow.keras import backend as K
import numpy as np
import keras
from abc import ABC, abstractmethod
import numpy as np

class Autoencoder(ABC):
    def __init__(self, latent_space_dim, decoder_out_filter, **kwargs):
        self.kwargs = kwargs # filters, kernel, stride, etc.
        self.latent_space_dim = latent_space_dim
        self.decoder_out_filter = decoder_out_filter
        self._default()
        
    def _default(self):
        self.encoder = None
        self.decoder = None
        self.model = None
        self._model_input = None
        self.conv_config = None
        self.num_conv_layers = None
        
        # This will hold the shape of the feature maps before the bottleneck layer.
        self._shape_before_bottleneck = None
    
    def _build(self, inputs_shape):
        encoder_output = self._build_encoder(inputs_shape)
        decoder_output = self._build_decoder(encoder_output)
        self.model = Model(inputs=self._model_input, outputs=decoder_output, name='autoencoder')
        return self.model
    
    def _build_decoder(self, encoder_output):
        decoder_input = encoder_output  # Use the encoder output as the decoder input.
        dense_layer = self._add_dense_layer(encoder_output)
        reshaped_layer = self._add_reshape_layer(dense_layer)
        conv_transpose_layers = self._add_conv_transpose_layers(reshaped_layer)
        decorder_output = self._add_decoder_output(conv_transpose_layers)
        self.decoder = Model(inputs=decoder_input, outputs=decorder_output, name='decoder')
        return decorder_output
        
    def _add_decoder_input_layer(self):
        return Input(shape=(self.latent_space_dim,), name='decoder_input')
    
    def _add_dense_layer(self, decoder_input):
        num_neurons = np.prod(self._shape_before_bottleneck)  # Number of neurons in the dense layer is the product of the dimensions before the bottleneck.
        dense_layer = Dense(
            num_neurons,
            name='decoder_dense_layer'
        )(decoder_input)
        return dense_layer
    
    def _add_reshape_layer(self, dense_layer):
        return Reshape(self._shape_before_bottleneck, name='decoder_reshape_layer')(dense_layer)
    
    def _add_conv_transpose_layers(self, x):
        """add convolutional transpose layers to the decoder"""
        # loop through the number of conv layers in reverse order
        for layer_index in reversed(range(1, self.num_conv_layers)):
            layer_params = self.conv_config[layer_index]
            x = self.add_conv_transpose_layer(layer_index, x, layer_params)
        return x
        
    def add_conv_transpose_layer(self, layer_index, x, layer_params):
        layer_num = self.num_conv_layers - layer_index
        # add a Conv2DTranspose layer with the parameters from the config
        try:
            filters = layer_params['filters']
            kernel_size = layer_params['kernel_size']
            strides = layer_params['strides']
        except KeyError as e:
            raise ValueError(
                f"Missing parameter {e} in conv transpose layer configuration for layer {layer_index}"
                )
        conv_transpose_layer = Conv2DTranspose(filters=filters,
                   kernel_size=kernel_size,
                   strides=strides, padding='same',
                   use_bias=False,
                   name=f'decoder_conv_transpose_layer_{layer_num}')
        x = conv_transpose_layer(x)
        x = BatchNormalization(name=f'decoder_batch_norm_layer_{layer_num}')(x)  # Normalizes the activations.
        x = ReLU(name=f'decoder_relu_layer_{layer_num}')(x)  # Applies the non-linearity.
        return x
    
    def _add_decoder_output(self, x):
        layer_params = self.conv_config[0]  # The first layer's parameters are used for the output layer.
        filters = self.decoder_out_filter # 1 grayscale output image, or 3 RGB output image.
        kernel_size = layer_params['kernel_size']
        strides = layer_params['strides']
        x = Conv2DTranspose(filters=filters,
                   kernel_size=kernel_size,
                   strides=strides, padding='same',
                   use_bias=False,
                   name=f'decoder_conv_transpose_layer_{self.num_conv_layers}')(x)
        output_layer = Activation('sigmoid', name='decoder_output_activation')(x)  # Sigmoid activation for the output layer.
        return output_layer
    
    def _build_encoder(self, inputs_shape):
        encoder_input = self._add_encoder_input_layer(inputs_shape)
        self._model_input = encoder_input
        conv_layers = self._add_encoder_conv_layers(encoder_input)
        bottleneck = self._add_bottleneck_layer(conv_layers)
        self.encoder = Model(inputs=encoder_input, outputs=bottleneck, name='encoder')
        return bottleneck
        
    def _add_encoder_input_layer(self, inputs_shape):
        return Input(shape=inputs_shape, name='encoder_input')
    
    def _add_encoder_conv_layers(self, encoder_input):
        x = encoder_input
        self.conv_config = self.kwargs.get('conv_layers_config', [])
        self.num_conv_layers = self.kwargs.get('num_conv_layers', len(self.conv_config))
        for layer_index in range(self.num_conv_layers):
            layer_params = self.conv_config[layer_index]
            x = self.add_conv_layer(layer_index, x, layer_params)
        return x
    
    def add_conv_layer(self, layer_index, x, layer_params):
        try:
            filters = layer_params['filters']
            kernel_size = layer_params['kernel_size']
            strides = layer_params['strides']
        except KeyError as e:
            raise ValueError(
                f"Missing parameter {e} in conv layer configuration for layer {layer_index}"
                )
        conv_layer = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   strides=strides, padding='same',
                   use_bias=False,
                   name=f'encoder_conv_layer_{layer_index + 1}')
        x = conv_layer(x) # Outputs feature maps (no bias because BatchNorm will normalize and shift them).
        x = BatchNormalization(name=f'encoder_batch_norm_layer_{layer_index + 1}')(x) # Normalizes the activations (mean=0, variance=1), and learns its own gamma (scale) and beta (shift) — replacing the need for bias.
        x = LeakyReLU(negative_slope=0.01, name=f'encoder_leaky_relu_layer_{layer_index + 1}')(x) # Applies the non-linearity after normalization. This improves gradient flow and prevents dying neurons.
        return x
    
    @abstractmethod
    def _add_bottleneck_layer(self, x):
        self._shape_before_bottleneck = K.int_shape(x)[1:] # Saves the shape of the feature maps before the bottleneck layer.
        x = Flatten(name='encoder_flatten_layer')(x) # Flattens the feature maps into a vector.
        return x  # This is an abstract method to be implemented by subclasses.