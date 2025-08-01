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
    def __init__(self, input_shape, latent_space_dim, decoder_out_filter, **kwargs):
        self.input_shape = input_shape
        self.kwargs = kwargs # filters, kernel, stride, etc.
        self.latent_space_dim = latent_space_dim
        self.decoder_out_filter = decoder_out_filter
        self._default()
        self.build_model()
        
    def _default(self):
        self.encoder = None
        self.decoder = None
        self.model = None
        self._model_input = None
        self.conv_config = None
        self.num_conv_layers = None
        # This will hold the shape of the feature maps before the bottleneck layer.
        self._shape_before_bottleneck = None
        
    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()
    
    @abstractmethod
    def compile(self, learning_rate=0.0001):
        """Compiles the autoencoder model with the specified optimizer and loss function."""
        if self.model is None:
            raise ValueError("Model has not been built yet. Call build_model() first.")
        
    def train(self, **kwargs):
        train_config = kwargs.get('train_config', {})
        x_train = train_config['x_train']
        y_train = train_config['y_train']
        batch_size = train_config['batch_size']
        epochs = train_config['epochs']
        validation_data = train_config.get('validation_data', None)
        shuffle = train_config.get('shuffle', True)
        self.model.fit(
            x=x_train,
            y=y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_data,
            shuffle=shuffle,
            verbose=1  # Set to 1 for progress bar
        )
        
    def model_save(self, filepath):
        """Saves the model to the specified filepath."""
        if self.model is None:
            raise ValueError("Model has not been built yet. Call build_model() first.")
        self.model.save("full_" + filepath)
        self.encoder.save("encoder_" + filepath)
        self.decoder.save("decoder_" + filepath)
    
    def build_model(self):
        # self._build_encoder()
        # self._build_decoder()
        self._build_autoencoder()
    
    @abstractmethod
    def _build_autoencoder(self):
        encoder_output = self._build_encoder()
        decoder_output = self._build_decoder(encoder_output)
        return decoder_output
    
    def _build_decoder(self, encoder_output):
        # decoder_input = self._add_decoder_input_layer()
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
    
    def _build_encoder(self):
        encoder_input = self._add_encoder_input_layer()
        self._model_input = encoder_input
        conv_layers = self._add_encoder_conv_layers(encoder_input)
        bottleneck = self._add_bottleneck_layer(conv_layers)
        self.encoder = Model(inputs=encoder_input, outputs=bottleneck, name='encoder')
        return bottleneck
        
    def _add_encoder_input_layer(self):
        return Input(shape=self.input_shape, name='encoder_input')
    
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

class VanillaAutoencoder(Autoencoder):
    def __init__(self, input_shape, latent_space_dim, decoder_out_filter, **kwargs):
        super().__init__(input_shape, latent_space_dim, decoder_out_filter, **kwargs)
        self._default()
        self.build_model()
    
    def _add_bottleneck_layer(self, x):
        x = super()._add_bottleneck_layer(x)  # Call the parent method to flatten the feature maps.
        x = Dense(self.latent_space_dim, name='encoder_bottleneck_layer')(x) # Dense layer to create the bottleneck, reducing the dimensionality to the latent space.
        return x
    
    def _build_autoencoder(self):
        decoder_output = super()._build_autoencoder()  # Call the parent method to build the decoder.
        self.model = Model(inputs=self._model_input, outputs=decoder_output, name='vanilla_autoencoder')
        
    def compile(self, learning_rate=0.0001):
        super().compile(learning_rate)  # Call the parent compile method.
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss=MeanSquaredError(),
            metrics=['mae', 'mse']  # both Keras shorthand
        )

class VariationalAutoencoder(Autoencoder):
    def __init__(self, input_shape, latent_space_dim, decoder_out_filter, **kwargs):
        super().__init__(input_shape, latent_space_dim, decoder_out_filter, **kwargs)
        self._default()
        self.build_model()
    
    def _add_bottleneck_layer(self, x):
        x = super()._add_bottleneck_layer(x)  # Call the parent method to flatten the feature maps.
        
        # Not Sequential, so we need to define the mu and log_variance layers separately.
        # These layers will output the parameters of the latent space distribution.
        self.mu = Dense(self.latent_space_dim, name='mu')(x) # Mean vector for the latent space.
        self.log_variance = Dense(self.latent_space_dim, name='log_variance')(x)  # Log variance vector for the latent space.
        
        # def sample_point_from_normal_distribution(args):
        #     mu, log_variance = args
        #     epsilon = K.random_normal(shape=K.shape(mu), mean=0., stddev=1.) # Explicitly sample from a standard normal distribution.
        #     sampled_point =  mu + K.exp(log_variance / 2) * epsilon  # Reparameterization trick: mu + sigma * epsilon, where sigma = exp(log_variance / 2).
        #     # This allows gradients to flow through the sampling process. 
        #     return sampled_point
        
        # x = Lambda(sample_point_from_normal_distribution, name='encoder_output')([self.mu, self.log_variance])  # Sampling layer to sample from the latent space distribution.
        
        x = Sampling(name='encoder_output')([self.mu, self.log_variance])  # Sampling layer to sample from the latent space distribution.
        return x
    
    def _build_autoencoder(self):
        decoder_output = super()._build_autoencoder()  # Call the parent method to build the decoder.
        self.model = Model(inputs=self._model_input, outputs=[decoder_output, self.mu, self.log_variance], name='variational_autoencoder')
    
    def compile(self, learning_rate=0.0001):
        super().compile(learning_rate)  # Call the parent compile method.
        loss_fn = VAELoss(recon_weight=1000.0, kl_weight=1.0)  # Custom VAE loss function.
        # The loss function will compute the reconstruction loss and KL divergence.
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss=loss_fn,  # Custom VAE loss function.
            metrics=[
                tf.keras.metrics.MeanMetricWrapper(
                    fn=lambda y_true, y_pred: loss_fn.reconstruction_loss(y_true, y_pred),
                    name="reconstruction_loss"
                ),
                tf.keras.metrics.MeanMetricWrapper(
                    fn=lambda y_true, y_pred: loss_fn.kl_loss(y_pred),
                    name="kl_divergence_loss"
                )
            ]  # Custom metrics for monitoring.
        )


class Sampling(Layer):
    @tf.function
    def call(self, inputs):
        mu, log_variance = inputs
        epsilon = tf.random.normal(shape=tf.shape(mu), mean=0., stddev=1.) # Explicitly sample from a standard normal distribution.
        sampled_point =  mu + tf.exp(log_variance / 2) * epsilon  # Reparameterization trick: mu + sigma * epsilon, where sigma = exp(log_variance / 2).
        # This allows gradients to flow through the sampling process. 
        return sampled_point

class VAELoss(tf.keras.losses.Loss):
    def __init__(self, recon_weight=1000.0, kl_weight=1.0, name="vae_loss"):
        super().__init__(name=name)
        self.recon_weight = recon_weight
        self.kl_weight = kl_weight

    def call(self, y_true, y_pred):
        y_recon, mu, log_variance = y_pred

        reconstruction_loss = self._calculate_reconstruction_loss(y_true, y_recon)
        kl_divergence_loss = self._calculate_Kullback_Leibler_divergence(log_variance, mu)
        
        # Add metrics for monitoring
        self.add_metric(tf.reduce_mean(reconstruction_loss), name="reconstruction_loss", aggregation="mean")
        self.add_metric(tf.reduce_mean(kl_divergence_loss), name="kl_loss", aggregation="mean")
        # [0.02, 0.03, 0.01, 0.04]
        # tf.reduce_mean(reconstruction_loss)  # = 0.025
        # self.add_metric(reconstruction_loss, name="reconstruction_loss") throws an error

        total_loss = self.recon_weight * reconstruction_loss + self.kl_weight * kl_divergence_loss
        return total_loss

    def _calculate_reconstruction_loss(self, y_true, y_recon):
        """Calculates the reconstruction loss (mean squared error per sample)."""
        error = y_true - y_recon
        reconstruction_loss = tf.reduce_mean(tf.square(error), axis=[1, 2, 3])  # shape: (batch_size,)
        return reconstruction_loss

    def _calculate_Kullback_Leibler_divergence(self, log_variance, mu):
        """Calculates the KL divergence loss per sample."""
        kl_divergence_loss = -0.5 * tf.reduce_sum(
            1 + log_variance - tf.square(mu) - tf.exp(log_variance), axis=1
        )  # shape: (batch_size,)
        return kl_divergence_loss
    
    def reconstruction_loss(self, y_true, y_pred):
        y_recon, _, _ = y_pred
        per_sample_loss = self._calculate_reconstruction_loss(y_true, y_recon)  # shape: (batch_size,)
        return tf.reduce_mean(per_sample_loss)

    def kl_loss(self, y_pred):
        _, mu, log_variance = y_pred
        per_sample_loss = self._calculate_Kullback_Leibler_divergence(log_variance, mu)  # shape: (batch_size,)
        return tf.reduce_mean(per_sample_loss)


if __name__ == "__main__":
    input_shape = (64, 64, 3)  # Example input shape
    latent_space_dim = 128  # Example latent space dimension
    decoder_out_filter = 3  # Example output filter for the decoder (e.g., 3 for RGB images)
    autoencoder = VariationalAutoencoder(input_shape, latent_space_dim, decoder_out_filter, conv_layers_config=[
        {'filters': 32, 'kernel_size': (3, 3), 'strides': (2, 2)},
        {'filters': 64, 'kernel_size': (3, 3), 'strides': (2, 2)},
        {'filters': 128, 'kernel_size': (3, 3), 'strides': (2, 2)}
    ])
    autoencoder.summary()
    print("Encoder model built successfully.")
    autoencoder.compile(learning_rate=0.001)
    print("Autoencoder compiled successfully.")