from autoencoder import Autoencoder
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError

class VanillaAutoencoder(Autoencoder, Model):
    def __init__(self, input_shape, latent_space_dim, decoder_out_filter, **kwargs):
        Model.__init__(self)  # Initialize the Model class
        Autoencoder.__init__(self, latent_space_dim, decoder_out_filter, **kwargs) # Initialize the Autoencoder class
        self._model = self._build(input_shape)
        
    def _add_bottleneck_layer(self, x):
        x = super()._add_bottleneck_layer(x)  # Call the parent method to flatten the feature maps.
        x = Dense(self.latent_space_dim, name='encoder_bottleneck_layer')(x) # Dense layer to create the bottleneck, reducing the dimensionality to the latent space.
        return x
    
    def call(self, inputs):
        return self._model(inputs)
        
    def compile(self, learning_rate=0.0001):
        super().compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss=MeanSquaredError(),
            metrics=['mae']
        )
        
    def summary(self, *args, **kwargs):
            return self._model.summary(*args, **kwargs)
        
    def save_all(self):
        self._model.save('full_vanilla_autoencoder_model.keras')
        self.encoder.save('encoder_vanilla_autoencoder_model.keras')
        self.decoder.save('decoder_vanilla_autoencoder_model.keras')
        
if __name__ == "__main__":
    input_shape = (64, 64, 3)  # Example input shape
    latent_space_dim = 128  # Example latent space dimension
    decoder_out_filter = 3  # Example output filter for the decoder (e.g., 3 for RGB images)
    autoencoder = VanillaAutoencoder(input_shape, latent_space_dim, decoder_out_filter, conv_layers_config=[
        {'filters': 32, 'kernel_size': (3, 3), 'strides': (2, 2)},
        {'filters': 64, 'kernel_size': (3, 3), 'strides': (2, 2)},
        {'filters': 128, 'kernel_size': (3, 3), 'strides': (2, 2)}
    ])
    autoencoder.compile(learning_rate=0.001)
    print("Autoencoder compiled successfully.")
    autoencoder.summary()
    print("Encoder model built successfully.")