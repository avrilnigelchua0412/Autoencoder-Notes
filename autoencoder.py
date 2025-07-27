from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, LeakyReLU, Flatten, Dense
from tensorflow.python.keras import backend as K
class Autoencoder:
    def __init__(self, input_shape, latent_space_dim, **kwargs):
        self.input_shape = input_shape
        
        self.kwargs = kwargs # filters, kernel, stride, etc.
        
        self.latent_space_dim = latent_space_dim
        
        self.encoder = None
        self.decoder = None
        self.model = None
        
        self._shape_before_bottleneck = None
    
        self.build_model()
    
    def summary(self):
        self.encoder.summary()
    
    def build_model(self):
        self._build_encoder()
        # self._build_decoder()
        # self._build_autoencoder()
        
    def _build_encoder(self):
        encoder_input = self._add_encoder_input_layer()
        conv_layers = self._add_encoder_conv_layers(encoder_input)
        bottleneck = self._add_bottleneck_layer(conv_layers)
        self.encoder = Model(inputs=encoder_input, outputs=bottleneck, name='encoder')
        
    def _add_encoder_input_layer(self):
        return Input(shape=self.input_shape, name='encoder_input')
    
    def _add_encoder_conv_layers(self, encoder_input):
        x = encoder_input
        conv_config = self.kwargs.get('conv_layers_config', [])
        num_conv_layers = self.kwargs.get('num_conv_layers', len(conv_config))
        for layer_index in range(num_conv_layers):
            layer_params = conv_config[layer_index]
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
    
    def _add_bottleneck_layer(self, x):
        self._shape_before_bottleneck = K.int_shape(x) # Saves the shape of the feature maps before the bottleneck layer.
        x = Flatten(name='encoder_flatten_layer')(x) # Flattens the feature maps into a vector.
        x = Dense(self.latent_space_dim, name='encoder_bottleneck_layer')(x) # Dense layer to create the bottleneck, reducing the dimensionality to the latent space.
        return x
    
if __name__ == "__main__":
    input_shape = (64, 64, 3)  # Example input shape
    latent_space_dim = 128  # Example latent space dimension
    autoencoder = Autoencoder(input_shape, latent_space_dim, conv_layers_config=[
        {'filters': 32, 'kernel_size': (3, 3), 'strides': (2, 2)},
        {'filters': 64, 'kernel_size': (3, 3), 'strides': (2, 2)}
    ])
    autoencoder.summary()
    print("Encoder model built successfully.")