import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization,LeakyReLU, Flatten, Dense, Reshape, Conv2DTranspose, ReLU, Activation, Lambda, Layer
import numpy as np    
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError

class DecoderBuilder(Model):
    def __init__(self, shape_before_bottleneck, decoder_out_filter, conv_config=None, **kwargs):
        super().__init__(**kwargs)
        self.conv_config = conv_config
        self._shape_before_bottleneck = shape_before_bottleneck
        self.decoder_out_filter = decoder_out_filter
        self.dense_layer = None
        self.reshaped_layer = None
        self.convT_layers = []
        self.decorder_output_layers = []
        
        self._set_decoder_layers()
        
    def call(self, inputs):
        return self._build_decoder(inputs)
    
    def get_config(self):
        base_config = super().get_config()
        config = {
            "shape_before_bottleneck": self.shape_before_bottleneck,
            "decoder_out_filter": self.decoder_out_filter,
            "conv_config": self.conv_config
        }
        return {**base_config, **config}
    
    def _build_decoder(self, x):
        x = self.dense_layer(x)
        tf.print("1st x's Shape: ", x.shape)
        x = self.reshaped_layer(x)
        tf.print("2nd x's Reshape: ", x.shape)
        
        for convT, bn, act in reversed(self.convT_layers):
            x = convT(x)
            x = bn(x)
            x = act(x)
        
        for convT, act in self.decorder_output_layers:
            x = convT(x)
            x = act(x)
            
        return x
    
    def _set_decoder_layers(self):
        self.dense_layer = self._add_dense_layer()
        self.reshaped_layer = self._add_reshape_layer()
        
        for i, params in enumerate(self.conv_config[1:]):
            convT = self._add_decoder_convT_layer(i, params)
            bn = self._add_decoder_bn_layer(i)
            act = self._add_decoder_act_layer(i)
            self.convT_layers.append((convT, bn, act))
            
        self._add_decoder_output_layers()
    
    def _add_dense_layer(self):
        num_neurons = np.prod(self._shape_before_bottleneck)  # Number of neurons in the dense layer is the product of the dimensions before the bottleneck.
        print("Number of Neurons: ", type(num_neurons))
        dense_layer = Dense(
            int(num_neurons),
            name='decoder_dense_layer'
        )
        return dense_layer
    
    def _add_reshape_layer(self):
        print(type(self._shape_before_bottleneck))
        print("Value: ", self._shape_before_bottleneck)
        return Reshape(self._shape_before_bottleneck, name='decoder_reshape_layer')
    
    def _add_decoder_convT_layer(self, layer_num, params):
        filters = params['filters']
        kernel_size = params['kernel_size']
        strides = params['strides']
        return Conv2DTranspose(filters=filters,
                   kernel_size=kernel_size,
                   strides=strides, padding='same',
                   use_bias=False,
                   name=f'decoder_conv_transpose_layer_{layer_num}')
        
    def _add_decoder_bn_layer(self, layer_index ):
        return BatchNormalization(name=f'decoder_batch_norm_layer_{layer_index}')  # Normalizes the activations.
    
    def _add_decoder_act_layer(self, layer_index ):
        return ReLU(name=f'decoder_relu_layer_{layer_index}')  # Applies the non-linearity.
    
    def _add_decoder_output_layers(self):
        num_conv_layers = len(self.convT_layers) + 1
        layer_params = self.conv_config[0] # The first layer's parameters are used for the output layer.
        filters = self.decoder_out_filter # 1 grayscale output image, or 3 RGB output image.
        kernel_size = layer_params['kernel_size']
        strides = layer_params['strides']
        
        convT = Conv2DTranspose(filters=filters,
                   kernel_size=kernel_size,
                   strides=strides, padding='same',
                   use_bias=False,
                   name=f'decoder_conv_transpose_layer_{num_conv_layers}')
        
        act = Activation('sigmoid', name='decoder_output_activation')  # Sigmoid activation for the output layer.

        self.decorder_output_layers.append((convT, act))
        
    def build_graph(self, input_shape):
        x = Input(shape=input_shape[1:])
        return Model(inputs=x, outputs=self.call(x))
    
if __name__ == "__main__":
    shape_before_bottleneck = (65, 18, 64)
    decoder_out_filter = 1
    conv_config = [
        {'filters': 32, 'kernel_size': (3, 3), 'strides': (1, 1)},
        {'filters': 64, 'kernel_size': (3, 3), 'strides': (2, 2)},
        {'filters': 64, 'kernel_size': (3, 3), 'strides': (2, 2)},
        {'filters': 64, 'kernel_size': (3, 3), 'strides': (1, 1)}
    ]
    decoder = DecoderBuilder(shape_before_bottleneck, decoder_out_filter, conv_config)
    latent_dim = 2
    dummy_input = tf.random.normal((1, latent_dim))
    decoder(dummy_input)
    decoder.compile(optimizer=Adam(learning_rate=0.0001), loss=MeanSquaredError())
    decoder_model = decoder.build_graph(input_shape=dummy_input.shape)
    decoder_model.summary()