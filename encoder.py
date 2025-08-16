import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization,LeakyReLU, Flatten, Dense, Reshape, Conv2DTranspose, ReLU, Activation, Lambda, Layer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras import backend as K

@tf.keras.utils.register_keras_serializable()
class EncoderBuilder(Model):
    def __init__(self, latent_space_dim, conv_config=None, **kwargs):
        super().__init__(**kwargs)
        self.conv_config = conv_config
        self.latent_space_dim = latent_space_dim
        
        self.conv_layers = []
        self._flatten_layer = None
        self._shape_before_bottleneck = None
        
        self.mu = None
        self.log_variance = None
        self.sampling_point_layer = None
        
        # self.mu_tensor = None
        # self.log_variance_tensor = None
        
        self._add_vae_bottleneck_layers()
        self._set_encoder_layers()
        
    def call(self, inputs):
        return self._build_model(inputs)

    def get_config(self):
        base_config = super().get_config()
        config = {
            "latent_space_dim": self.latent_space_dim,
            "conv_config": self.conv_config
        }
        return {**base_config, **config}
    
    def _build_model(self, x):
        for conv, bn, act in self.conv_layers:
            x = conv(x)
            x = bn(x)
            x = act(x)
        if self._shape_before_bottleneck is None:
            self._set_shape_before_bottleneck(x)
        x = self._flatten_layer(x)
        mu_tensor = self.mu(x)
        log_variance_tensor = self.log_variance(x)
        latent_vector_z = self.sampling_point_layer([mu_tensor, log_variance_tensor])
        return mu_tensor, log_variance_tensor, latent_vector_z
    
    def _set_encoder_layers(self):
        for i, params in enumerate(self.conv_config):
            conv = self._add_encoder_conv_layer(i, params)
            bn = self._add_encoder_bn_layer(i)
            act = self._add_encoder_act_layer(i)
            self.conv_layers.append((conv, bn, act))
        self._flatten_layer = self._add_encoder_flatten_layer()
        
    def _add_encoder_conv_layer(self, layer_index, params):
        filters = params['filters']
        kernel_size = params['kernel_size']
        strides = params['strides']
        return Conv2D(
            filters=filters, kernel_size=kernel_size, strides=strides,
            padding='same', use_bias=False, name=f'encoder_conv_layer_{layer_index + 1}')
    
    def _add_encoder_bn_layer(self, layer_index):
        return BatchNormalization(name=f'encoder_batch_norm_layer_{layer_index + 1}')
    
    def _add_encoder_act_layer(self, layer_index):
        return LeakyReLU(negative_slope=0.01, name=f'encoder_leaky_relu_layer_{layer_index + 1}')
    
    def _add_encoder_flatten_layer(self):
        return Flatten(name='encoder_flatten_layer')
    
    def _add_vae_bottleneck_layers(self):
        self.mu = Dense(self.latent_space_dim, name='mu')
        self.log_variance = Dense(self.latent_space_dim, name='log_variance')
        self.sampling_point_layer = Sampling(name='encoder_output')
    
    def _set_shape_before_bottleneck(self, x):
        self._shape_before_bottleneck = K.int_shape(x)[1:]
    
    def get_shape_before_bottleneck(self):
        return self._shape_before_bottleneck
    
    def build_graph(self, input_shape):
        x = Input(shape=input_shape[1:])
        return Model(inputs=x, outputs=self.call(x))
    
    # def get_mean_vector_tensor(self):
    #     return self.mu_tensor
    
    # def get_log_variance_vector_tensor(self):
    #     return self.log_variance_tensor
@tf.keras.utils.register_keras_serializable()
class Sampling(Layer):
    @tf.function
    def call(self, inputs):
        mu, log_variance = inputs
        epsilon = tf.random.normal(shape=tf.shape(mu), mean=0., stddev=1.) # Explicitly sample from a standard normal distribution.
        sampled_point =  mu + tf.exp(log_variance / 2) * epsilon  # Reparameterization trick: mu + sigma * epsilon, where sigma = exp(log_variance / 2).
        # This allows gradients to flow through the sampling process. 
        return sampled_point
    
if __name__ == "__main__":
    latent_space_dim = 2  # or whatever you use
    encoder = EncoderBuilder(latent_space_dim, conv_config=[
        {'filters': 32, 'kernel_size': (3, 3), 'strides': (1, 1)},
        {'filters': 64, 'kernel_size': (3, 3), 'strides': (2, 2)},
        {'filters': 64, 'kernel_size': (3, 3), 'strides': (2, 2)},
        {'filters': 64, 'kernel_size': (3, 3), 'strides': (1, 1)}
    ])
    dummy_input = tf.random.normal((1, 260, 72, 1))
    encoder(dummy_input)
    print("Shape before bottleneck:", encoder.get_shape_before_bottleneck())
    encoder.compile(optimizer=Adam(learning_rate=0.0001), loss=MeanSquaredError())
    
    encoder_model = encoder.build_graph(input_shape=dummy_input.shape)
    encoder_model.summary()