import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Layer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError, Loss
from encoder import EncoderBuilder
from decoder import DecoderBuilder

"""Use tf if you're subclassing tf.keras.Model and
want to avoid Keras Backend dependency"""

class VariationalAutoencoder(Model):
    def __init__(self, latent_space_dim, recon_weight, beta, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.latent_space_dim = latent_space_dim
        self.recon_weight = recon_weight
        self.beta = beta
        
        self._encoder = encoder
        self._decoder = decoder

        self.__set_default()
        
    def call(self, inputs):
        return self._build(inputs)
        
    def __set_default(self):
        self.reconstruction_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_tracker = tf.keras.metrics.Mean(name="kl_loss")
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        
        self.val_reconstruction_tracker = tf.keras.metrics.Mean(name="val_reconstruction_loss")
        self.val_kl_tracker = tf.keras.metrics.Mean(name="val_kl_loss")
        self.val_total_loss_tracker = tf.keras.metrics.Mean(name="val_total_loss")
        
    @property
    def metrics(self):
        return [
            self.reconstruction_tracker, self.kl_tracker, self.total_loss_tracker,
            self.val_reconstruction_tracker, self.val_kl_tracker, self.val_total_loss_tracker
        ]
    
    def _build(self, x):
        mu, log_var, z = self._encoder(x)
        # mu = self._encoder.get_mean_vector_tensor()
        # log_var = self._encoder.get_log_variance_vector_tensor()
        reconstruction = self._decoder(z)
        
        kl_loss = self._kl_divergence_loss(mu, log_var)
        self.add_loss(kl_loss)
        
        return reconstruction
    
    def _kl_divergence_loss(self, mean_vector, log_variance_vector):
        return tf.reduce_mean(-0.5 * tf.reduce_sum(
            tf.ones_like(log_variance_vector) + log_variance_vector - tf.square(mean_vector) - tf.exp(log_variance_vector),
            axis = 1
        ))
    
    def _reconstruction_loss(self, x, reconstruction):
        return tf.reduce_mean(
            tf.reduce_sum(tf.square(x - reconstruction), axis=[1, 2, 3])
        )
    
    @tf.function
    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            reconstruction = self(x, training=True)
            reconstruction_loss = self._reconstruction_loss(x, reconstruction)
            kl_loss = sum(self.losses)
            total_loss = self.recon_weight * reconstruction_loss + self.beta * kl_loss

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.reconstruction_tracker.update_state(reconstruction_loss)
        self.kl_tracker.update_state(kl_loss)
        self.total_loss_tracker.update_state(total_loss)

        return {
            "reconstruction_loss": self.reconstruction_tracker.result(),
            "kl_loss": self.kl_tracker.result(),
            "total_loss": self.total_loss_tracker.result(),
        }
        
    @tf.function
    def test_step(self, data):
        x, y = data

        reconstruction = self(x, training=False)
        reconstruction_loss = self._reconstruction_loss(x, reconstruction)
        kl_loss = sum(self.losses)
        total_loss = self.recon_weight * reconstruction_loss + self.beta * kl_loss
        
        self.val_reconstruction_tracker.update_state(reconstruction_loss)
        self.val_kl_tracker.update_state(kl_loss)
        self.val_total_loss_tracker.update_state(total_loss)
        
        return {
            "val_reconstruction_loss": self.val_reconstruction_tracker.result(),
            "val_kl_loss": self.val_kl_tracker.result(),
            "val_total_loss": self.val_total_loss_tracker.result(),
        }

    def summary(self):
        self._encoder.summary()
        self._decoder.summary()
        
if __name__ == "__main__":
    conv_config=[
        {'filters': 32, 'kernel_size': (3, 3), 'strides': (1, 1)},
        {'filters': 64, 'kernel_size': (3, 3), 'strides': (2, 2)},
        {'filters': 64, 'kernel_size': (3, 3), 'strides': (2, 2)},
        {'filters': 64, 'kernel_size': (3, 3), 'strides': (1, 1)}
    ]
    latent_space_dim = 2  # or whatever you use
    # Instantiate Encoder
    encoder = EncoderBuilder(latent_space_dim, conv_config)
    dummy_input = tf.random.normal((1, 260, 72, 1))
    encoder(dummy_input)
    # Get shape before bottleneck for Decoder
    shape_before_bottleneck = encoder.get_shape_before_bottleneck()
    print("Shape before bottleneck:", shape_before_bottleneck)   
    
    encoder_model = encoder.build_graph(input_shape=dummy_input.shape)
    
    # Instantiate Decoder
    decoder_out_filter = 1
    decoder = DecoderBuilder(shape_before_bottleneck, decoder_out_filter, conv_config)
    dummy_input = tf.random.normal((1, latent_space_dim))
    decoder(dummy_input)  # Build decoder without dummy data
    
    decoder_model = decoder.build_graph(input_shape=dummy_input.shape)
    
    # Now instantiate the VAE
    vae = VariationalAutoencoder(
        latent_space_dim,
        recon_weight=1.0,
        beta=1.0,
        encoder=encoder_model,
        decoder=decoder_model
    )
    dummy_input = tf.random.normal((1, 260, 72, 1))
    vae(dummy_input)
    vae.compile(optimizer=Adam(learning_rate=0.0001), loss=MeanSquaredError())
    vae.summary()