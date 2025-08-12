from autoencoder import Autoencoder
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Layer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError, Loss

class VariationalAutoencoder(Autoencoder, Model):
    def __init__(self, input_shape, latent_space_dim, decoder_out_filter, recon_weight, beta, **kwargs):
        Model.__init__(self)  # Initialize the Model class
        Autoencoder.__init__(self, latent_space_dim, decoder_out_filter, **kwargs) # Initialize the Autoencoder class
        
        self.recon_weight = recon_weight # Weight for the reconstruction loss.
        self.beta = beta # Weight for the KL divergence loss.
        
        self.reconstruction_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_tracker = tf.keras.metrics.Mean(name="kl_loss")
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        
        self.val_reconstruction_tracker = tf.keras.metrics.Mean(name="val_reconstruction_loss")
        self.val_kl_tracker = tf.keras.metrics.Mean(name="val_kl_loss")
        self.val_total_loss_tracker = tf.keras.metrics.Mean(name="val_total_loss")
        
        self.mu = None  # Placeholder for the mean vector of the latent space.
        self.log_variance = None  # Placeholder for the log variance vector of the latent space
        self._model = self._build(input_shape)
        self._model.trainable = True
        self._model._name = 'vae_submodel'
        self.__setattr__('_model', self._model)  # Optional but explicit
        self._trainable_weights = self._model.trainable_weights
        self._non_trainable_weights = self._model.non_trainable_weights
        
    @property
    def metrics(self):
        return [
            self.reconstruction_tracker, self.kl_tracker, self.total_loss_tracker,
            self.val_reconstruction_tracker, self.val_kl_tracker, self.val_total_loss_tracker
        ]
    
    def _build(self, inputs_shape):
        super()._build(inputs_shape)  # Call the parent class's _build method to build the encoder and decoder.
        decoder_output = super()._build(inputs_shape)
        return Model(inputs=self._model_input, outputs=[decoder_output, self.mu, self.log_variance], name='variational_autoencoder')
    
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
    
    def call(self, inputs):
        reconstruction, mu, log_variance = self._model(inputs)  # Forward pass through the model.
        # print(f"In the call method of the model with an output of: {reconstruction, mu, log_variance}")
        
        # KL divergence: sum over latent dim, mean over batch
        kl_loss = -0.5 * tf.reduce_sum(1 + log_variance - tf.square(mu) - tf.exp(log_variance), axis=1)
        self.add_loss(self.beta * tf.reduce_mean(kl_loss))
        
        return {
            "reconstruction": reconstruction,
            "mu": mu,
            "log_variance": log_variance
        }

    @tf.function
    def train_step(self, data):
        # x and y are the same.
        x, y = data # Unpack the input data.
        
        with tf.GradientTape() as tape:
            outputs = self(x, training=True)  # Forward pass through the model. Outputs a dictionary with reconstruction, mu, and log_variance.
            # print(f"In the train_step method of the model with an output of: {output}")
            # loss = tf.reduce_mean(self.loss_fn(y, output))  # Compute the loss.
            
            reconstruction = outputs["reconstruction"]
            # mu = outputs["mu"]
            # log_variance = outputs["log_variance"]

            # # Reconstruction loss
            recon_loss = self._calculate_reconstruction_loss(x, reconstruction)  # Calculate the reconstruction loss.
            # KL divergence
            # kl_loss = self._calculate_Kullback_Leibler_divergence(log_variance, mu)  # Calculate the KL divergence loss.
            
            # Total loss includes KL from model.add_loss
            total_loss = self.recon_weight * tf.reduce_mean(recon_loss) + tf.add_n(self.losses)
            
        # print(f"Loss computed in train_step: {loss}")
        
        # reconstructed_pred = output['reconstruction'] # Get the reconstructed output from the model's output.
        # mu = output['mu'] # Get the mean vector from the model's output.
        # log_variance = output['logvar'] # Get the log variance vector from the model's output.
        # print(f"Reconstructed prediction shape: {reconstructed_pred.shape}, mu shape: {mu.shape}, log_variance shape: {log_variance.shape}")
        
        # Backpropagation
        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        # # Get the output[0] or reconstructed_pred, (H, W, C) shape.
        # self.compiled_metrics.update_state(y, reconstructed_pred)  # Update the metrics with the true and predicted values.
        
        # For tracking (you still want KL separately for logging)
        kl_loss = tf.add_n(self.losses) / self.beta
        
        # Update metrics
        self.reconstruction_tracker.update_state(recon_loss)
        self.kl_tracker.update_state(kl_loss)
        self.total_loss_tracker.update_state(total_loss)
        
        return {
            "total_loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_tracker.result(),
            "kl_loss": self.kl_tracker.result(),
        }
    
    @tf.function
    def test_step(self, data):
        # x and y are the same.
        x, y = data # Unpack the input data.
        outputs = self(x, training=False)
        reconstruction = outputs["reconstruction"]
        mu = outputs["mu"]
        log_variance = outputs["log_variance"]
        recon_loss = self._calculate_reconstruction_loss(x, reconstruction)  # Calculate the reconstruction loss.
        kl_loss = self._calculate_Kullback_Leibler_divergence(log_variance, mu)  # Calculate the KL divergence loss.
        total_loss = self.recon_weight * recon_loss + self.beta * kl_loss  # Total loss is a weighted sum of the reconstruction loss and KL divergence.
        
        self.val_reconstruction_tracker.update_state(recon_loss)
        self.val_kl_tracker.update_state(kl_loss)
        self.val_total_loss_tracker.update_state(total_loss)
        
        return {
            "val_total_loss": self.val_total_loss_tracker.result(),
            "val_reconstruction_loss": self.val_reconstruction_tracker.result(),
            "val_kl_loss": self.val_kl_tracker.result(),
        }
    
    def compile(self, learning_rate=0.0001):
        
        # self.loss_fn = VAELoss(recon_weight=1000.0, kl_weight=1.0)  # Custom VAE loss function.
        
        # The loss function will compute the reconstruction loss and KL divergence.
        super().compile(
            optimizer=Adam(learning_rate=learning_rate),
            # loss={
            #     "reconstruction": MeanSquaredError()
            # }  # Reconstruction loss for the output.
        )
    
    def summary(self, *args, **kwargs):
            return self._model.summary(*args, **kwargs)
        
    def _calculate_reconstruction_loss(self, y_true, y_recon):
        """Calculates the reconstruction loss (mean squared error per sample)."""
        error = y_true - y_recon
        reconstruction_loss = tf.reduce_mean(tf.square(error), axis=[1, 2, 3])  # shape: (batch_size,)
        return reconstruction_loss

    def _calculate_Kullback_Leibler_divergence(self, log_variance, mu):
        """Calculates the KL divergence loss per sample."""
        kl_divergence_loss = -0.5 * tf.reduce_sum(
            1 + log_variance - tf.square(mu) - tf.exp(log_variance), axis=1
        ) # shape: (batch_size,)
        return kl_divergence_loss
        
    def save_all(self):
        self._model.save('full_variational_autoencoder_model.keras')
        self.encoder.save('encoder_variational_autoencoder_model.keras')
        self.decoder.save('decoder_variational_autoencoder_model.keras')
        
class Sampling(Layer):
    @tf.function
    def call(self, inputs):
        mu, log_variance = inputs
        epsilon = tf.random.normal(shape=tf.shape(mu), mean=0., stddev=1.) # Explicitly sample from a standard normal distribution.
        sampled_point =  mu + tf.exp(log_variance / 2) * epsilon  # Reparameterization trick: mu + sigma * epsilon, where sigma = exp(log_variance / 2).
        # This allows gradients to flow through the sampling process. 
        return sampled_point

# class VAELoss(Loss):
#     def __init__(self, recon_weight=1000.0, kl_weight=1.0, name="vae_loss"):
#         super().__init__(name=name)
#         self.recon_weight = recon_weight
#         self.kl_weight = kl_weight

#     def call(self, y_true, y_pred):
#         y_recon = y_pred['reconstruction']
#         mu = y_pred['mu']
#         log_variance = y_pred['logvar']

#         reconstruction_loss = self._calculate_reconstruction_loss(y_true, y_recon)
#         kl_divergence_loss = self._calculate_Kullback_Leibler_divergence(log_variance, mu)
        
#         # Add metrics for monitoring
#         # self.add_metric(tf.reduce_mean(reconstruction_loss), name="reconstruction_loss", aggregation="mean")
#         # self.add_metric(tf.reduce_mean(kl_divergence_loss), name="kl_loss", aggregation="mean")
#         # [0.02, 0.03, 0.01, 0.04]
#         # tf.reduce_mean(reconstruction_loss)  # = 0.025
#         # self.add_metric(reconstruction_loss, name="reconstruction_loss") throws an error

#         total_loss = self.recon_weight * reconstruction_loss + self.kl_weight * kl_divergence_loss
#         return total_loss

    # def _calculate_reconstruction_loss(self, y_true, y_recon):
    #     """Calculates the reconstruction loss (mean squared error per sample)."""
    #     error = y_true - y_recon
    #     reconstruction_loss = tf.reduce_mean(tf.square(error), axis=[1, 2, 3])  # shape: (batch_size,)
    #     return reconstruction_loss

    # def _calculate_Kullback_Leibler_divergence(self, log_variance, mu):
    #     """Calculates the KL divergence loss per sample."""
    #     kl_divergence_loss = -0.5 * tf.reduce_sum(
    #         1 + log_variance - tf.square(mu) - tf.exp(log_variance), axis=1
    #     )  # shape: (batch_size,)
    #     return kl_divergence_loss
    
#     def reconstruction_loss(self, y_true, y_pred):
#         y_recon = y_pred['reconstruction']
#         per_sample_loss = self._calculate_reconstruction_loss(y_true, y_recon)  # shape: (batch_size,)
#         return tf.reduce_mean(per_sample_loss)

#     def kl_loss(self, y_pred):
#         mu = y_pred['mu']
#         log_variance = y_pred['logvar']
#         per_sample_loss = self._calculate_Kullback_Leibler_divergence(log_variance, mu)  # shape: (batch_size,)
#         return tf.reduce_mean(per_sample_loss)
    
if __name__ == "__main__":
    input_shape = (257, 69, 1)  # Example input shape
    latent_space_dim = 2  # Example latent space dimension
    decoder_out_filter = 1  # Example output filter for the decoder (e.g., 3 for RGB images)
    recon_weight = 1000.0  # Weight for the reconstruction loss.
    beta = 1.0  # Weight for the KL divergence loss.
    autoencoder = VariationalAutoencoder(input_shape, latent_space_dim, decoder_out_filter, recon_weight, beta, conv_layers_config=[
        {'filters': 32, 'kernel_size': (3, 3), 'strides': (1, 1)},
        {'filters': 64, 'kernel_size': (3, 3), 'strides': (2, 2)},
        {'filters': 64, 'kernel_size': (3, 3), 'strides': (2, 2)},
        {'filters': 64, 'kernel_size': (3, 3), 'strides': (1, 1)}
    ])
    autoencoder.compile(learning_rate=0.001)
    print("Autoencoder compiled successfully.")
    autoencoder.summary()