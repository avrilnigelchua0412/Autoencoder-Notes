
class VariationalAutoencoder(Autoencoder):
    def __init__(self, input_shape, latent_space_dim, decoder_out_filter, **kwargs):
        super().__init__(input_shape, latent_space_dim, decoder_out_filter, **kwargs)
    
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

class VAELoss(Loss):
    def __init__(self, recon_weight=1000.0, kl_weight=1.0, name="vae_loss"):
        super().__init__(name=name)
        self.recon_weight = recon_weight
        self.kl_weight = kl_weight

    def call(self, y_true, y_pred):
        y_recon, mu, log_variance = y_pred

        reconstruction_loss = self._calculate_reconstruction_loss(y_true, y_recon)
        kl_divergence_loss = self._calculate_Kullback_Leibler_divergence(log_variance, mu)
        
        # Add metrics for monitoring
        # self.add_metric(tf.reduce_mean(reconstruction_loss), name="reconstruction_loss", aggregation="mean")
        # self.add_metric(tf.reduce_mean(kl_divergence_loss), name="kl_loss", aggregation="mean")
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