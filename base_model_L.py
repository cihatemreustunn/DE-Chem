import tensorflow as tf

class BaseModel(tf.keras.Model):
    def __init__(self, learning_rate=0.001, seed=None):
        super(BaseModel, self).__init__()
        if seed is not None:
            tf.random.set_seed(seed)
        
        # Main path - wider layers
        self.dense1 = tf.keras.layers.Dense(48, activation='gelu', 
                                          kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed))
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.dropout1 = tf.keras.layers.Dropout(0.2)
        
        self.dense2 = tf.keras.layers.Dense(32, activation='gelu',
                                          kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed+1 if seed else None))
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.dropout2 = tf.keras.layers.Dropout(0.15)

        self.dense3 = tf.keras.layers.Dense(24, activation='gelu',
                                          kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed+2 if seed else None))
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.dropout3 = tf.keras.layers.Dropout(0.10)

        self.dense4 = tf.keras.layers.Dense(16, activation='gelu', 
                                          kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed))
        self.bn4 = tf.keras.layers.BatchNormalization()
        self.dropout4 = tf.keras.layers.Dropout(0.05)
        
        # Skip connection
        self.skip_dense = tf.keras.layers.Dense(16, activation='gelu',
                                            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed+3 if seed else None))
        
        # Output layers - separate for mean and log variance
        self.output_mean = tf.keras.layers.Dense(10)
        self.output_logvar = tf.keras.layers.Dense(10)
        
        self.learning_rate = learning_rate
        
    def get_optimizer(self):
        return tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        
    def call(self, inputs, training=False):
        # Skip connection
        skip = self.skip_dense(inputs)
        
        # Main path
        x = self.dense1(inputs)
        x = self.bn1(x, training=training)
        x = self.dropout1(x, training=training)

        x = self.dense2(x)
        x = self.bn2(x, training=training)
        x = self.dropout2(x, training=training)
        
        x = self.dense3(x)
        x = self.bn3(x, training=training)
        x = self.dropout3(x, training=training)

        x = self.dense4(x)
        x = self.bn4(x, training=training)
        x = self.dropout4(x, training=training)
        
        # Combine with skip connection
        x = tf.concat([x, skip], axis=-1)
        
        # Generate mean and log variance predictions
        mean = self.output_mean(x)
        
        # For log variance, we use a different range to ensure stability
        # Initialize with slightly negative values to start with small variances
        logvar_init = tf.keras.initializers.RandomUniform(minval=-3, maxval=-1)
        logvar = self.output_logvar(x)
        
        # Stack mean and logvar for output
        return mean, logvar