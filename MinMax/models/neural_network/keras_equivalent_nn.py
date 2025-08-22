import tensorflow as tf
from tensorflow.keras import layers, Model, initializers
import numpy as np

class Normalise(tf.keras.layers.Layer):
    def __init__(self, n_neurons):
        super(Normalise, self).__init__()
        self.minimum = self.add_weight(shape=(n_neurons,), initializer='zeros', trainable=False)
        self.delta = self.add_weight(shape=(n_neurons,), initializer='ones', trainable=False)
        self.eps = tf.constant(1e-8, dtype=tf.float32)

    def call(self, inputs):
        return (inputs - self.minimum) / (self.delta + self.eps)

    def set_normalisation(self, minimum, delta):
        minimum = tf.convert_to_tensor(minimum, dtype=tf.float32)
        delta = tf.convert_to_tensor(delta, dtype=tf.float32)

        if minimum.shape.rank != 1 or delta.shape.rank != 1:
            raise ValueError("Input statistics must be 1-D tensors.")
        if tf.reduce_any(delta <= 1e-12):
            raise ValueError("Delta contains zero or near-zero values.")
        
        self.minimum.assign(minimum)
        self.delta.assign(delta)

class Denormalise(tf.keras.layers.Layer):
    def __init__(self, n_neurons):
        super(Denormalise, self).__init__()
        self.delta = self.add_weight(shape=(n_neurons,), initializer='ones', trainable=False)

    def call(self, inputs):
        return inputs * self.delta

    def set_normalisation(self, delta):
        delta = tf.convert_to_tensor(delta, dtype=tf.float32)
        if delta.shape.rank != 1:
            raise ValueError("Delta must be a 1-D tensor.")
        if tf.reduce_any(delta <= 1e-12):
            raise ValueError("Delta contains zero or near-zero values.")
        
        self.delta.assign(delta)

class Clamp(tf.keras.layers.Layer):
    def __init__(self, lower=0.0, upper=1.0):
        super(Clamp, self).__init__()
        self.lower = lower
        self.upper = upper

    def call(self, inputs):
        return tf.clip_by_value(inputs, self.lower, self.upper)

class NeuralNetwork(tf.keras.Model):
    def __init__(self, num_features, hidden_layer_size=[100, 100, 100], num_output=1, keras_init_seed=0):
        super(NeuralNetwork, self).__init__()
        tf.random.set_seed(keras_init_seed)

        self.Input_Normalise = Normalise(num_features)
        self.Output_De_Normalise = Denormalise(num_output)

        self.L_1 = layers.Dense(hidden_layer_size[0], activation='relu',
                                kernel_initializer=initializers.GlorotUniform(seed=keras_init_seed))
        self.L_2 = layers.Dense(hidden_layer_size[1], activation='relu',
                                kernel_initializer=initializers.GlorotUniform(seed=keras_init_seed + 1))
        self.L_3 = layers.Dense(hidden_layer_size[2], activation='relu',
                                kernel_initializer=initializers.GlorotUniform(seed=keras_init_seed + 2))
        self.L_4 = layers.Dense(num_output,
                                kernel_initializer=initializers.GlorotUniform(seed=keras_init_seed + 3))

        self.clamp_layer = Clamp(lower=0.0, upper=1.0)

    def normalise_input(self, minimum, delta):
        self.Input_Normalise.set_normalisation(minimum, delta)

    def normalise_output(self, delta):
        self.Output_De_Normalise.set_normalisation(delta)

    def call(self, x, training=False, mode="default"):
        """
        Mode:
        - "default" (forward): Includes constraint via y = sum(x) - sum(Pd)
        - "train": Raw network output
        - "aft": Output with clamp layer applied
        """
        Pd = x
        x = self.Input_Normalise(x)
        x = self.L_1(x)
        x = self.L_2(x)
        x = self.L_3(x)
        x = self.L_4(x)

        if mode == "aft":
            x = self.clamp_layer(x)

        x = self.Output_De_Normalise(x)

        if mode == "default":
            y = tf.reduce_sum(x, axis=1, keepdims=True) - tf.reduce_sum(Pd, axis=1, keepdims=True)
            return y
        else:
            return x
