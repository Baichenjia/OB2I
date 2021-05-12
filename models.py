import tensorflow as tf
import numpy as np
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.enable_eager_execution(config=config)
layers = tf.keras.layers

####################
# Ordinary Q-network
####################

class QNetwork(tf.keras.Model):
    def __init__(self, num_actions):
        super(QNetwork, self).__init__()
        self.conv1 = layers.Conv2D(filters=32, kernel_size=8, strides=4, activation='relu',
                                   padding='same', name='conv1')
        self.conv2 = layers.Conv2D(filters=64, kernel_size=4, strides=2, activation='relu',
                                   padding='same', name='conv2')
        self.conv3 = layers.Conv2D(filters=64, kernel_size=3, strides=1, activation='relu',
                                   padding='same', name='conv3')
        self.fla = layers.Flatten(name='flatten')
        self.dense1 = layers.Dense(units=512, activation='relu', name='dense1')
        self.dense2 = layers.Dense(units=num_actions, activation=None, name='dense2')

    def call(self, h):
        h = tf.cast(h, tf.float32) / 255.     # (None, 84, 84, 4)
        h = self.conv1(h)                     # (None, 21, 21, 32)
        h = self.conv2(h)                     # (None, 11, 11, 64)
        h = self.conv3(h)                     # (None, 11, 11, 64)
        h = self.fla(h)                       # (None, 7744)
        h = self.dense1(h)                    # (None, 512)
        action_scores = self.dense2(h)        # (None, 4)
        return action_scores


####################
# Ensemble Q-network
####################

class HeadNN(tf.keras.Model):
    def __init__(self, num_actions, name=""):
        super(HeadNN, self).__init__(name=name)
        self.dense1 = layers.Dense(units=512, activation='relu', name=name+'_dense')
        self.dense2 = layers.Dense(units=num_actions, activation=None, name=name+'_Q')

    def call(self, x):
        x = self.dense1(x)
        return self.dense2(x)


class BootQNetwork(tf.keras.Model):
    def __init__(self, num_actions, num_ensemble=10, name=""):
        super(BootQNetwork, self).__init__(name=name)
        self.conv1 = layers.Conv2D(filters=32, kernel_size=8, strides=4, activation='relu',
                                   padding='same', name='conv1')
        self.conv2 = layers.Conv2D(filters=64, kernel_size=4, strides=2, activation='relu',
                                   padding='same', name='conv2')
        self.conv3 = layers.Conv2D(filters=64, kernel_size=3, strides=1, activation='relu',
                                   padding='same', name='conv3')
        self.fla = layers.Flatten(name='flatten')
        self.boot_head = [HeadNN(num_actions, name="head_"+str(i)) for i in range(num_ensemble)]

    def call(self, h, k=None):
        h = tf.cast(h, tf.float32) / 255.     # (None, 84, 84, 4)
        h = self.conv1(h)                     # (None, 21, 21, 32)
        h = self.conv2(h)                     # (None, 11, 11, 64)
        h = self.conv3(h)                     # (None, 11, 11, 64)
        h = self.fla(h)                       # (None, 7744)
        if k is not None:
            ensemble_q = self.boot_head[k](h)  # choose one of ensemble.  (None, num_action)
        else:
            ensemble_q = tf.stack([m(h) for m in self.boot_head], axis=1)  # (None, num_ensemble, num_action)
        return ensemble_q

####################
# Ensemble Q-network With Randomized Prior Function (Osband.2018)
####################


class BootQNetworkWithPrior(tf.keras.Model):
    def __init__(self, prior, main, prior_scale):
        super(BootQNetworkWithPrior, self).__init__()
        self.prior_network = prior
        self.main_network = main
        self.prior_scale = prior_scale

    def call(self, h, k=None):
        return tf.stop_gradient(self.prior_network(h, k)) * self.prior_scale + self.main_network(h, k)
