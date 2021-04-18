# AlexNet Model

"""
AlexNet, Krizhevsky, Alex, Ilya Sutskever and Geoffrey E. Hinton, 2012
https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
Michael Guerzhoy and Davi Frossard, 2016
AlexNet implementation in TensorFlow, with weights Details:
#http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
The script is the realizatio of object oriented style of AlexNet. The construction method of AlexNet
includes three parameters, including self, input_shape, num_classes, of which, input_shape works as a
placeholder.
"""

import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, \
    MaxPooling2D, Dropout, Lambda


def tsn_alexnet(input_shape, weights_file="bvlc_alexnet.npy"):
    input_ = Input(shape=input_shape)

    x = Conv2D(filters=96, kernel_size=(11,11), strides=4,
               padding='valid', activation='relu',
               kernel_initializer='GlorotNormal',
               name='conv1')(input_)

    radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
    x = Lambda(tf.nn.local_response_normalization,
               arguments={"depth_radius": radius, "alpha": alpha,
                          "beta": beta, "bias": bias})(x)

    x = MaxPooling2D(pool_size=(3,3), strides=(2,2),
                     padding='valid', data_format=None)(x)

    x = Conv2D(filters=256, kernel_size=(5,5), strides=1,
               padding='same', activation='relu',
               kernel_initializer='GlorotNormal',
               name='conv2', groups=2)(x)

    radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
    x = Lambda(tf.nn.local_response_normalization,
               arguments={"depth_radius": radius, "alpha": alpha,
                          "beta": beta, "bias": bias})(x)

    x = MaxPooling2D(pool_size=(3,3), strides=(2,2),
                     padding='valid', data_format=None)(x)

    x = Conv2D(filters=384, kernel_size=(3,3), strides=1,
               padding='same', activation='relu',
               kernel_initializer='GlorotNormal',
               name='conv3')(x)

    x = Conv2D(filters=384, kernel_size=(3,3), strides=1,
               padding='same', activation='relu',
               kernel_initializer='GlorotNormal',
               name='conv4', groups=2)(x)

    x = Conv2D(filters=256, kernel_size=(3,3), strides=1,
               padding='same', activation='relu',
               kernel_initializer='GlorotNormal',
               name='conv5', groups=2)(x)
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2),
                     padding='valid', name='last_maxpool')(x)

    x = Flatten()(x)

    x = Dense(4096, activation='relu', name='fc6')(x)
    x = Dropout(0.5)(x)

    x = Dense(4096, activation='relu', name='fc7')(x)
    x = Dropout(0.5)(x)

    x = Dense(1, name='fc8')(x)
    model = Model(inputs=input_, outputs=x)

    print("[INFO] Loading pretrained weights...")
    net_data = np.load(open(weights_file, "rb"), encoding="latin1", allow_pickle=True).item()

    layers = ['conv2', 'conv3', 'conv4', 'conv5']

    if input_shape[2] == 3:
        layers = ["conv1"] + layers

    for layer in layers:
        lay = model.get_layer(name=layer)
        print(lay.name, lay.output_shape, lay.get_weights()[0].shape, net_data[layer][0].shape)
        model.get_layer(name=layer).set_weights(net_data[layer])
    return model


if __name__ == "__main__":
    tsn_model = tsn_alexnet()
    tsn_model.summary()
    import numpy as np
    x = np.random.rand(1, 224, 224, 3)
    tsn_model(x)
