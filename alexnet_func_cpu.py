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
    MaxPooling2D, Dropout, Lambda, Concatenate


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

    x1 = Lambda(lambda x: x[:, :, :, :48], name='lambda_conv2_a')(x)
    x1 = Conv2D(filters=128, kernel_size=(5,5), strides=1,
               padding='same', activation='relu',
               kernel_initializer='GlorotNormal',
               name='conv2_a')(x1)

    x2 = Lambda(lambda x: x[:, :, :, 48:], name='lambda_conv2_b')(x)
    x2 = Conv2D(filters=128, kernel_size=(5,5), strides=1,
               padding='same', activation='relu',
               kernel_initializer='GlorotNormal',
               name='conv2_b')(x2)
    x = Concatenate()([x1, x2])

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

    x1 = Lambda(lambda x: x[:, :, :, :192], name='lambda_conv4_a')(x)
    x1 = Conv2D(filters=192, kernel_size=(3,3), strides=1,
               padding='same', activation='relu',
               kernel_initializer='GlorotNormal',
               name='conv4_a')(x1)

    x2 = Lambda(lambda x: x[:, :, :, 192:], name='lambda_conv4_b')(x)
    x2 = Conv2D(filters=192, kernel_size=(3,3), strides=1,
               padding='same', activation='relu',
               kernel_initializer='GlorotNormal',
               name='conv4_b')(x2)
    x = Concatenate()([x1, x2])

    x1 = Lambda(lambda x: x[:, :, :, :192], name='lambda_conv5_a')(x)
    x1 = Conv2D(filters=128, kernel_size=(3,3), strides=1,
               padding='same', activation='relu',
               kernel_initializer='GlorotNormal',
               name='conv5_a')(x1)
    x2 = Lambda(lambda x: x[:, :, :, 192:], name='lambda_conv5_b')(x)
    x2 = Conv2D(filters=128, kernel_size=(3,3), strides=1,
               padding='same', activation='relu',
               kernel_initializer='GlorotNormal',
               name='conv5_b')(x2)
    x = Concatenate()([x1, x2])

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

    layers = ['conv2_a', 'conv2_b', 'conv3', 'conv4_a', 'conv4_b', 'conv5_a', 'conv5_b']

    if input_shape[2] == 3:
        layers = ["conv1"] + layers

    net_data['conv2_a'] = [net_data['conv2'][0][..., :128], net_data['conv2'][1][..., :128]]
    net_data['conv2_b'] = [net_data['conv2'][0][..., 128:], net_data['conv2'][1][..., 128:]]

    net_data['conv4_a'] = [net_data['conv4'][0][..., :192], net_data['conv4'][1][..., :192]]
    net_data['conv4_b'] = [net_data['conv4'][0][..., 192:], net_data['conv4'][1][..., 192:]]

    net_data['conv5_a'] = [net_data['conv5'][0][..., :128], net_data['conv5'][1][..., :128]]
    net_data['conv5_b'] = [net_data['conv5'][0][..., 128:], net_data['conv5'][1][..., 128:]]

    for layer_name in layers:
        # print("[INFO] Setting weights for layer: {}".format(layer_name))
        layer = model.get_layer(layer_name)
        layer.set_weights(net_data[layer_name])

    return model


if __name__ == "__main__":
    tsn_model = tsn_alexnet()
    import numpy as np
    x = np.random.rand(1, 224, 224, 3)
    print(tsn_model(x))
