
import numpy as np
import torch
from torch.autograd import Variable
from pytorch2keras.converter import pytorch_to_keras
import torchvision
from torchsummary import summary

from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model

SAVE_KERAS_MODEL_AS = "keras_alexnet_imagenet"


def create_keras_alexnet():
    pytorch_model = torch.hub.load('pytorch/vision:v0.9.0', 'alexnet', pretrained=True)
    input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
    input_var = Variable(torch.FloatTensor(input_np))
    k_model = pytorch_to_keras(
        pytorch_model,
        input_var,
        [input_var.shape[-3:]],
        change_ordering=True,
        verbose=False,
        name_policy="keep",
    )
    pytorch_model.eval()
    # k_model.summary()
    # summary(model_to_transfer, input_size=input_var.shape[-3:])
    check_error = 0
    if (check_error):
        max_error = 0
        for i in range(100):
            input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
            input_var = Variable(torch.FloatTensor(input_np))
            pytorch_output = pytorch_model(input_var).data.numpy()
            input_np = np.transpose(input_np, (0, 2, 3, 1))
            keras_output = k_model.predict(input_np)
            error = np.max(pytorch_output - keras_output)
            if max_error < error:
                max_error = error
        print('Max error: {0}'.format(max_error))


    # Layers as in Keras Alexnet Model obtained above
    '''
    InputLayer -> ZeroPadding2D -> Conv2D -> Activation -> MaxPooling2D -> ZeroPadding2D ->
        Conv2D -> Activation -> MaxPooling2D -> ZeroPadding2D -> Conv2D -> Activation ->
        ZeroPadding2D -> Conv2D -> Activation -> ZeroPadding2D -> Conv2D -> Activation ->
        MaxPooling2D -> AveragePooling2D -> Lambda -> Flatten -> Dense -> Activation ->
        Dense -> Activation -> Dense(1000)
    '''

    # pytorch2keras drops Dropout Layers which are before the 1st and 2nd Dense layers
    # Add Dropout before the 1st and 2nd Dense layers i.e.,
    # Add Dropout after Flatten and after 1st Dense->Activation

    flatten_1 = k_model.layers[-6]  # Add Dropout after this
    dense_1 = k_model.layers[-5]
    act_1 = k_model.layers[-4]  # Add Dropout after this
    dense_2 = k_model.layers[-3]
    act_2 = k_model.layers[-2]
    dense_3 = k_model.layers[-1]

    x = flatten_1.output
    x = Dropout(0.5)(x)  # Adding Dropout
    x = dense_1(x)
    x = act_1(x)
    x = Dropout(0.5)(x)  # Adding Dropout
    x = dense_2(x)
    x = act_2(x)
    # x = dense_3(x) | Replace last dense layer of 1000 nodes with 1 node
    x = Dense(1)(x)
    # x = Dense(1, activation='sigmoid')(x)
    # x = k_model.outputs

    updated_model = Model(inputs=k_model.inputs, outputs=x)
    return updated_model


if __name__ == '__main__':
    model = create_keras_alexnet()
    print("[INFO] Saved Keras modified model at: {}".format(SAVE_KERAS_MODEL_AS))
    model.save(SAVE_KERAS_MODEL_AS)
