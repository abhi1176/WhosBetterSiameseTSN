
import tensorflow as tf

from tensorflow.keras.layers import Concatenate, Input, Lambda, Subtract
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import plot_model

from alexnet_func_cpu import tsn_alexnet
from p2k.alexnet_pytorch_to_keras import create_keras_alexnet
from custom_loss import get_custom_loss

# base_model_file = 'p2k/keras_alexnet_imagenet'
# base_model_file = 'keras_alexnet_imagenet'

def create_model(num_snippets, num_input_channels, plot_model_as=None):
    better_inputs = []
    worse_inputs = []
    outputs = []

    # base_model = load_model(base_model_file)
    # base_model = create_keras_alexnet()
    base_model = tsn_alexnet(input_shape=(224, 224, num_input_channels))
    base_model.summary()
    for i in range(num_snippets):
        # Better Branch
        better_input_layer = Input(shape=(224, 224, num_input_channels))
        better_skill = base_model(better_input_layer)
        better_inputs.append(better_input_layer)
        outputs.append(better_skill)

        # Worse Branch
        worse_input_layer = Input(shape=(224, 224, num_input_channels))
        worse_skill = base_model(worse_input_layer)
        worse_inputs.append(worse_input_layer)
        outputs.append(worse_skill)

    inputs = better_inputs + worse_inputs
    model = Model(inputs=inputs, outputs=outputs)
    try:
        if plot_model_as:
            plot_model(model, to_file=plot_model_as, show_shapes=True,
                       show_layer_names=True)
    except Exception as e:
        print("[EXCEPTION] Unable to plot model..")
        print(e.message)
    return model
