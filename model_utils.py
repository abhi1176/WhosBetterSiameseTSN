
import tensorflow as tf

from tensorflow.keras.layers import Concatenate, Input, Lambda, Subtract
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

# from AlexNet import AlexNet
from alexnet_func_cpu import tsn_alexnet
from custom_loss import custom_loss
from custom_loss import ranking_loss, similarity_loss

from tensorflow.keras.applications import VGG16


def create_base_model(weights_path=None):
    # model = AlexNet()
    model = VGG16(
        include_top=False, weights='imagenet', input_tensor=None,
        input_shape=(224, 224, 3), pooling=None, classes=1,
        classifier_activation='sigmoid')
    if weights_path:
        model.load_weights(weights_path)
    # x = model.layers[-2].output
    # x = Dense(1, activation='sigmoid')(x)
    return Model(inputs=model.input, outputs=model.output)


def create_model(num_snippets, num_input_channels, plot_model_as=None):
    # inputs = []
    better_inputs = []
    worse_inputs = []
    ranking_outputs = []
    similarity_outputs = []
    # base_model = create_base_model()
    base_model = tsn_alexnet(input_shape=(224, 224, num_input_channels))
    # base_model.summary()
    for i in range(num_snippets):
        better_input_layer = Input(shape=(224, 224, num_input_channels),
                                   name='better_input_layer_{}'.format(i))
        worse_input_layer = Input(shape=(224, 224, num_input_channels),
                                  name='worse_input_layer_{}'.format(i))
        better_skill = base_model(better_input_layer)
        worse_skill = base_model(worse_input_layer)
        diff_ranking = Subtract(name='diff_ranking_{}'.format(i))([worse_skill, better_skill])
        diff_similarity = Subtract(name='diff_similarity_{}'.format(i))([better_skill, worse_skill])
        better_inputs.append(better_input_layer)
        worse_inputs.append(worse_input_layer)
        ranking_outputs.append(diff_ranking)
        similarity_outputs.append(diff_similarity)
    inputs = better_inputs + worse_inputs
    if num_snippets == 1:
        outputs = [ranking_outputs[0], similarity_outputs[0]]
    else:
        r_output = Lambda(lambda x: x)(ranking_outputs.pop())
        s_output = Lambda(lambda x: x)(similarity_outputs.pop())
        while ranking_outputs:
            r_output = Concatenate()([r_output, ranking_outputs.pop()])
            s_output = Concatenate()([s_output, similarity_outputs.pop()])
        r_output = Lambda(lambda x: x, name="ranking_concat")(r_output)
        s_output = Lambda(lambda x: x, name="similarity_concat")(s_output)
        outputs = [r_output, s_output]
    model = Model(inputs=inputs, outputs=outputs)
    try:
        if plot_model_as:
            plot_model(model, to_file=plot_model_as, show_shapes=True,
                       show_layer_names=True)
    except Exception as e:
        print("[EXCEPTION] Unable to plot model..")
        print(e.message)
    return model


# @tf.function
def train_step(model, optimizer, margin, beta, batch):
    with tf.GradientTape() as tape:
        y = model(batch, training=True)
        r_loss = ranking_loss(margin)(y[0])
        s_loss = similarity_loss(margin)(y[1])
        losses = [r_loss, s_loss]
    model_gradient = tape.gradient(losses, model.trainable_variables)
    optimizer.apply_gradients(zip(model_gradient, model.trainable_variables))
    return model_gradient, losses


def validate_batch(model, margin, beta, batch):
    y = model(batch, training=False)
    val_loss = custom_loss(margin, beta)(y)
    return val_loss
