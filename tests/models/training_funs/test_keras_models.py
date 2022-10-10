import tensorflow as tf
from oodeel.types import *
from oodeel.models.training_funs import train_convnet, train_keras_app
from tests import generate_data_tfds, generate_model


def test_convnet():

    train_config = {
        "batch_size": 128,
        "epochs": 2
    }

    input_shape = (32, 32, 3)
    num_labels = 10
    samples = 100

    data = generate_data_tfds(
        x_shape=input_shape, num_labels=num_labels, samples=samples,
        one_hot=False
        )

    model = train_convnet(data, **train_config)


def test_train_keras_app_imagenet():

    train_config = {
        "batch_size": 5,
        "epochs": 2
    }

    input_shape = (224, 224, 3)
    num_labels = 1000
    samples = 100

    data = generate_data_tfds(
        x_shape=input_shape, num_labels=num_labels, samples=samples,
        one_hot=False
        )

    model = train_keras_app(
        data, model_name='MobileNet', imagenet_pretrained=True, **train_config
        )


def test_train_keras_app():

    train_config = {
        "batch_size": 5,
        "epochs": 3
    }

    input_shape = (56, 56, 3)
    num_labels = 123
    samples = 100

    data = generate_data_tfds(
        x_shape=input_shape, num_labels=num_labels, samples=samples,
        one_hot=False
        )

    validation_data = generate_data_tfds(
        x_shape=input_shape, num_labels=num_labels, samples=samples,
        one_hot=False
        )

    model = train_keras_app(
        data, model_name='MobileNet', imagenet_pretrained=False, 
        validation_data=validation_data, **train_config
        )
