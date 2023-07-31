import math
import os

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from commom import normalize, display_images, plot_image, plot_value_array


if __name__ == '__main__':
    """Set seed for TensorFlow and Numpy to ensure that your code is repeatable"""
    np.random.seed(42)
    tf.keras.utils.set_random_seed(42)

    """Loading the dataset returns metadata as well as training dataset and test dataset."""
    dataset, metadata = tfds.load('fashion_mnist', data_dir="./datasets", as_supervised=True, with_info=True)
    print(metadata)

    train_dataset: tf.data.Dataset = dataset['train']
    print("Element type: {}".format(train_dataset.element_spec))
    test_dataset: tf.data.Dataset = dataset['test']

    """Display the first 25 images from the training set and display the class name below each image"""
    class_names = metadata.features['label'].names
    # display_images(25, train_dataset, class_names)

    """The map function applies the normalize function to each element in the train and test datasets"""
    train_dataset = train_dataset.map(normalize)
    test_dataset = test_dataset.map(normalize)

    trained_model_name = 'model_cnn.h5'
    is_model_trained = os.path.isfile(trained_model_name)
    if is_model_trained is False:
        """Set up the layers"""
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation=tf.nn.relu,
                                   input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D((2, 2), strides=2),
            tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation=tf.nn.relu),
            tf.keras.layers.MaxPooling2D((2, 2), strides=2),
            tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(10, activation=tf.nn.softmax)
        ])
        model.summary()

        """Compile the model"""
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      metrics=['accuracy'])

        """Train model"""
        BATCH_SIZE = 32
        num_train_examples = metadata.splits['train'].num_examples
        train_dataset = train_dataset.cache().repeat().shuffle(num_train_examples).batch(BATCH_SIZE)
        test_dataset = test_dataset.cache().batch(BATCH_SIZE)
        history = model.fit(train_dataset, epochs=5, steps_per_epoch=math.ceil(num_train_examples / BATCH_SIZE))

        """Save Model"""
        model.save(trained_model_name)
    else:
        """Reload Model"""
        BATCH_SIZE = 32
        num_train_examples = metadata.splits['train'].num_examples
        train_dataset = train_dataset.cache().repeat().shuffle(num_train_examples).batch(BATCH_SIZE)
        test_dataset = test_dataset.cache().batch(BATCH_SIZE)
        model = tf.keras.models.load_model(trained_model_name)

    """Evaluate"""
    num_test_examples = metadata.splits['test'].num_examples
    test_loss, test_accuracy = model.evaluate(test_dataset, steps=math.ceil(num_test_examples / 32))
    print('Accuracy on test dataset:', test_accuracy)

    """Make prediction for images in first batch"""
    test_images, labels = next(test_dataset.take(1).as_numpy_iterator())
    predictions = model.predict(test_images)
    first_image_prediction = predictions[0]
    print(first_image_prediction)
    print("Real label: {}".format(labels[0]))

    # Plot the first X test images, their predicted label, and the true label
    # Color correct predictions in blue, incorrect predictions in red
    num_rows = 5
    num_cols = 3
    num_images = num_rows * num_cols
    plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        plot_image(i, predictions, labels, test_images, class_names)
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
        plot_value_array(i, predictions, labels)
    plt.show()