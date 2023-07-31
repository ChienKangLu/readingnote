import json
import os

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
from matplotlib import pyplot as plt

from tensorflow.keras import layers

if __name__ == '__main__':
    """Load dataset"""
    # (train_examples, validation_examples), info = tfds.load(
    #     'cats_vs_dogs',
    #     data_dir="./datasets",
    #     with_info=True,
    #     as_supervised=True,
    #     split=['train[:80%]', 'train[80%:]'],
    # )
    #
    # # print(info)
    #
    # num_examples = info.splits['train'].num_examples
    # num_classes = info.features['label'].num_classes
    #
    # IMAGE_RES = 224
    #
    # def format_image(image, label):
    #     image = tf.image.resize(image, (IMAGE_RES, IMAGE_RES)) / 255.0
    #     return image, label
    #
    # BATCH_SIZE = 32
    # train_batches = train_examples.shuffle(num_examples // 4).map(format_image).batch(BATCH_SIZE).prefetch(1)
    # validation_batches = validation_examples.map(format_image).batch(BATCH_SIZE).prefetch(1)

    # _URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
    # dir_name = os.path.abspath(os.path.dirname(__file__))
    # path = os.path.join(dir_name, "dataset/cats_and_dogs_filterted.zip")
    # unzip_path = os.path.join(dir_name, "dataset")
    # zip_dir = tf.keras.utils.get_file(path, origin=_URL, extract=True)
    base_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'datasets/cats_and_dogs_filtered')
    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')

    print("Display dataset structure:")
    os.system("tree {} -L 2".format(base_dir))

    train_cats_dir = os.path.join(train_dir, 'cats')  # directory with our training cat pictures
    train_dogs_dir = os.path.join(train_dir, 'dogs')  # directory with our training dog pictures
    validation_cats_dir = os.path.join(validation_dir, 'cats')  # directory with our validation cat pictures
    validation_dogs_dir = os.path.join(validation_dir, 'dogs')  # directory with our validation dog pictures

    num_cats_tr = len(os.listdir(train_cats_dir))
    num_dogs_tr = len(os.listdir(train_dogs_dir))

    num_cats_val = len(os.listdir(validation_cats_dir))
    num_dogs_val = len(os.listdir(validation_dogs_dir))

    total_train = num_cats_tr + num_dogs_tr
    total_val = num_cats_val + num_dogs_val

    BATCH_SIZE = 100  # Number of training examples to process before updating our models variables
    IMAGE_RES = 224  # Our training data consists of images with width of 150 pixels and height of 150 pixels

    """Resize image"""
    train_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
    train_data_gen = train_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                               directory=train_dir,
                                                               shuffle=True,
                                                               target_size=(IMAGE_RES, IMAGE_RES),  # (150,150)
                                                               class_mode='binary')

    validation_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
    val_data_gen = validation_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                                  directory=validation_dir,
                                                                  shuffle=False,
                                                                  target_size=(IMAGE_RES, IMAGE_RES),  # (150,150)
                                                                  class_mode='binary')

    """Download from TensorFlow Hub"""
    URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
    feature_extractor = hub.KerasLayer(URL,
                                       input_shape=(IMAGE_RES, IMAGE_RES, 3))
    feature_extractor.trainable = False

    EPOCHS = 6 #100

    trained_model_name = 'model.h5'  # 'model_100.h5'
    history_name = 'history.json'  # 'history_100.json'
    is_model_trained = os.path.isfile(trained_model_name)
    if is_model_trained is False:
        """Build model"""
        model = tf.keras.Sequential([
            feature_extractor,
            layers.Dense(2)
        ])

        model.summary()

        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

        history = model.fit(train_data_gen,
                            epochs=EPOCHS,
                            validation_data=val_data_gen)

        """Dump history"""
        # Get the dictionary containing each metric and the loss for each epoch
        history_dict = history.history
        # Save it under the form of a json file
        json.dump(history_dict, open(history_name, 'w'))

        """Save Model"""
        model.save(trained_model_name)
    else:
        """Reload Model"""
        model = tf.keras.models.load_model(trained_model_name, custom_objects={'KerasLayer': hub.KerasLayer})
        history_dict = json.load(open(history_name, 'r'))


    # Visualizing results of the training
    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']

    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs_range = range(EPOCHS)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    # plt.savefig('./foo.png')
    plt.show()
