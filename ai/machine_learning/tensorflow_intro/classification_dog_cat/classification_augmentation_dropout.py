import os
import zipfile

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import json

from classification_dog_cat.common import plot_images, plot_images_with_labels

if __name__ == '__main__':
    """Set seed for TensorFlow and Numpy to ensure that your code is repeatable"""
    np.random.seed(42)
    tf.keras.utils.set_random_seed(42)

    """Data Loading"""
    _URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
    dir_name = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(dir_name, "dataset/cats_and_dogs_filterted.zip")
    unzip_path = os.path.join(dir_name, "dataset")
    zip_dir = tf.keras.utils.get_file(path, origin=_URL, extract=True)

    is_unzipped = os.path.isdir(os.path.join(unzip_path, "cats_and_dogs_filterted"))
    if is_unzipped is False:
        with zipfile.ZipFile(path, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)

    base_dir = os.path.join(os.path.dirname(zip_dir), 'cats_and_dogs_filtered')
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

    print('total training cat images:', num_cats_tr)
    print('total training dog images:', num_dogs_tr)

    print('total validation cat images:', num_cats_val)
    print('total validation dog images:', num_dogs_val)
    print("--")
    print("Total training images:", total_train)
    print("Total validation images:", total_val)

    """Setting Model Parameters"""
    BATCH_SIZE = 100  # Number of training examples to process before updating our models variables
    IMG_SHAPE = 150  # Our training data consists of images with width of 150 pixels and height of 150 pixels

    """Resize image"""
    train_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255,
                                                                            rotation_range=40,
                                                                            width_shift_range=0.2,
                                                                            height_shift_range=0.2,
                                                                            shear_range=0.2,
                                                                            zoom_range=0.2,
                                                                            horizontal_flip=True,
                                                                            fill_mode='nearest')

    train_data_gen = train_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                               directory=train_dir,
                                                               shuffle=True,
                                                               target_size=(IMG_SHAPE, IMG_SHAPE),  # (150,150)
                                                               class_mode='binary')

    validation_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
    val_data_gen = validation_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                                  directory=validation_dir,
                                                                  shuffle=False,
                                                                  target_size=(IMG_SHAPE, IMG_SHAPE),  # (150,150)
                                                                  class_mode='binary')
    sample_training_images, _ = next(train_data_gen)
    # plot_images(sample_training_images[:5])
    augmented_images = [train_data_gen[0][0][1] for i in range(5)]
    # plot_images(augmented_images)

    EPOCHS = 100

    trained_model_name = 'model_augmentation_dropout.h5'
    history_name = 'history_augmentation_dropout.json'
    is_model_trained = os.path.isfile(trained_model_name)
    if is_model_trained is False:
        """Build model"""
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
            tf.keras.layers.MaxPooling2D(2, 2),

            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),

            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),

            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),

            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(2)
        ])

        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        model.summary()

        """Train the model"""
        history = model.fit(
            train_data_gen,
            epochs=EPOCHS,
            steps_per_epoch=int(np.ceil(total_train / float(BATCH_SIZE))),
            validation_data=val_data_gen,
            validation_steps=int(np.ceil(total_val / float(BATCH_SIZE)))
        )

        """Dump history"""
        # Get the dictionary containing each metric and the loss for each epoch
        history_dict = history.history
        # Save it under the form of a json file
        json.dump(history_dict, open(history_name, 'w'))

        """Save Model"""
        model.save(trained_model_name)
    else:
        """Reload Model"""
        model = tf.keras.models.load_model(trained_model_name)
        history_dict = json.load(open(history_name, 'r'))

    # Visualizing results of the training
    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']

    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs_range = range(EPOCHS)

    # plt.figure(figsize=(8, 8))
    # plt.subplot(1, 2, 1)
    # plt.plot(epochs_range, acc, label='Training Accuracy')
    # plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    # plt.legend(loc='lower right')
    # plt.title('Training and Validation Accuracy')
    #
    # plt.subplot(1, 2, 2)
    # plt.plot(epochs_range, loss, label='Training Loss')
    # plt.plot(epochs_range, val_loss, label='Validation Loss')
    # plt.legend(loc='upper right')
    # plt.title('Training and Validation Loss')
    # # plt.savefig('./foo.png')
    # plt.show()

    # img = [val_data_gen[0][0]]

    """Predict"""
    classes_indices = val_data_gen.class_indices
    reversed_classes_indices = dict([(value, key) for (key, value) in classes_indices.items()])

    print(val_data_gen.class_indices)

    cats_batch = 1
    dogs_batch = 9
    for _ in range(cats_batch):
        image_batch, label_batch = next(val_data_gen)

    # predicted_batch = model.predict(val_data_gen)
    predicted_batch = model.predict(image_batch)
    predicted_ids = np.argmax(predicted_batch, axis=1)
    print(predicted_ids)

    from_index = 4
    size = 9
    plot_images_with_labels(image_batch[from_index:size], predicted_ids[from_index:size],
                            label_batch[from_index:size],
                            reversed_classes_indices)
