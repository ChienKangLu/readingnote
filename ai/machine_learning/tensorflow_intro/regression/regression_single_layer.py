import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

if __name__ == '__main__':
    """0. Set seed for TensorFlow and Numpy to ensure that your code is repeatable"""
    np.random.seed(42)
    tf.keras.utils.set_random_seed(42)

    """1. Define data"""
    celsius_q = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
    fahrenheit_a = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

    """2. Build Layer"""
    l0 = tf.keras.layers.Dense(units=1, input_shape=[1])

    """3. Assemble layers into the model"""
    model = tf.keras.Sequential([l0])
    model.summary()

    """4. Compile model"""
    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.legacy.Adam(0.1))
    history = model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False)

    """5. Predict"""
    print(model.predict([100.0]))

    """Display training statistic"""
    plt.xlabel('Epoch Number')
    plt.ylabel("Loss Magnitude")
    plt.plot(history.history['loss'])
    # plt.show()
