import numpy as np
import tensorflow as tf

if __name__ == '__main__':
    p = [1.0, 9.0]
    t = [0]
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    print(loss(t, p).numpy())

    """Apply softmax automatically for getting normalized probability"""
    zero = np.exp(1)
    one = np.exp(9)
    total = zero + one

    zero_normalize = zero / total
    one_normalize = one / total

    normalized = [zero_normalize, one_normalize]
    print(normalized)

    """Calculate binary cross entropy"""
    def binary_cross_entropy(target, one_probability):
        print("target: {}".format(target))
        print("one_probability: {}".format(one_probability))
        zero_probability = 1 - one_probability
        print(-(target * np.log(one_probability) + (1 - target) * np.log(zero_probability)))

    binary_cross_entropy(t[0], normalized[1])
