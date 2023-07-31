import platform
import pip

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D

if __name__ == '__main__':
    print("Python: {}".format(platform.python_version()))
    print("Pip: {}".format(pip.__version__))
    print("TensorFlow: {}".format(tf.__version__))
