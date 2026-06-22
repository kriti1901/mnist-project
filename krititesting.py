import os
# Configure Keras 3 to use the PyTorch backend BEFORE importing keras!
os.environ["KERAS_BACKEND"] = "torch"

import keras
import torch
import numpy as np
import matplotlib.pyplot as plt

print(f"Keras version: {keras.__version__}")
print(f"Using backend: {keras.config.backend()}")
print(f"PyTorch version: {torch.__version__}")
print(f"GPU available: {torch.cuda.is_available()}")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

plt.imshow(x_train[500], cmap='gray')
plt.show()
print(x_train[500])