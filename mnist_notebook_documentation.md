# MNIST Digit Classifier (Keras 3 / PyTorch Backend): original Project Documentation

## Project Name
**MNIST Digit Classifier & Weight Visualizer (Keras 3 with PyTorch Backend)**

---

## Technology Used
- **Python**: Core programming language.
- **Keras 3**: Unified high-level deep learning API configured with a PyTorch backend.
- **PyTorch**: Native tensor operations and backpropagation engine.
- **NumPy**: Data manipulation, array casting, and metric computations.
- **Matplotlib**: Generation of static visual graphs, grid subplots, training curves, and diverging colormaps (`RdYlGn` and `viridis`).
- **Jupyter Notebook / Google Colab**: Execution runtime environment.

---

## Technical Description
This project trains a feedforward neural network (Multilayer Perceptron) on the MNIST dataset of handwritten digits, utilizing the PyTorch execution backend via Keras 3. The pipeline consists of the following key phases:

1. **Backend Configuration**:
   - Programmatically sets `os.environ["KERAS_BACKEND"] = "torch"` prior to importing Keras, binding Keras commands to PyTorch tensor operations.

2. **Dataset Load & Normalization**:
   - Loads the MNIST database (60,000 training images, 10,000 test images).
   - Normalizes image matrices from uint8 integer values $[0, 255]$ to float32 values in range $[0.0, 1.0]$.

3. **Model Architecture**:
   - Defines a sequential architecture: Input layer ($28 \times 28$) $\rightarrow$ Flattening layer ($784$) $\rightarrow$ Hidden Layer 1 ($16$ neurons) $\rightarrow$ Hidden Layer 2 ($16$ neurons) $\rightarrow$ Output Layer ($10$ classification neurons).
   - Configures the final layer with Softmax activation to convert raw logits into output probability distributions.

4. **Model Compilation & Training**:
   - Compiles the network using the `Adam` optimizer, `sparse_categorical_crossentropy` loss, and tracking the `accuracy` metric.
   - Fits the model for 15 epochs, batch size 64, reserving a 20% validation split. Plots training curves to track loss reduction and accuracy gains.

5. **Weight Extraction & Spatial Heatmap Visualization**:
   - Extracts weight matrices ($784 \times 16$) and bias vectors ($16$) of the first hidden layer.
   - Reshapes the 784 weights of each hidden neuron back into a $28 \times 28$ matrix.
   - Plots the weights using a diverging `RdYlGn` colormap, showing spatial positive features (green) and negative features (red) mapped out by each neuron.

6. **Functional Submodel Activation Flow**:
   - Creates a Keras functional `Model(inputs=model.input, outputs=[...])` that outputs intermediate layer activations.
   - Visualizes the activation flow of single test samples: displaying the input image, 4x4 viridis heatmaps of hidden layer activations, and output classification confidence bar charts.

7. **Sigmoid vs. ReLU Performance Comparison**:
   - Re-builds and compiles the model using Rectified Linear Unit (ReLU) activations in the hidden layers instead of Sigmoid.
   - Evaluates and plots training rates, showing that the constant gradient flow of ReLU ($1.0$ for positive inputs) circumvents the vanishing gradient problem and achieves higher test accuracy ($95\text{--}97\%$) compared to Sigmoid ($92\text{--}94\%$).

---

## Uses
- **Keras 3 Multi-Backend Education**: Illustrates how Keras 3 provides a unified framework to execute PyTorch-backed code.
- **Academic & Explanatory Demos**: Translates mathematical abstractions (like weights, biases, activations, and backpropagation) into visual patterns, aligning with Chapters 1-4 of the *3Blue1Brown* neural network series.
- **Activation Performance Audits**: Analyzes the impact of activation functions (Sigmoid vs. ReLU) on saturation and optimization speed.

---

## Future Enhancements
- **Convolutional Network Expansion**: Switch the model from an MLP to a CNN structure (using `layers.Conv2D` and `layers.MaxPooling2D`) to capture spatial features and improve accuracy to $>99\%$.
- **Custom Callback Metrics**: Implement custom callback metrics in PyTorch/Keras to monitor layer-wise gradient norm distributions during training.
- **Model Quantization**: Apply post-training integer quantization to shrink model size for low-latency browser or edge device deployment.
- **Dynamic Training Rate Schedule**: Integrate a learning rate decay schedule (like CosineDecay) to fine-tune final convergence epochs.
