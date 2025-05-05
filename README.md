# FashionMNIST ANN Classifier

This project implements a **feedforward Artificial Neural Network (ANN)** in PyTorch to classify grayscale fashion product images from the **FashionMNIST** dataset. The code allows you to experiment with different hyperparameter settings and observe their impact on validation and test performance.

## Project Structure

| File | Description |
|------|-------------|
| `classifier.py` | The **main script** to run the ANN. It trains the model, validates it, saves logs of experiments, and allows **interactive image classification** after training. |
| `ann_model.py` | Defines the architecture of the ANN. It allows for **customizable hidden layers, activation functions, dropout, and output layers**. |
| `train.py` | Contains the **training logic**. It performs model training, validation after each epoch, and implements **early stopping** based on validation loss. |
| `evaluate.py` | Defines a function to **evaluate the trained model on the test set**, reporting accuracy. |
| `predict.py` | Handles **image preprocessing and prediction** for custom images supplied after training is done. |
| `utils.py` | Provides utility functions, such as getting the **class names** for FashionMNIST labels. |
| `plot_training_curves.py` | Reads training logs and **plots training/validation loss and accuracy curves** for visual analysis. |
| `save_log.py` | Saves each experiment's **configuration and results** into `logs.txt` in a clear, human-readable format. |



## How to Run

### Make sure your environment is ready:

- Install required packages (best in a virtual environment):

pip install -r requirements.txt

### Run the main script:

python classifier.py


### Setting Hyper-Parameters

# Hyper-parameters
hidden_layers = 3
hidden_nodes = 512
activation = 'relu'
dropout = 0.2
optimizer_choice = 'sgd'
learning_rate = 0.01
loss_function_choice = 'nll'
epochs = 20
early_stopping_patience = 5

You can customize these parameters before running the code.
| Parameter                 | Options / Examples                           |
| ------------------------- | -------------------------------------------- |
| `hidden_layers`           | Number of hidden layers (e.g., 2, 3, 4)      |
| `hidden_nodes`            | Nodes per hidden layer (e.g., 128, 256, 512) |
| `activation`              | `'relu'` or `'leaky_relu'`                   |
| `dropout`                 | Dropout rate (e.g., 0.0, 0.2, 0.5)           |
| `optimizer_choice`        | `'sgd'` or `'adam'`                          |
| `learning_rate`           | Learning rate (e.g., 0.01, 0.001, 0.0005)    |
| `loss_function_choice`    | `'cross_entropy'` or `'nll'`                 |
| `epochs`                  | Number of training epochs (e.g., 10, 20)     |
| `early_stopping_patience` | Early stopping patience (e.g., 3, 5)         |


### Logs and Results
After running, your results will be automatically saved to logs.txt.

Each entry logs:

The hyperparameter configuration.

Final validation loss & accuracy.

Final test accuracy.

### Plotting Training Curves
To visualize training and validation losses & accuracies, use the plotting script:

python plot_training_curves.py


### How to Classify Your Own Image
Once training & evaluation are complete, the script enters an interactive mode:

Done!
Please enter a filepath:

Here, you can provide the relative path to a grayscale image (28x28 pixels) for prediction.

Example from the fashion-jpegs directory:

fashion-jpegs/bag.jpg