# FashionMNIST ANN Classifier

This project implements a **feedforward Artificial Neural Network (ANN)** in PyTorch to classify grayscale fashion product images from the **FashionMNIST** dataset. The code allows you to experiment with different hyperparameter settings and observe their impact on validation and test performance.

## 📂 Project Structure

.
├── ann_model.py
├── classifier.py
├── evaluate.py
├── predict.py
├── train.py
├── utils.py
├── plot_training_curves.py
├── loss_accuracy_plot.png # (auto-generated) Contains generated plots for experiments
├── logs.txt # Logs of experiments (auto-generated)
├── requirements.txt
└── README.md


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

Here, you can provide the path to a grayscale image (28x28 pixels) for prediction.

Example: