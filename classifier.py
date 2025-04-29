import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torch import nn, optim

from ann_model import FashionMNISTANN
from train import train_model
from evaluate import evaluate_model
from predict import preprocess_image, predict
from utils import get_class_names
from plot_training_curves import plot_training_curves
from save_log import save_log_to_txt

def main():
    DATA_DIR = "."

    # Optionally clear logs.txt before starting
    open("logs.txt", "w").close()
    print("logs.txt cleared!")

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    transform = transforms.ToTensor()
    train_data = datasets.FashionMNIST(DATA_DIR, train=True, download=False, transform=transform)
    test_data = datasets.FashionMNIST(DATA_DIR, train=False, download=False, transform=transform)

    # Split into training and validation sets
    train_size = int(0.8 * len(train_data))
    val_size = len(train_data) - train_size
    train_subset, val_subset = random_split(train_data, [train_size, val_size])

    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    # Hyper-parameters
    hidden_layers = 2
    hidden_nodes = 128
    activation = 'relu'
    dropout = 0.0
    optimizer_choice = 'sgd'
    learning_rate = 0.01
    loss_function_choice = 'cross_entropy'
    epochs = 20
    early_stopping_patience = 3

    # Model setup
    if loss_function_choice == 'nll':
        output_activation = 'logsoftmax'
    else:
        output_activation = 'none'

    model = FashionMNISTANN(
        hidden_layers=hidden_layers,
        hidden_nodes=hidden_nodes,
        activation=activation,
        dropout=dropout,
        output_activation=output_activation
    ).to(device)

    # Loss function
    if loss_function_choice == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
    elif loss_function_choice == 'nll':
        criterion = nn.NLLLoss()
    else:
        raise ValueError(f"Invalid loss function choice: {loss_function_choice}")

    # Optimizer
    if optimizer_choice == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer_choice == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Invalid optimizer choice: {optimizer_choice}")

    # Train & Validate
    train_losses, val_losses, val_accuracies = train_model(
        model, train_loader, val_loader, criterion, optimizer, device, epochs, early_stopping_patience
    )

    plot_training_curves(train_losses, val_losses, val_accuracies)

    # Test Evaluation
    test_accuracy = evaluate_model(model, test_loader, device)

    # Save experiment logs
    config = {
        "hidden_layers": hidden_layers,
        "hidden_nodes": hidden_nodes,
        "activation": activation,
        "dropout": dropout,
        "optimizer_choice": optimizer_choice,
        "learning_rate": learning_rate,
        "loss_function_choice": loss_function_choice,
        "epochs": epochs,
        "early_stopping_patience": early_stopping_patience
    }

    save_log_to_txt(config, val_losses[-1], val_accuracies[-1], test_accuracy)

    # Interactive image classification loop
    class_names = get_class_names()
    print("Done!\nPlease enter a filepath:")
    while True:
        filepath = input("> ")
        if filepath.lower() == 'exit':
            print("Exiting...")
            break
        try:
            image_tensor = preprocess_image(filepath).to(device)
            label = predict(model, image_tensor, class_names)
            print(f"Classifier: {label}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
