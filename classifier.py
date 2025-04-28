import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn, optim

from ann_model import FashionMNISTANN
from train import train_model
from evaluate import evaluate_model
from predict import preprocess_image, predict
from utils import get_class_names

def main():
    DATA_DIR = "."

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    transform = transforms.ToTensor()
    train_data = datasets.FashionMNIST(DATA_DIR, train=True, download=False, transform=transform)
    test_data = datasets.FashionMNIST(DATA_DIR, train=False, download=False, transform=transform)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    # Hyper-parameters
    hidden_layers = 3
    hidden_nodes = 256
    activation = 'leaky_relu'
    dropout = 0.2
    epochs = 20
    early_stopping_patience = 3 

    # Initialize model, loss, optimizer
    model = FashionMNISTANN(hidden_layers=hidden_layers, hidden_nodes=hidden_nodes, activation=activation, dropout=dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Train & Evaluate
    train_model(model, train_loader, criterion, optimizer, device, epochs, early_stopping_patience)

    evaluate_model(model, test_loader, device)

    # Interactive loop
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
