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

    # Load data
    transform = transforms.ToTensor()
    train_data = datasets.FashionMNIST(DATA_DIR, train=True, download=False, transform=transform)
    test_data = datasets.FashionMNIST(DATA_DIR, train=False, download=False, transform=transform)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    # Initialize model, loss, optimizer
    model = FashionMNISTANN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Train & Evaluate
    train_model(model, train_loader, criterion, optimizer)
    evaluate_model(model, test_loader)

    # Interactive loop
    class_names = get_class_names()
    print("Done!\nPlease enter a filepath:")
    while True:
        filepath = input("> ")
        if filepath.lower() == 'exit':
            print("Exiting...")
            break
        try:
            image_tensor = preprocess_image(filepath)
            label = predict(model, image_tensor, class_names)
            print(f"Classifier: {label}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
