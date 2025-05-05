import torch

def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs=20,
    early_stopping_patience=3,
    log_file="logs.txt"
):
    best_loss = float('inf')
    patience_counter = 0

    train_losses = []
    val_losses = []
    val_accuracies = []

    with open(log_file, "a") as f:
        f.write("===== New Training Run =====\n")

    for epoch in range(epochs):
        running_loss = 0.0
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)

        log_line = (
            f"Epoch [{epoch + 1}/{epochs}] - "
            f"Train Loss: {avg_train_loss:.4f} - "
            f"Val Loss: {avg_val_loss:.4f} - "
            f"Val Accuracy: {val_accuracy:.2f}%\n"
        )

        print(log_line.strip())

        # Write each epoch's log to file
        with open(log_file, "a") as f:
            f.write(log_line)

        # Early stopping
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                with open(log_file, "a") as f:
                    f.write("Early stopping triggered.\n")
                break

    return train_losses, val_losses, val_accuracies
