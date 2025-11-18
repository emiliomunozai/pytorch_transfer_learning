import torch
import torch.nn as nn
import random
from tqdm import tqdm
from src.models import build_inception
from src.data_loaders import make_dataloaders


# ================================
# Regularization functions
# ================================
def l1_regularization(model):
    return sum(p.abs().sum() for p in model.parameters() if p.requires_grad)


# ================================
# Training & Evaluation
# ================================
def train_one_epoch(model, loader, optimizer, criterion, device, l1_lambda=0.0):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        if isinstance(outputs, tuple):
            logits, aux_logits = outputs[:2]  # Handle possible multiple aux
            loss_main = criterion(logits, labels)
            loss_aux = criterion(aux_logits, labels)
            loss = loss_main + 0.4 * loss_aux
        else:
            logits = outputs
            loss = criterion(logits, labels)

        # Add L1 regularization if enabled
        if l1_lambda > 0:
            l1_reg = l1_regularization(model)
            loss += l1_lambda * l1_reg

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    acc = 100.0 * correct / total
    return running_loss / len(loader), acc


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs

            loss = criterion(logits, labels)
            running_loss += loss.item()

            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    acc = 100.0 * correct / total
    return running_loss / len(loader), acc


# ================================
# Full training loop
# ================================
def train_model(model, model_name, train_loader, val_loader, criterion, optimizer, device, epochs, l1_lambda=0.0):
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, l1_lambda)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

    # Save model
    torch.save(model, f"models/{model_name}_model.pth")

    return train_losses, val_losses, train_accs, val_accs


# ================================
# Random Search Pipeline (CORREGIDO para tu grid con "optimizers" y "regularizers")
# ================================
def random_search_pipeline(
    param_grid,
    train_ds,
    val_ds,
    test_ds,
    epochs,
    n_iter=10,
    n_classes=10
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_acc = 0.0
    best_experiment_name = None
    experiment_results = []

    for i in range(n_iter):
        # Sample unique config
        config = {k: random.choice(v) for k, v in param_grid.items()}

        # Avoid exact duplicates
        while any(
            config["batch_size"] == e.get("batch_size") and
            config["learning_rate"] == e.get("learning_rate") and
            config["optimizers"] == e.get("optimizers") and
            config["regularizers"] == e.get("regularizers")
            for e in experiment_results
        ):
            config = {k: random.choice(v) for k, v in param_grid.items()}

        print(f"\nTrial {i+1}/{n_iter}: {config}")

        experiment_name = "inception_" + "_".join(f"{k}_{v}" if k != "optimizers" else f"optim_{v.__name__}" 
                                                 for k, v in config.items())

        # Data loaders
        train_loader, val_loader, test_loader, _, _ = make_dataloaders(
            train_ds, val_ds, test_ds, 
            batch_size=config["batch_size"], 
            model_name="inception", 
            augment=True
        )

        # Model & loss
        model = build_inception(num_classes=n_classes).to(device)
        criterion = nn.CrossEntropyLoss()

        # === OPTIMIZER + REGULARIZATION ===
        l1_lambda = 0.0
        opt_params = {"lr": config["learning_rate"]}

        regularizer = config["regularizers"]  # ← aquí usamos tu nombre correcto

        if regularizer == "l2":
            opt_params["weight_decay"] = 0.01
        elif regularizer == "l1":
            l1_lambda = 0.01
        # "none" → no hacemos nada

        optimizer = config["optimizers"](model.parameters(), **opt_params)  # ← plural correcto

        # Train
        train_losses, val_losses, train_accs, val_accs = train_model(
            model, experiment_name, train_loader, val_loader,
            criterion, optimizer, device, epochs, l1_lambda=l1_lambda
        )

        # Test
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        # Track best
        if test_acc > best_acc:
            best_acc = test_acc
            best_experiment_name = experiment_name

        # Save results
        experiment_results.append({
            "experiment_name": experiment_name,
            "batch_size": config["batch_size"],
            "learning_rate": config["learning_rate"],
            "optimizer": config["optimizers"].__name__,
            "regularizer": config["regularizers"],
            "test_accuracy": test_acc,
            "test_loss": test_loss,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_accuracies": train_accs,
            "val_accuracies": val_accs,
        })

        print(f"→ Test Accuracy: {test_acc:.2f}% | Best so far: {best_acc:.2f}%\n")

    print(f"\nBEST MODEL: {best_experiment_name} → {best_acc:.2f}% test accuracy")
    return best_experiment_name, experiment_results