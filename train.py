import os
import warnings
from os import PathLike

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import DataLoader
from torchvision import models, datasets, transforms
from tqdm import tqdm

# ------------------------------
# Hyperparameters etc.
# ------------------------------

LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
NUM_EPOCHS = 100
NUM_WORKERS = os.cpu_count()
PIN_MEMORY = True
DATA_DIR = "data/"

# Suppress warnings
warnings.filterwarnings("ignore")


# ------------------------------
# Utility Classes and Functions
# ------------------------------

class EarlyStopping:
    """
    Early stops the training if the validation loss doesn't improve after a given patience.
    """

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pth', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_val_loss = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        # If validation loss is not a number, skip
        if np.isnan(val_loss):
            self.trace_func("Validation loss is NaN. Ignoring this epoch.")
            return

        if self.best_val_loss is None:
            self.best_val_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss < self.best_val_loss - self.delta:
            self.best_val_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0
        else:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}\n')
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decreases."""
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...\n')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def train_step(model: torch.nn.Module,
               dataloader: DataLoader,
               optimizer: torch.optim.Optimizer,
               loss_fn: nn.Module = nn.CrossEntropyLoss(),
               device='cuda'):
    """
    Perform a single training step for the model.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): The DataLoader for training data.
        optimizer (Optimizer): The optimizer used to update the model parameters.
        loss_fn (nn.Module): The loss function, default is CrossEntropyLoss.
        device (str): The device to run the model on, 'cuda' or 'cpu'.

    Returns:
        train_loss (float): The total training loss for the step.
        train_acc (float): The average training accuracy for the step.
    """
    model.train()
    train_loss, train_acc, train_f1 = 0, 0, 0

    for batch, (X, y) in enumerate(pbar := tqdm(dataloader, total=len(dataloader), desc='Training')):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()

        # Forward pass
        y_pred = model(X).squeeze()
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        # Update metrics
        _, preds = torch.max(y_pred, 1)
        y = y.cpu().detach().numpy()
        preds = preds.cpu().detach().numpy()

        train_acc += accuracy_score(y, preds)
        train_f1 += f1_score(y, preds, average='macro')

        # Update pbar
        pbar.set_postfix(loss=train_loss / (batch + 1), acc=train_acc / (batch + 1), f1=train_f1 / (batch + 1))

    avg_loss = train_loss / len(dataloader)
    avg_acc = train_acc / len(dataloader)
    avg_f1 = train_f1 / len(dataloader)

    return avg_loss, (avg_acc, avg_f1)


def test_step(model: torch.nn.Module,
              dataloader: DataLoader,
              loss_fn: nn.Module = nn.CrossEntropyLoss(),
              device='cuda'):
    """
    Perform a single validation step for the model.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): The DataLoader for testing data.
        loss_fn (nn.Module): The loss function, default is CrossEntropyLoss.
        device (str): The device to run the model on, 'cuda' or 'cpu'.

    Returns:
        test_loss (float): The average test loss for the step.
        test_acc (float): The average test accuracy for the step.
    """
    model.eval()
    test_loss, test_acc, test_f1 = 0, 0, 0

    with torch.inference_mode():
        for batch, (X, y) in enumerate(pbar := tqdm(dataloader, total=len(dataloader), desc='Validation')):
            X, y = X.to(device), y.to(device)
            y_pred = model(X).squeeze()
            loss = loss_fn(y_pred, y)
            test_loss += loss.item()

            # Update metrics
            _, preds = torch.max(y_pred, 1)
            y = y.cpu().detach().numpy()
            preds = preds.cpu().detach().numpy()

            test_acc += accuracy_score(y, preds)
            test_f1 += f1_score(y, preds, average='macro')

            # Update pbar
            pbar.set_postfix(loss=test_loss / (batch + 1), acc=test_acc / (batch + 1), f1=test_f1 / (batch + 1))

    avg_loss = test_loss / len(dataloader)
    avg_acc = test_acc / len(dataloader)
    avg_f1 = test_f1 / len(dataloader)

    return avg_loss, (avg_acc, avg_f1)


def train_fn(
        model: torch.nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        loss_fn: nn.Module,
        device: torch.device = DEVICE,
        epochs: int = 10,
        best_checkpoint: str | PathLike[str] = "checkpoints/checkpoint.pth",
        patience: int = 5
):
    """
    Train and evaluate the model over multiple epochs and applies early stopping based on validation loss.

    Args:
        model (nn.Module): The neural network model.
        train_dataloader (DataLoader): The DataLoader for training data.
        val_dataloader (DataLoader): The DataLoader for testing/validation data.
        optimizer (Optimizer): The optimizer used to update the model parameters.
        scheduler (lr_scheduler): The learning rate scheduler to adjust the learning rate.
        loss_fn (nn.Module): The loss function.
        device (torch.device): The device to run the model on.
        epochs (int): The number of epochs to train the model.
        best_checkpoint (str): Path to save the model checkpoints.
        patience (int): Number of epochs to wait before early stopping.

    Returns:
        dict: A dictionary containing training and validation metrics for each epoch.
    """

    early_stopping = EarlyStopping(patience=patience, path=best_checkpoint, verbose=True)

    history = {
        "train_loss": [],
        "train_acc": [],
        "train_f1": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": []
    }

    for epoch in range(epochs):
        tqdm.write(f"\n======== Epoch {epoch + 1}/{epochs} ========")
        train_loss, (train_acc, train_f1) = train_step(model, train_dataloader, optimizer, loss_fn, device)
        val_loss, (val_acc, val_f1) = test_step(model, val_dataloader, loss_fn, device)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["train_f1"].append(train_f1)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        print(
            f"Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.6f}, Train F1: {train_f1:.6f} | Val Loss: {val_loss:.6f}, Val Acc: {val_acc:.6f}, Val F1: {val_f1:.6f}")

        # Step the learning rate scheduler
        scheduler.step()

        # Check early stopping condition
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break

    # Load the best model weights from checkpoint
    model.load_state_dict(torch.load(early_stopping.path))
    print("Training complete!")
    return history


# ------------------------------
# K-Fold Cross Validation Loop
# ------------------------------
# kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
# fold_results = {}
#
# for fold, (train_idx, val_idx) in enumerate(kfold.split(train_data)):
#     print(f"\n--- Fold {fold + 1}/{k_folds} ---")
#
#     # Create subsets for the current fold
#     train_subset = Subset(train_data, train_idx)
#     val_subset = Subset(train_data, val_idx)
#
#     # Create dataloaders for training and validation subsets
#     train_dataloader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=os.cpu_count())
#     val_dataloader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=os.cpu_count())
#
#     # Initialize a new instance of the model for this fold
#     num_classes = 3  # Set number of output classes
#
#     # Load pre-trained HRNet model
#     model = timm.create_model('hrnet_w18', pretrained=True, num_classes=num_classes)
#
#     # Move model to device
#     model = model.to(device)
#
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.5 * 1e-4)
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
#     loss_fn = nn.CrossEntropyLoss()
#
#     # Define checkpoint path for saving the best model in this fold
#     checkpoint_path = f"checkpoints/hrnet_1e-4_5_fold{fold + 1}.pth"
#     os.makedirs(name="checkpoints", exist_ok=True)
#
#     start_time = timer()
#     history = train(model=model,
#                     train_dataloader=train_dataloader,
#                     val_dataloader=val_dataloader,
#                     optimizer=optimizer,
#                     scheduler=scheduler,
#                     loss_fn=loss_fn,
#                     device=device,
#                     epochs=100,
#                     path=checkpoint_path,
#                     patience=5)
#     end_time = timer()
#     training_time = end_time - start_time
#     print(f"Fold {fold + 1} training time: {training_time:.2f} seconds")
#
#     # Store the training history for this fold
#     fold_results[f"fold_{fold + 1}"] = history
#
# # Optionally, save the fold results to a JSON file for later analysis
# with open("hrnet_kfold_1e-4_5_results.json", "w") as f:
#     json.dump(fold_results, f, indent=4)
#
# print("\nK-Fold Cross Validation complete!")


if __name__ == "__main__":
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    train_data = datasets.ImageFolder(root="__data", transform=train_transform)
    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    val_data = datasets.ImageFolder(root="__data", transform=val_transform)
    val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

    loss_fn = nn.CrossEntropyLoss()
    model = models.resnet18(num_classes=6).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=5)
    cosine = CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[5])

    train_fn(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=DEVICE,
        epochs=NUM_EPOCHS,
        best_checkpoint=f"checkpoints/best_model_{NUM_EPOCHS}.pth"
    )
