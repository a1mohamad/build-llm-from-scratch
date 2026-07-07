"""Distributed Data Parallel toy example for the PyTorch appendix.

This script mirrors the DDP section from the annotated notebook, but it is designed
for terminal execution. DDP is usually more reliable in a standalone Python file
than inside Jupyter notebooks, especially on Windows.

Run:
    python multiple_gpus.py
"""

import os

import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


SEED = 28
BATCH_SIZE = 2
NUM_EPOCHS = 3


class NeuralNetwork(torch.nn.Module):
    """Small fully connected classifier used for the appendix examples."""

    def __init__(self, num_inputs: int, num_outputs: int) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(num_inputs, 30),
            torch.nn.ReLU(),
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, num_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw class logits for a batch of examples."""
        return self.layers(x)


class ToyDataset(Dataset):
    """Minimal tensor-backed dataset for classification."""

    def __init__(self, X: torch.Tensor, y: torch.Tensor) -> None:
        self.features = X
        self.labels = y

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


def build_datasets() -> tuple[ToyDataset, ToyDataset]:
    """Create the tiny train/test datasets used in the appendix."""

    X_train = torch.tensor(
        [
            [-1.2, 3.1],
            [-0.9, 2.9],
            [-0.5, 2.6],
            [2.3, -1.1],
            [2.7, -1.5],
        ]
    )
    y_train = torch.tensor([0, 0, 0, 1, 1])

    X_test = torch.tensor(
        [
            [-0.8, 2.8],
            [2.6, -1.6],
        ]
    )
    y_test = torch.tensor([0, 1])

    return ToyDataset(X_train, y_train), ToyDataset(X_test, y_test)


def ddp_setup(rank: int, world_size: int) -> None:
    """Initialize distributed communication for one worker process."""

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"

    backend = "nccl" if torch.cuda.is_available() else "gloo"

    init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
    )

    if torch.cuda.is_available():
        torch.cuda.set_device(rank)


def prepare_dataloaders(train_ds: Dataset, test_ds: Dataset) -> tuple[DataLoader, DataLoader]:
    """Create dataloaders that shard data across DDP processes."""

    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,  # DistributedSampler controls shuffling.
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        sampler=DistributedSampler(train_ds),
    )

    test_loader = DataLoader(
        dataset=test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        sampler=DistributedSampler(test_ds, shuffle=False),
    )

    return train_loader, test_loader


def compute_accuracy(model: torch.nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    """Compute classification accuracy for the local process data shard."""

    correct = 0
    total_examples = 0

    model.eval()

    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device)

            logits = model(features)
            preds = torch.argmax(logits, dim=1)

            correct += torch.sum(preds == labels).item()
            total_examples += labels.shape[0]

    return correct / total_examples


def train(rank: int, world_size: int, num_epochs: int) -> None:
    """Train the toy model in one DDP worker process."""

    ddp_setup(rank, world_size)
    torch.manual_seed(SEED)

    train_ds, test_ds = build_datasets()
    train_loader, test_loader = prepare_dataloaders(train_ds, test_ds)

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    model = NeuralNetwork(num_inputs=2, num_outputs=2).to(device)
    device_ids = [rank] if torch.cuda.is_available() else None
    model = DDP(model, device_ids=device_ids)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

    for epoch in range(num_epochs):
        model.train()
        train_loader.sampler.set_epoch(epoch)

        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)

            logits = model(features)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(
                f"[Rank {rank}] Epoch: {epoch + 1:03d}/{num_epochs:03d}"
                f" | Batch size: {labels.shape[0]:03d}"
                f" | Train Loss: {loss:.2f}"
            )

    train_acc = compute_accuracy(model.module, train_loader, device=device)
    test_acc = compute_accuracy(model.module, test_loader, device=device)

    print(f"[Rank {rank}] Training Accuracy: {train_acc:.3f}")
    print(f"[Rank {rank}] Test Accuracy: {test_acc:.3f}")

    destroy_process_group()


def main() -> None:
    """Launch one worker process per available CUDA GPU, or one CPU worker."""

    if torch.cuda.is_available():
        world_size = torch.cuda.device_count()
    else:
        world_size = 1
        print("CUDA is not available. Running a single-process CPU DDP demo with Gloo.")

    print(f"Number of DDP processes: {world_size}")
    mp.spawn(train, args=(world_size, NUM_EPOCHS), nprocs=world_size)


if __name__ == "__main__":
    main()
