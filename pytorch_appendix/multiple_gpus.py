import torch
import torch.nn.functional as F
import os
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

X_train = torch.tensor([
    [-1.2, 3.1],
    [-0.9, 2.9],
    [-0.5, 2.6],
    [2.3, -1.1],
    [2.7, -1.5]
])
y_train = torch.tensor([0, 0, 0, 1, 1])
X_test = torch.tensor([
    [-0.8, 2.8],
    [2.6, -1.6]
])
y_test = torch.tensor([0, 1])

class ToyDataset(Dataset):
    def __init__(self, X, y):
        self.features = X
        self.labels = y

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        one_x = self.features[idx]
        one_y = self.labels[idx]
        return one_x, one_y

train_ds = ToyDataset(X_train, y_train)
test_ds = ToyDataset(X_test, y_test)

class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(num_inputs, 30),
            torch.nn.ReLU(),

            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),

            torch.nn.Linear(20, num_outputs)
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits

def compute_accuracy(model, dataloader, device):
    correct = 0.0
    total_examples = 0
    model = model.eval()
    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device)
            logits = model(features)
            preds = torch.argmax(logits, dim=1)
            compare = preds == labels
            correct += torch.sum(compare)
            total_examples += len(compare)
    return (correct / total_examples).item()

def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size
    )
    torch.cuda.set_device(rank)

def prepare_dataset():
    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=2,
        num_workers=0,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
        sampler=DistributedSampler(train_ds)
    )
    test_loader = DataLoader(
        dataset=test_ds,
        num_workers=0,
        batch_size=2,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        sampler=DistributedSampler(test_ds, shuffle=False)
    )

    return train_loader, test_loader

def main(rank, world_size, num_epochs):
    ddp_setup(rank, world_size)
    train_loader, test_loader = prepare_dataset()
    model = NeuralNetwork(num_inputs=2, num_outputs=2).to(rank)
    model = DDP(model, device_ids=[rank])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    for epoch in range(num_epochs):
        train_loader.sampler.set_epoch(epoch)
        model.train()
        for features, labels in train_loader:
            features, labels = features.to(rank), labels.to(rank)
            optimizer.zero_grad()
            logits = model(features)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            ##LOGGING
            print(f"[GPU{rank}] Epoch: {epoch+1:03d}/{num_epochs:03d}"
              f" | Batchsize {labels.shape[0]:03d}"
              f" | Train/Val Loss: {loss:.2f}")

    model.eval()
    train_acc = compute_accuracy(model, train_loader, device=rank)
    print(f"[GPU{rank}] Training Accuracy", train_acc)
    test_acc = compute_accuracy(model, test_loader, device=rank)
    print(f"[GPU{rank}] Test Accuracy", test_acc)
    destroy_process_group()


if __name__ == "__main__":
    print(f"Number GPUs available on Device: {torch.cuda.device_count()}")
    torch.manual_seed(28)
    num_epochs = 3
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, num_epochs), nprocs=world_size)
