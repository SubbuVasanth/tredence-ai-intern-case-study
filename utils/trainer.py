import torch
import torch.nn as nn
from tqdm import tqdm


def run_training(
    model,
    train_loader,
    test_loader,
    lambd: float,
    device: torch.device,
    epochs: int = 15,
    warmup_epochs: int = 3,
) -> tuple:

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()


        if epoch < warmup_epochs:
            effective_lambda = lambd * (epoch + 1) / warmup_epochs
        else:
            effective_lambda = lambd

        running_class_loss   = 0.0
        running_sparsity_loss = 0.0

        pbar = tqdm(train_loader, desc=f"λ={lambd:.0e} | Epoch {epoch+1:02d}/{epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)


            class_loss = criterion(outputs, labels)

            gates_list   = model.get_gates()
            sparsity_loss = sum(torch.sum(torch.abs(g)) for g in gates_list)

            total_loss = class_loss + effective_lambda * sparsity_loss
            total_loss.backward()
            optimizer.step()

            running_class_loss    += class_loss.item()
            running_sparsity_loss += sparsity_loss.item()

            pbar.set_postfix(
                cls=f"{class_loss.item():.3f}",
                spar=f"{sparsity_loss.item():.1f}",
                eff_lam=f"{effective_lambda:.1e}",
            )

        scheduler.step()


    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs   = model(images)
            _, predicted = torch.max(outputs, 1)
            total   += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100.0 * correct / total


    stats = model.sparsity_stats(threshold=0.1)

    return accuracy, stats["overall"], stats["gate_values"]