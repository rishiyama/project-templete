import argparse
import os
import json
from datetime import datetime
import shutil

import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from tqdm.auto import tqdm


def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc="Training", leave=False):
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)


def validate_one_epoch(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", leave=False):
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(dataloader)


def run(cfg, model, loss_fn, train_dataset, val_dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.train.batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.learning_rate)

    log_file_path = os.path.join(cfg.path.save_dir, "log.jsonl")
    for epoch in range(cfg.train.epochs):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss = validate_one_epoch(model, val_loader, loss_fn, device)

        log_entry = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        with open(log_file_path, "a") as log_file:
            log_file.write(json.dumps(log_entry) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model.")
    parser.add_argument("--config", type=str, default="./configs/config.yaml",
                        help="Path to the configuration file (default: ./configs/config.yaml)")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    cfg.config_path = args.config
    cfg.path.save_dir = os.path.join(
        cfg.path.save_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(cfg.path.save_dir, exist_ok=True)

    OmegaConf.save(cfg, os.path.join(cfg.path.save_dir, "config.yaml"))
    shutil.copy(os.path.realpath(__file__), os.path.join(cfg.path.save_dir, os.path.basename(__file__)))

    # Placeholder for model, loss function, and datasets
    model = None
    loss_fn = None
    train_dataset = None
    val_dataset = None

    run(cfg, model, loss_fn, train_dataset, val_dataset)
