# Standard Libraries
import os
import logging
import warnings
from typing import List, Tuple, Dict, Union, Callable, Any

# Third-Party Libraries
import torch
from torch.utils.data import DataLoader, Dataset
import pickle
from tqdm import tqdm

# Application-specific

# ------------------------
# Constants & Configurations
# ------------------------

# Ignore Warnings
IGNORE_WARNINGS = True
if IGNORE_WARNINGS:
    warnings.filterwarnings("ignore")

# Logger Configurations
LOGGER_NAME = "probe"
logger = logging.getLogger(LOGGER_NAME)

# DEVICE Configurations
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.DEVICE("cuda" if USE_CUDA else "cpu")


def train_probe(
    embedding_func: Callable[..., Any],
    collator_fn: Callable[..., Any],
    train_dataset: Dataset,
    valid_dataset: Dataset,
    batch_size: int,
    patience: int,
    probe_model: torch.nn.Module,
    probe_loss: torch.nn.Module,
    model_under_probe: torch.nn.Module,
    train_epochs: int,
    output_path: str,
) -> None:
    """
    Train a probe model.

    Parameters:
        embedding_func (Callable): Function to generate embeddings from the model under probing.
        collator_fn (Callable): Function for collating batches of data.
        train_dataset (torch.utils.data.Dataset): Dataset for training.
        valid_dataset (torch.utils.data.Dataset): Dataset for validation.
        batch_size (int): Batch size.
        patience (int): Number of epochs to wait before early stopping.
        probe_model (torch.nn.Module): The probe model to be trained.
        probe_loss (torch.nn.Module): The loss function.
        model_under_probe (torch.nn.Module): The model that's being probed.
        train_epochs (int): Number of training epochs.
        output_path (str): Directory to save the trained probe model.

    Returns:
        None
    """
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator_fn,
        num_workers=0,
    )

    # Using Adam optimizer with learning rate of 1e-3
    optimizer = torch.optim.Adam(probe_model.parameters(), lr=1e-3)
    # Reduce learning rate when a metric has stopped improving.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.1,
        patience=0,
    )

    criterion = probe_loss

    probe_model.train()  # Set the probe model to training mode
    model_under_probe.eval()  # Set the model under probe to evaluation mode

    best_eval_loss = float("inf")
    metrics = {
        "training_loss": [],
        "validation_loss": [],
        "test_precision": None,
        "test_recall": None,
        "test_f1": None,
    }
    patience_count = 0

    for epoch in range(1, train_epochs + 1):
        training_loss = 0.0
        for step, batch in enumerate(
            tqdm(
                train_dataloader,
                desc="Training probe",
                bar_format="{desc:<10}{percentage:3.0f}%|{bar:100}{r_bar}",
            )
        ):
            ds, cs, us, batch_len_tokens, original_code_strings = batch

            embds = embedding_func(original_code_strings, model_under_probe, max_len=29)
            # d_pred, c_pred, u_pred = probe_model(embds.permute(0, 2, 1).to(DEVICE)) #! THIS SHOULD BE TURNED ON FOR ASTNN UNTIL I FIND A FIX
            d_pred, c_pred, u_pred = probe_model(embds.to(DEVICE))

            loss = criterion(
                d_pred=d_pred.to(DEVICE),
                c_pred=c_pred.to(DEVICE),
                u_pred=u_pred.to(DEVICE),
                d_real=torch.tensor(ds).to(DEVICE),
                c_real=torch.tensor(cs).to(DEVICE),
                u_real=torch.tensor(us).to(DEVICE),
                length_batch=batch_len_tokens.to(DEVICE),
            )

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            training_loss += loss.item()

        training_loss = training_loss / len(train_dataloader)
        eval_loss, acc_d, acc_c, acc_u = eval_probe(
            embedding_func,
            collator_fn,
            valid_dataset,
            batch_size,
            probe_model,
            probe_loss,
            model_under_probe,
        )
        scheduler.step(eval_loss)
        logger.info(
            f"[epoch {epoch}] train loss: {round(training_loss, 4)}, validation loss: {round(eval_loss, 4)}"
        )
        logger.info(
            f"[epoch {epoch}] D accuracy: {round(acc_d, 4)}, C accuracy: {round(acc_c, 6)}, U accuracy: {round(acc_u, 4)}"
        )
        metrics["training_loss"].append(round(training_loss, 4))
        metrics["validation_loss"].append(round(eval_loss, 4))

        if eval_loss < best_eval_loss:
            logger.info("Saving model checkpoint")
            model_save_path = os.path.join(output_path, f"pytorch_model.bin")
            torch.save(probe_model.state_dict(), model_save_path)
            logger.info(f"Probe model saved: {output_path}")
            patience_count = 0
            best_eval_loss = eval_loss
        else:
            patience_count += 1
        # Implement early stopping if validation loss doesn't improve after 'patience' epochs
        if patience_count == patience:
            logger.info("Stopping training loop (out of patience).")
            break


def eval_probe(
    embedding_func: Callable[..., Any],
    collator_fn: Callable[..., Any],
    eval_dataset: torch.utils.data.Dataset,
    batch_size: int,
    probe_model: torch.nn.Module,
    probe_loss: torch.nn.Module,
    model_under_probe: torch.nn.Module,
) -> Tuple[float, float, float, float]:
    """
    Evaluate a probe model.

    Parameters:
        embedding_func (Callable): Function to generate embeddings from the model under probing.
        collator_fn (Callable): Function for collating batches of data.
        eval_dataset (torch.utils.data.Dataset): Dataset for evaluation.
        batch_size (int): Batch size.
        probe_model (torch.nn.Module): The probe model to be evaluated.
        probe_loss (torch.nn.Module): The loss function.
        model_under_probe (torch.nn.Module): The model that's being probed.

    Returns:
        Tuple[float, float, float, float]: A tuple containing the evaluation loss and three types of accuracies (D, C, U).
    """
    probe_model.eval()

    eval_loss = 0.0

    valid_dataloader = DataLoader(
        dataset=eval_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collator_fn(batch),
        num_workers=0,
    )

    d_hits_total = []
    c_hits_total = []
    u_hits_total = []

    with torch.no_grad():
        for step, batch in enumerate(
            tqdm(
                valid_dataloader,
                desc="Testing  probe",
                bar_format="{desc:<10}{percentage:3.0f}%|{bar:100}{r_bar}",
            )
        ):
            ds, cs, us, batch_len_tokens, original_code_strings = batch

            ds = ds.to(DEVICE)
            cs = cs.to(DEVICE)
            us = us.to(DEVICE)

            embds = embedding_func(original_code_strings, model_under_probe, max_len=29)

            # d_pred, c_pred, u_pred = probe_model(embds.permute(0, 2, 1).to(DEVICE)) #! THIS SHOULD BE TURNED ON FOR ASTNN UNTIL I FIND A FIX
            d_pred, c_pred, u_pred = probe_model(embds.to(DEVICE))

            loss = probe_loss(
                d_pred=d_pred.to(DEVICE),
                c_pred=c_pred.to(DEVICE),
                u_pred=u_pred.to(DEVICE),
                d_real=torch.tensor(ds).to(DEVICE),
                c_real=torch.tensor(cs).to(DEVICE),
                u_real=torch.tensor(us).to(DEVICE),
                length_batch=batch_len_tokens.to(DEVICE),
            )

            eval_loss += loss.item()

            d_hits, c_hits, u_hits, d_hits_len, c_hits_len = probe_loss.calculate_hits(
                d_pred, c_pred, u_pred, ds, cs, us, batch_len_tokens
            )

            d_hits_total.append((d_hits, d_hits_len))
            c_hits_total.append((c_hits, c_hits_len))
            u_hits_total.append((u_hits, d_hits_len))

        # Computing accuracy for 'D', 'C', and 'U' from hit counts
        total_accuracy_d = sum([x[0] for x in d_hits_total]) / sum(
            x[1] for x in d_hits_total
        )
        total_accuracy_c = sum([x[0] for x in c_hits_total]) / sum(
            x[1] for x in c_hits_total
        )
        total_accuracy_u = sum([x[0] for x in u_hits_total]) / sum(
            x[1] for x in u_hits_total
        )

        return (
            (eval_loss / len(valid_dataloader)),
            total_accuracy_d.data.item(),
            total_accuracy_c.data.item(),
            total_accuracy_u.data.item(),
        )
