import os, logging, pickle, torch, warnings
from torch.utils.data import DataLoader
from tqdm import tqdm


warnings.filterwarnings("ignore")

logger = logging.getLogger("probe")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_probe(
    embedding_func: callable,
    collator_fn: callable,
    train_dataset: torch.utils.data.Dataset,
    valid_dataset: torch.utils.data.Dataset,
    batch_size: int,
    patience: int,
    probe_model: torch.nn.Module,
    probe_loss: torch.nn.Module,
    model_under_probe: torch.nn.Module,
    train_epochs: int,
    output_path: str,
):
    """
    traines the probe model

    Args:
        embedding_func (callable): the functoin to generate the embeddings from the model under probing
        train_dataset (torch.utils.data.Dataset): train dataset
        valid_dataset (torch.utils.data.Dataset): test dataset
        batch_size (int): batch_size
        patience (int): number of epochs to wait before early stopping
        probe_model (torch.nn.Module): the probe itself
        probe_loss (torch.nn.Module): the loss function for the probe
        model_under_probe (torch.nn.Module): model to be probed
        train_epochs (int): number of training epochs
        output_path (str): where to save the probe model and probing results
    """
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collator_fn(batch),
        num_workers=0,
    )

    optimizer = torch.optim.Adam(probe_model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=0
    )
    criterion = probe_loss

    probe_model.train()
    model_under_probe.eval()

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

            embds = embedding_func(
                original_code_strings, "funcgnn", model_under_probe, max_len=29
            )

            d_pred, c_pred, u_pred = probe_model(embds.to(device))

            loss = criterion(
                d_pred=d_pred.to(device),
                c_pred=c_pred.to(device),
                u_pred=u_pred.to(device),
                d_real=torch.tensor(ds).to(device),
                c_real=torch.tensor(cs).to(device),
                u_real=torch.tensor(us).to(device),
                length_batch=batch_len_tokens.to(device),
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
        if patience_count == patience:
            logger.info("Stopping training loop (out of patience).")
            break


def eval_probe(
    embedding_func,
    collator_fn,
    eval_dataset,
    batch_size,
    probe_model,
    probe_loss,
    model_under_probe,
):
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

            ds = ds.to(device)
            cs = cs.to(device)
            us = us.to(device)

            embds = embedding_func(
                original_code_strings, "funcgnn", model_under_probe, max_len=29
            )

            d_pred, c_pred, u_pred = probe_model(embds.to(device))

            loss = probe_loss(
                d_pred=d_pred.to(device),
                c_pred=c_pred.to(device),
                u_pred=u_pred.to(device),
                d_real=torch.tensor(ds).to(device),
                c_real=torch.tensor(cs).to(device),
                u_real=torch.tensor(us).to(device),
                length_batch=batch_len_tokens.to(device),
            )

            eval_loss += loss.item()

            d_hits, c_hits, u_hits, d_hits_len, c_hits_len = probe_loss.calculate_hits(
                d_pred, c_pred, u_pred, ds, cs, us, batch_len_tokens
            )
            d_hits_total.append((d_hits, d_hits_len))
            c_hits_total.append((c_hits, c_hits_len))
            u_hits_total.append((u_hits, d_hits_len))

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
