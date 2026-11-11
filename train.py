"""Training / evaluation loop for TG-Transformer (one dataset, one seed).

Test labels are isolated: the training loop and model selection only ever see a
label array in which every test entry is replaced by -1. The real test labels
are passed to ``evaluate`` exactly once, after the best-validation checkpoint
has been restored.
"""

from __future__ import annotations

import copy
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

from model import TGTransformer
from sampling import build_slices


def set_seed(seed: int, deterministic: bool = True):
    """Seed Python, numpy and torch, and enable deterministic algorithms."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_val_split(train_idx: np.ndarray, seed: int, frac: float = 0.1):
    """Randomly hold out ``frac`` of the training documents as validation.

    Uses its own ``RandomState(seed)``, so the split depends only on the seed and
    is identical across all model configurations for the same seed.

    Returns:
        (train_sub_idx, val_idx), both sorted int64 arrays.
    """
    rs = np.random.RandomState(seed)
    perm = rs.permutation(len(train_idx))
    n_val = int(round(frac * len(train_idx)))
    return np.sort(train_idx[perm[n_val:]]), np.sort(train_idx[perm[:n_val]])


def build_optimizer(model: TGTransformer, lr: float, weight_decay: float,
                    embedding_weight_decay: float):
    """Adam with weight decay on all parameters except the node-embedding table X.

    The embedding table gets ``embedding_weight_decay`` (0 in the baseline): with
    L2 decay, rows that never receive a gradient (test/validation documents) would
    otherwise be shrunk towards zero by Adam's normalised updates.
    The N1 attention-bias scales ``attn_beta`` get no weight decay, so they are not
    pulled towards 0 (= towards the baseline) artificially.
    """
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    emb = [p for n, p in named if n.startswith("node_emb.")]
    beta = [p for n, p in named if n == "attn_beta"]
    rest = [p for n, p in named if not n.startswith("node_emb.") and n != "attn_beta"]
    groups = [{"params": rest, "weight_decay": weight_decay}]
    if emb:
        groups.append({"params": emb, "weight_decay": embedding_weight_decay})
    if beta:
        groups.append({"params": beta, "weight_decay": 0.0})
    return torch.optim.Adam(groups, lr=lr)


@torch.no_grad()
def evaluate(model, node_ids, mask, labels, idx, batch_size: int, bias=None):
    """Mean cross-entropy loss, accuracy, macro-F1 and predictions on documents ``idx``."""
    model.eval()
    losses, preds = [], []
    for s in range(0, len(idx), batch_size):
        b = idx[s:s + batch_size]
        logits = model(node_ids[b], mask[b], None if bias is None else bias[b])
        losses.append(F.cross_entropy(logits, labels[b], reduction="sum").item())
        preds.append(logits.argmax(1))
    preds = torch.cat(preds).numpy()
    y = labels[idx].numpy()
    return {"loss": sum(losses) / len(idx), "acc": accuracy_score(y, preds),
            "macro_f1": f1_score(y, preds, average="macro"), "preds": preds}


def run_training(data, mcfg: dict, tcfg: dict, seed: int, log_every: int = 1,
                 attn_bias=None, return_model: bool = False):
    """Train one model on one dataset with one seed and evaluate on the test set.

    Args:
        data: ``data.PreparedData``.
        mcfg: model/sampler config (k, sampler, use_type_enc, use_wl_enc, readout,
            freeze_embeddings, hidden_size, ...).
        tcfg: training config (lr, weight_decay, batch_size, max_epochs, patience, ...).
        seed: random seed.
        attn_bias: for N1, a dict from ``attn_bias.get_attn_bias`` with ``B``
            (n_docs, k+1, k+1) and the ``node_ids`` it was built for; None otherwise.
        return_model: also return the model with the best checkpoint loaded.

    Returns:
        result: dict with test accuracy / macro-F1, best epoch, timings.
        curves: list of per-epoch dicts (train_loss, val_loss, val_acc, time).
    """
    set_seed(seed, tcfg.get("deterministic", True))
    train_idx = np.where(~data.is_test)[0]
    test_idx = np.where(data.is_test)[0]
    tr_idx, va_idx = make_val_split(train_idx, seed, tcfg["val_frac"])

    # Labels visible to training / model selection: test entries hidden.
    y_visible = data.labels.copy()
    y_visible[data.is_test] = -1
    assert (y_visible[tr_idx] >= 0).all() and (y_visible[va_idx] >= 0).all()
    assert not np.intersect1d(np.concatenate([tr_idx, va_idx]), test_idx).size
    y_visible = torch.as_tensor(y_visible)

    ids_np, mask_np, _ = build_slices(data, mcfg["sampler"], mcfg["k"], seed)
    node_ids, mask = torch.as_tensor(ids_np), torch.as_tensor(mask_np)
    bias_mode = mcfg.get("attn_bias", "none")
    if (bias_mode != "none") != (attn_bias is not None):
        raise ValueError("attn_bias data must be given exactly when mcfg['attn_bias'] != 'none'")
    bias = None
    if attn_bias is not None:
        if not np.array_equal(attn_bias["node_ids"], ids_np):
            raise ValueError("attention bias B was built for different slices")
        bias = torch.as_tensor(attn_bias["B"])

    model = TGTransformer(
        data.X, data.node_types, data.wl_codes, data.num_classes,
        hidden_size=mcfg["hidden_size"], num_layers=mcfg["num_layers"],
        num_heads=mcfg["num_heads"], intermediate_size=mcfg["intermediate_size"],
        dropout=mcfg["dropout"], attn_dropout=mcfg["attn_dropout"],
        use_type_enc=mcfg["use_type_enc"], use_wl_enc=mcfg["use_wl_enc"],
        readout=mcfg["readout"], freeze_embeddings=mcfg["freeze_embeddings"],
        graph_residual=mcfg["graph_residual"], attn_bias=bias_mode)
    opt = build_optimizer(model, tcfg["lr"], tcfg["weight_decay"],
                          tcfg["embedding_weight_decay"])
    gen = torch.Generator().manual_seed(seed)
    bs = tcfg["batch_size"]
    tr_t = torch.as_tensor(tr_idx)

    best = {"val_loss": float("inf"), "epoch": 0, "state": None}
    curves, epoch_times = [], []
    bad_epochs = 0
    t_start = time.perf_counter()
    for epoch in range(1, tcfg["max_epochs"] + 1):
        t0 = time.perf_counter()
        model.train()
        perm = tr_t[torch.randperm(len(tr_t), generator=gen)]
        tot, n = 0.0, 0
        for s in range(0, len(perm), bs):
            b = perm[s:s + bs]
            logits = model(node_ids[b], mask[b], None if bias is None else bias[b])
            loss = F.cross_entropy(logits, y_visible[b])        # paper Eq. 9
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
            n += len(b)
        val = evaluate(model, node_ids, mask, y_visible, va_idx, bs, bias)
        dt = time.perf_counter() - t0
        epoch_times.append(dt)
        curves.append({"epoch": epoch, "train_loss": tot / n, "val_loss": val["loss"],
                       "val_acc": val["acc"], "epoch_time_s": dt})
        improved = val["loss"] < best["val_loss"]
        if improved:
            best = {"val_loss": val["loss"], "val_acc": val["acc"], "epoch": epoch,
                    "state": copy.deepcopy(model.state_dict())}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if log_every and epoch % log_every == 0:
            print(f"    epoch {epoch:3d}  train_loss {tot / n:.4f}  val_loss {val['loss']:.4f}  "
                  f"val_acc {val['acc']:.4f}  {dt:.2f}s{'  *' if improved else ''}", flush=True)
        if bad_epochs >= tcfg["patience"]:
            break
    total_time = time.perf_counter() - t_start

    # ---- final test evaluation with the best-validation checkpoint ----------
    model.load_state_dict(best["state"])
    test = evaluate(model, node_ids, mask, torch.as_tensor(data.labels), test_idx, bs, bias)
    result = {
        "dataset": data.name, "seed": seed,
        "test_acc": test["acc"], "test_macro_f1": test["macro_f1"],
        "best_epoch": best["epoch"], "epochs_run": len(curves),
        "best_val_loss": best["val_loss"], "best_val_acc": best.get("val_acc"),
        "mean_epoch_time_s": float(np.mean(epoch_times)), "total_train_time_s": total_time,
        "n_train": int(len(tr_idx)), "n_val": int(len(va_idx)), "n_test": int(len(test_idx)),
        "n_padded_positions": int((~mask_np).sum()),
        "n_trainable_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
    }
    if bias is not None:
        result["attn_beta"] = best["state"]["attn_beta"].tolist()   # (num_layers, num_heads)
    if return_model:
        return result, curves, model
    return result, curves
