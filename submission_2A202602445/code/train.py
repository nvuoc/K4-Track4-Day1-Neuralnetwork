"""train.py — Vòng lặp huấn luyện, đánh giá và thí nghiệm.

Gồm: đặt seed, đánh giá, vòng huấn luyện `run_experiment(cfg, data)`, dự đoán và ghi file nộp.
Mọi thí nghiệm chỉ là đổi dict cfg rồi gọi lại run_experiment.
Mọi chỉ số (loss, accuracy, macro-F1) dùng cùng định nghĩa với scripts/evaluate.py.
"""
from __future__ import annotations

import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from data import iterate_batches
from model import MLP, EXPECTED_PARAMS, count_params
from optimizer import build_optimizer, clip_gradients

# Cấu hình mặc định = BASELINE (M-base). lr được chọn trên val.
DEFAULT_CFG = dict(
    exp_id="base-s1", group="baseline", description="Baseline M-base",
    loss="ce",                 # "ce" | "mse"
    optimizer="sgd_momentum",  # "sgd" | "sgd_momentum" | "adam" | "adamw"
    lr=0.05,                   # chọn bằng val
    weight_decay=0.0, momentum=0.9,
    batch=512, epochs=20,
    hidden=(256, 128), dropout=0.0, init="he",
    clip_norm=None,            # None = không clip; hoặc số, ví dụ 1.0
    precision="fp32",          # "fp32" | "fp16" | "bf16"
    seed=1,
)


def set_seed(seed: int) -> None:
    """Đặt seed cho random, numpy, torch (và torch.cuda nếu có)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """macro-F1 = trung bình cộng F1 của 7 lớp; F1_c = 2PR/(P+R), bằng 0 nếu P+R = 0.

    cm: ma trận nhầm lẫn (7, 7), hàng = nhãn thật, cột = dự đoán.
    """
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    denom = prec + rec
    f1 = np.divide(2 * prec * rec, denom, out=np.zeros_like(tp), where=denom > 0)
    return float(np.mean(f1))


@torch.no_grad()
def predict(model: torch.nn.Module, X: torch.Tensor, batch_size: int = 8192) -> torch.Tensor:
    """Trả về nhãn dự đoán int64 (N,) = argmax của logits."""
    model.eval()
    preds = []
    n = len(X)
    for i in range(0, n, batch_size):
        xb = X[i:min(i + batch_size, n)]
        logits = model(xb)
        preds.append(logits.argmax(dim=-1))
    return torch.cat(preds, dim=0)


def compute_loss(logits: torch.Tensor, y: torch.Tensor, loss_name: str) -> torch.Tensor:
    """"ce"  : cross-entropy nhận logit thô và nhãn int64 (F.cross_entropy).
       "mse" : MSE giữa logit và one-hot của y (chia cho số lớp để tương đương trung bình trên mọi phần tử).
    """
    if loss_name == "ce":
        return F.cross_entropy(logits, y)
    elif loss_name == "mse":
        y_onehot = F.one_hot(y, num_classes=logits.size(-1)).float()
        return F.mse_loss(logits, y_onehot)
    else:
        raise ValueError(f"Hàm mất mát không hợp lệ: {loss_name}")


@torch.no_grad()
def evaluate(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor,
             loss_name: str = "ce", batch_size: int = 8192) -> dict:
    """Trả về dict(loss, acc, macro_f1) ở chế độ eval() (dropout tắt) và no_grad."""
    model.eval()
    n = len(X)
    total_loss = 0.0
    preds = []
    for i in range(0, n, batch_size):
        xb = X[i:min(i + batch_size, n)]
        yb = y[i:min(i + batch_size, n)]
        logits = model(xb)
        if loss_name == "ce":
            batch_loss = F.cross_entropy(logits, yb, reduction="sum")
        else:
            yb_onehot = F.one_hot(yb, num_classes=logits.size(-1)).float()
            # F.mse_loss sum over all elements, chia số lớp để có tổng loss theo mẫu
            batch_loss = F.mse_loss(logits, yb_onehot, reduction="sum") / logits.size(-1)
        total_loss += float(batch_loss.item())
        preds.append(logits.argmax(dim=-1))

    all_preds = torch.cat(preds, dim=0)
    acc = float((all_preds == y).float().mean().item())

    cm = np.zeros((7, 7), dtype=np.int64)
    np.add.at(cm, (y.cpu().numpy(), all_preds.cpu().numpy()), 1)
    macro_f1 = macro_f1_from_confusion(cm)

    return {
        "loss": float(total_loss / n),
        "acc": acc,
        "macro_f1": macro_f1,
    }


def run_experiment(cfg: dict, data: dict) -> dict:
    """Huấn luyện một cấu hình và trả về lịch sử + tóm tắt."""
    set_seed(cfg.get("seed", 1))

    device = data["X_tr"].device
    hidden = tuple(cfg.get("hidden", (256, 128)))
    dropout = float(cfg.get("dropout", 0.0))
    init = cfg.get("init", "he")

    model = MLP(hidden=hidden, dropout=dropout, init=init).to(device)
    assert count_params(model) == EXPECTED_PARAMS[hidden], (
        f"Số tham số không khớp: có {count_params(model)}, kỳ vọng {EXPECTED_PARAMS[hidden]}"
    )

    optimizer = build_optimizer(
        name=cfg.get("optimizer", "sgd_momentum"),
        params=model.parameters(),
        lr=cfg.get("lr", 0.05),
        weight_decay=cfg.get("weight_decay", 0.0),
        momentum=cfg.get("momentum", 0.9),
    )

    precision = cfg.get("precision", "fp32")
    use_scaler = (precision == "fp16" and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    if precision == "fp16":
        autocast_dtype = torch.float16
    elif precision == "bf16":
        autocast_dtype = torch.bfloat16
    else:
        autocast_dtype = torch.float32

    loss_name = cfg.get("loss", "ce")

    # Đo loss bước 0 TRƯỚC bước cập nhật đầu tiên
    step0_res = evaluate(model, data["X_val"], data["y_val"], loss_name=loss_name)
    step0_loss = float(step0_res["loss"])

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
        "grad_norm": [],
        "epoch_time_s": [],
    }

    best_val_loss = float("inf")
    best_epoch = -1
    best_state = None
    best_val_acc = 0.0
    best_val_macro_f1 = 0.0
    diverged = False

    epochs = cfg.get("epochs", 20)
    batch_size = cfg.get("batch", 512)
    clip_norm = cfg.get("clip_norm", None)

    # Tập con cố định của train để đánh giá train loss ở chế độ eval() nhanh chóng
    eval_train_size = min(50000, len(data["X_tr"]))
    eval_X_tr = data["X_tr"][:eval_train_size]
    eval_y_tr = data["y_tr"][:eval_train_size]

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_grad_norms = []

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        for xb, yb in iterate_batches(data["X_tr"], data["y_tr"], batch_size=batch_size, shuffle=True):
            optimizer.zero_grad(set_to_none=True)

            if precision in ("fp16", "bf16") and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    logits = model(xb)
                    loss = compute_loss(logits, yb, loss_name)
            else:
                logits = model(xb)
                loss = compute_loss(logits, yb, loss_name)

            if torch.isnan(loss) or torch.isinf(loss):
                diverged = True
                break

            if use_scaler:
                scaler.scale(loss).backward()
                if clip_norm is not None:
                    scaler.unscale_(optimizer)
                gn = clip_gradients(model.parameters(), clip_norm)
                epoch_grad_norms.append(gn)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                gn = clip_gradients(model.parameters(), clip_norm)
                epoch_grad_norms.append(gn)
                optimizer.step()

        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_time = time.perf_counter() - t0

        if diverged:
            print(f"[{cfg.get('exp_id', 'exp')}] Diverged tại epoch {epoch}!")
            break

        # Cuối epoch: đo ở eval mode
        val_res = evaluate(model, data["X_val"], data["y_val"], loss_name=loss_name)
        train_res = evaluate(model, eval_X_tr, eval_y_tr, loss_name=loss_name)
        avg_gn = float(np.mean(epoch_grad_norms)) if epoch_grad_norms else 0.0

        history["epoch"].append(epoch)
        history["train_loss"].append(train_res["loss"])
        history["val_loss"].append(val_res["loss"])
        history["val_acc"].append(val_res["acc"])
        history["val_macro_f1"].append(val_res["macro_f1"])
        history["grad_norm"].append(avg_gn)
        history["epoch_time_s"].append(epoch_time)

        if val_res["loss"] < best_val_loss:
            best_val_loss = val_res["loss"]
            best_epoch = epoch
            best_val_acc = val_res["acc"]
            best_val_macro_f1 = val_res["macro_f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    peak_mem_MB = 0.0
    if device.type == "cuda":
        peak_mem_MB = float(torch.cuda.max_memory_allocated(device) / (1024 * 1024))

    summary = {
        "step0_loss": step0_loss,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "final_train_loss": history["train_loss"][-1] if history["train_loss"] else float("nan"),
        "final_val_loss": history["val_loss"][-1] if history["val_loss"] else float("nan"),
        "val_acc": best_val_acc,
        "val_macro_f1": best_val_macro_f1,
        "time_per_epoch_s": float(np.mean(history["epoch_time_s"])) if history["epoch_time_s"] else 0.0,
        "peak_mem_MB": peak_mem_MB,
        "diverged": diverged,
    }

    return {
        "cfg": cfg,
        "history": history,
        "summary": summary,
        "best_state": best_state,
    }


def write_predictions(row_id: np.ndarray, preds: np.ndarray, path: str) -> None:
    """Ghi file nộp cho scripts/evaluate.py: CSV có tiêu đề `row_id,pred`."""
    r_id = np.asarray(row_id, dtype=np.int64)
    p = np.asarray(preds, dtype=np.int64)
    df = pd.DataFrame({"row_id": r_id, "pred": p})
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    df.to_csv(path, index=False)


def final_eval(cfg: dict, result: dict, data: dict, pred_path: str) -> None:
    """Dùng MỘT LẦN cho cấu hình cuối cùng (và baseline): nạp best_state, dự đoán eval, ghi predictions."""
    device = data["X_eval"].device
    hidden = tuple(cfg.get("hidden", (256, 128)))
    dropout = float(cfg.get("dropout", 0.0))
    init = cfg.get("init", "he")

    model = MLP(hidden=hidden, dropout=dropout, init=init).to(device)
    model.load_state_dict(result["best_state"])
    preds = predict(model, data["X_eval"])
    write_predictions(data["eval_row_id"], preds.cpu().numpy(), pred_path)
