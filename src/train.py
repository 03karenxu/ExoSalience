import os, json, math, time
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader

from cnn import CNN1DEncoder, AstronetMVP
from dataset import ExoplanetDataset, collate

    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
torch.manual_seed(42); np.random.seed(42)
print("Device:", device)

def train_epoch(model, dloader, optim, criterion):
    model.train()
    total, n = 0.0, 0

    # feed train data
    for xg, xl, xt, y, _ids in dloader:

        # move to gpu
        xg = xg.to(device)
        xl = xl.to(device)
        y = y.to(device)# expected

        # feed
        xt = None if xt is None else xt.to(device)
        logits = model(xg, xl, xt).squeeze(1) # actual
        loss = criterion(logits, y)

        #compute optim
        optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), max_norm=5.0)
        optim.step()

        bs = y.size(0)
        total += loss.item() * bs
        n+= bs

    return total / max(n,1)


@torch.no_grad()
def evaluate(model, dloader, criterion, threshold: float = 0.5):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    tp = fp = tn = fn = 0

    for xg, xl, xt, y, _ids in dloader:
        xg = xg.to(device)
        xl = xl.to(device)
        y = y.to(device)
        xt = None if xt is None else xt.to(device)

        logits = model(xg, xl, xt).squeeze(1)
        loss = criterion(logits, y)
        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).float()

        bs = y.size(0)
        total_loss += loss.item() * bs
        total_correct += (preds == y).sum().item()
        total += bs

        tp += ((preds == 1) & (y == 1)).sum().item()
        fp += ((preds == 1) & (y == 0)).sum().item()
        tn += ((preds == 0) & (y == 0)).sum().item()
        fn += ((preds == 0) & (y == 1)).sum().item()

    avg_loss = total_loss / max(total, 1)
    accuracy = total_correct / max(total, 1)
    precision = tp / max(tp + fp, 1e-8)
    recall = tp / max(tp + fn, 1e-8)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


@torch.no_grad()
def predict_on_loader(model, dloader):
    model.eval()
    out = []  # list of dicts: {id, logit, prob}
    for xg, xl, xt, y, ids in dloader:
        xg = xg.to(device); xl = xl.to(device)
        xt = None if xt is None else xt.to(device).float()
        logits = model(xg, xl, xt).squeeze(1)          # [B]
        probs  = torch.sigmoid(logits)
        for _id, lo, pr in zip(ids, logits.cpu().tolist(), probs.cpu().tolist()):
            out.append({"id": _id, "logit": float(lo), "prob": float(pr)})
    return out

def main():
    # DATA
    # load npz for train / test
    import numpy as np

    npz_train = np.load("npzs/train.npz")
    npz_test = np.load("npzs/test.npz")

    # create train / test set from numpy
    global_train = torch.from_numpy(npz_train['global_view']).float()
    local_train = torch.from_numpy(npz_train['local_view']).float()
    labels_train = torch.from_numpy(npz_train['label']).float()
    ids_train = torch.from_numpy(npz_train['kepid']).float()
    tab_train = None #torch.from_numpy(npz_train['tabular'])

    global_test = torch.from_numpy(npz_test['global_view']).float()
    local_test = torch.from_numpy(npz_test['local_view']).float()
    labels_test = torch.from_numpy(npz_test['label']).float()
    ids_test = torch.from_numpy(npz_test['kepid']).float()
    tab_test = None #torch.from_numpy(npz_test['tabular'])

    ds_train = ExoplanetDataset(global_train, local_train, labels_train, ids_train, tabular=tab_train)
    ds_test  = ExoplanetDataset(global_test,  local_test,  labels_test,  ids_test,  tabular=tab_test)

    # loss, optimizer, pos weight from train set
    def compute_pos_weight_from_loader(dloader):
        pos = neg = 0
        with torch.no_grad():
            for _xg, _xl, _xt, _y, _ids in dloader:
                pos += (_y > 0.5).sum().item()
                neg += (_y <= 0.5).sum().item()
        if pos == 0 or neg == 0: return None
        return torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)

    # create data loaders from train / test data sets
    dl_train = DataLoader(ds_train, batch_size = 64, shuffle=True, collate_fn=collate, drop_last=False)
    dl_test = DataLoader(ds_test, batch_size = 64, shuffle=True, collate_fn=collate, drop_last=False)

    # Infer tabular_dim from one train batch
    xg0, xl0, xt0, y0, ids0 = next(iter(dl_train))
    tabular_dim = 0 if xt0 is None else xt0.shape[1]

    model = AstronetMVP(hidden=64, k_global=7, k_local=5, tabular_dim=tabular_dim).to(device)

    pos_weight = compute_pos_weight_from_loader(dl_train)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) if pos_weight is not None else nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)


    # make & save config / checkpoints dir
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    # make & save config
    config = {
        "hidden":64,
        "k_global": 7,
        "k_local": 5,
        "tabular_dim": tabular_dim,
        "lr": 1e-3,
        "wd": 1e-5,
        "pos_weight": None if pos_weight is None else float(pos_weight.item())
    }
    json.dump(config, open(os.path.join(save_dir, "config.json"), "w"), indent = 2)

    # train loop: train + save checkpoints
    num_epochs = 1000
    for epoch in range(1, num_epochs+1):
        t0 = time.time()
        tr_loss = train_epoch(model, dl_train, optimizer, criterion)
        dt = time.time() - t0

        print(f"Epoch {epoch:02d}/{num_epochs} | train_loss={tr_loss:.4f} | {dt:.1f}s")

    test_metrics = evaluate(model, dl_test, criterion)
    print(
        "Test metrics | "
        f"loss={test_metrics['loss']:.4f} "
        f"acc={test_metrics['accuracy']:.4f} "
        f"prec={test_metrics['precision']:.4f} "
        f"recall={test_metrics['recall']:.4f} "
        f"f1={test_metrics['f1']:.4f}"
    )

    # Save final checkpoint
    ckpt_path = os.path.join(save_dir, "model_final.pt")
    torch.save({"model_state": model.state_dict(), "config": config}, ckpt_path)
    print("Saved:", ckpt_path)

    # create test set predictions for eval
    test_preds = predict_on_loader(model, dl_test)
    print("Test preds:", len(test_preds))

    # EXPORT THE MODEL
    import copy
    model.eval()
    model_cpu = copy.deepcopy(model).cpu()
    scripted = torch.jit.script(model_cpu)
    scripted.save("model_scripted.pt")

    
if __name__ == "__main__":
    main()
