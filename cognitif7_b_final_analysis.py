import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time

torch.manual_seed(42)
np.random.seed(42)

def generate_stressed_mg_events(T=2000, tau=17, noise_std=0.3):
    rng = np.random.default_rng(42)
    x = np.zeros(T + tau)
    x[:tau] = 1.2 + 0.1 * rng.standard_normal(tau)
    for t in range(tau, T + tau - 1):
        x[t+1] = x[t] + (0.2 * x[t-tau] / (1 + x[t-tau]**10) - 0.1 * x[t])
    mg = x[tau:]
    mg = (mg - np.mean(mg)) / np.std(mg)
    mg_noisy = mg + rng.normal(0, noise_std, size=T)
    events = np.zeros((T, 1))
    target = np.zeros(T)
    last_event_t = -100
    for t in range(T):
        if rng.random() < 0.02:
            events[t, 0] = 1.0
            last_event_t = t
        if t > 20:
            is_pos = mg[t] > 0
            window_size = 5 + int((t % 15))
            event_recent = (t - last_event_t) <= window_size
            target[t] = 1.0 if (is_pos and event_recent) else 0.0
    return mg_noisy[:, None], events, target

class MambaBlock(nn.Module):
    def __init__(self, d_in=64, d_state=64, d_conv=4):
        super().__init__()
        self.d_state, self.d_in, self.d_conv = d_state, d_in, d_conv
        self.conv1d = nn.Conv1d(d_in, d_in, kernel_size=d_conv, groups=d_in, padding=0)
        self.A_real = nn.Parameter(torch.log(torch.linspace(0.1, 1.0, d_state)))
        self.A_imag = nn.Parameter(torch.linspace(0.1, 5.0, d_state))
        self.W_delta = nn.Linear(d_in, d_state)
        self.B_proj = nn.Linear(d_in, d_state)
        self.W_gate = nn.Linear(d_in, d_in)
        self.C_proj = nn.Sequential(nn.Linear(d_state * 2 + d_in, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, d_in))

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        x_t = x.transpose(1, 2)
        x_pad = F.pad(x_t, (self.d_conv - 1, 0))
        x_conv = F.silu(self.conv1d(x_pad)).transpose(1, 2)
        h = torch.zeros(batch_size, self.d_state, dtype=torch.complex64, device=x.device)
        A = -torch.exp(self.A_real) + 1j * torch.exp(self.A_imag)
        outputs, all_deltas, hidden_states = [], [], []
        for t in range(seq_len):
            u_t = x_conv[:, t, :]
            delta_t = 0.05 + 0.35 * torch.sigmoid(self.W_delta(u_t))
            all_deltas.append(delta_t)
            A_bar = torch.exp(delta_t * A)
            B_bar = delta_t * self.B_proj(u_t)
            h = A_bar * h + B_bar.to(torch.complex64)
            hidden_states.append(h.detach())
            h_features = torch.cat([h.real, h.imag, u_t], dim=-1)
            y_ssm = self.C_proj(h_features)
            gate = F.silu(self.W_gate(x[:, t, :]))
            outputs.append(y_ssm * gate)
        return torch.stack(outputs, dim=1), torch.stack(all_deltas, dim=1), torch.stack(hidden_states, dim=1)

class ResidualMambaBlock(nn.Module):
    def __init__(self, d_model=64, d_state=64, dropout=0.15):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.block = MambaBlock(d_in=d_model, d_state=d_state)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out, deltas, hidden = self.block(self.norm(x))
        return x + self.dropout(out), deltas, hidden

class Cognitif7Deep(nn.Module):
    def __init__(self, d_in=2, d_model=64, d_state=64, n_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(d_in, d_model)
        self.layers = nn.ModuleList([ResidualMambaBlock(d_model=d_model, d_state=d_state) for _ in range(n_layers)])
        self.norm_out = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 2))
    def forward(self, x):
        h = self.input_proj(x)
        layer_deltas, layer_hidden = [], []
        for layer in self.layers:
            h, deltas, hidden = layer(h)
            layer_deltas.append(deltas)
            layer_hidden.append(hidden)
        return self.head(self.norm_out(h)), torch.stack(layer_deltas).mean(0), torch.stack(layer_hidden)

def compute_loss(logits, targets, deltas, criterion, lambda_var=2.0, lambda_smooth=0.5):
    cls_loss = criterion(logits.reshape(-1, 2), targets.reshape(-1))
    var_pen = torch.relu(0.02 - torch.var(deltas, dim=1).mean())
    smooth_pen = torch.mean((deltas[:, 1:, :] - deltas[:, :-1, :]) ** 2)
    return cls_loss + lambda_var * var_pen + lambda_smooth * smooth_pen, cls_loss.item()

if __name__ == "__main__":
    cont, events, target = generate_stressed_mg_events(2000, 17, 0.3)
    tr, va, te = slice(200, 1400), slice(1400, 1600), slice(1600, 2000)
    
    scaler = StandardScaler().fit(cont[tr])
    X_c = torch.FloatTensor(scaler.transform(cont)).unsqueeze(0)
    X_e = torch.FloatTensor(events).unsqueeze(0)
    Y = torch.LongTensor(target).unsqueeze(0)
    X_all = torch.cat([X_c, X_e], dim=-1)
    
    X_tr, Y_tr = X_all[:, tr, :], Y[:, tr]
    X_va, Y_va = X_all[:, va, :], Y[:, va]
    X_te, Y_te = X_all[:, te, :], Y[:, te]
    
    model = Cognitif7Deep()
    optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=5e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60)
    w1 = (Y_tr.squeeze().numpy() == 0).sum() / (Y_tr.squeeze().numpy() == 1).sum()
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([1.0, w1]))
    
    best_val, best_state = -1.0, None
    for epoch in range(60):
        model.train()
        optimizer.zero_grad()
        logits, deltas, _ = model(X_tr)
        loss, _ = compute_loss(logits, Y_tr, deltas, criterion)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                val_logits, _, _ = model(X_va)
                val_bacc = balanced_accuracy_score(Y_va.squeeze().numpy(), torch.argmax(val_logits, dim=-1).squeeze().numpy())
            if val_bacc > best_val:
                best_val = val_bacc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
    
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits_te, deltas_te, hidden_te = model(X_te)
        preds_te = torch.argmax(logits_te, dim=-1)
        test_bacc = balanced_accuracy_score(Y_te.squeeze().numpy(), preds_te.squeeze().numpy())
        print(f"Test BAcc: {test_bacc:.4f}")
        
    # --- LA LIGNE MANQUANTE EST ICI ---
    h_abs = hidden_te[-1, 0].abs().cpu().numpy()
    y_test_np = Y_te.squeeze().numpy()
    preds_np = preds_te.squeeze().numpy()
    
    np.savez_compressed(
        "analysis/cognitif7_b_hidden_states_test.npz",
        h_abs=h_abs, labels=y_test_np, predictions=preds_np
    )
    print("Fichier .npz sauvegardé !")
