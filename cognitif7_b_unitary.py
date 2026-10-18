import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
import time

torch.manual_seed(42)
np.random.seed(42)

# ============================================================
# 1. GÉNÉRATEUR HOSTILE
# ============================================================
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

# ============================================================
# 2. MAMBA BLOCK (Capacité Réduite)
# ============================================================
class MambaBlock(nn.Module):
    def __init__(self, d_in=2, d_state=64, d_conv=4):
        super().__init__()
        self.d_state = d_state
        self.d_in = d_in
        self.d_conv = d_conv
        
        self.conv1d = nn.Conv1d(d_in, d_in, kernel_size=d_conv, groups=d_in, padding=0)
        
        self.A_real = nn.Parameter(torch.log(torch.linspace(0.1, 1.0, d_state)))
        self.A_imag = nn.Parameter(torch.linspace(0.1, 5.0, d_state))
        
        self.W_delta = nn.Linear(d_in, d_state)
        self.B_proj = nn.Linear(d_in, d_state)
        self.W_gate = nn.Linear(d_in, 2)
        
        # Dropout augmenté à 0.2
        self.C_proj = nn.Sequential(
            nn.Linear(d_state * 2 + d_in, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        x_t = x.transpose(1, 2)
        x_pad = F.pad(x_t, (self.d_conv - 1, 0))
        x_conv = F.silu(self.conv1d(x_pad)).transpose(1, 2)
        
        h = torch.zeros(batch_size, self.d_state, dtype=torch.complex64, device=x.device)
        A = -torch.exp(self.A_real) + 1j * torch.exp(self.A_imag)
        
        outputs = []
        all_deltas = []
        
        for t in range(seq_len):
            u_t = x_conv[:, t, :]
            delta_t = 0.05 + 0.35 * torch.sigmoid(self.W_delta(u_t))
            all_deltas.append(delta_t)
            
            A_bar = torch.exp(delta_t * A)
            B_bar = delta_t * self.B_proj(u_t)
            h = A_bar * h + B_bar.to(torch.complex64)
            
            h_features = torch.cat([h.real, h.imag, u_t], dim=-1)
            y_ssm = self.C_proj(h_features)
            
            gate = F.silu(self.W_gate(x[:, t, :]))
            y_t = y_ssm * gate
            outputs.append(y_t)
            
        outputs = torch.stack(outputs, dim=1)
        deltas = torch.stack(all_deltas, dim=1)
        return outputs, deltas

# ============================================================
# 3. PERTE LINÉAIRE
# ============================================================
def compute_loss(logits, targets, deltas, criterion, lambda_var=2.0):
    cls_loss = criterion(logits.view(-1, 2), targets.view(-1))
    delta_time_var = torch.var(deltas, dim=1).mean()
    target_var = 0.02
    
    deficit = torch.relu(target_var - delta_time_var)
    variance_penalty = deficit
    total = cls_loss + lambda_var * variance_penalty
    return total, cls_loss.item(), delta_time_var.item()

# ============================================================
# 4. PROTOCOLE (WALK-FORWARD VALIDATION)
# ============================================================
if __name__ == "__main__":
    print("Génération des données (Bruit 0.3)...")
    cont, events, target = generate_stressed_mg_events(T=2000, noise_std=0.3)
    
    scaler_c = StandardScaler().fit(cont[:1200]) # Fit sur la première moitié
    X_c = torch.FloatTensor(scaler_c.transform(cont)).unsqueeze(0)
    X_e = torch.FloatTensor(events).unsqueeze(0)
    Y = torch.LongTensor(target).unsqueeze(0)
    X_all = torch.cat([X_c, X_e], dim=-1) 

    # 3 Folds de Validation Glissante (Walk-Forward)
    splits = [
        (slice(200, 900), slice(900, 1100), "Fold 1"),
        (slice(200, 1100), slice(1100, 1300), "Fold 2"),
        (slice(200, 1300), slice(1300, 1500), "Fold 3"),
    ]

    all_folds_train = []
    all_folds_val = []

    for tr_slice, va_slice, name in splits:
        print(f"\n--- {name} : Train {tr_slice.start}-{tr_slice.stop} | Val {tr_slice.stop}-{va_slice.stop} ---")
        
        X_tr, Y_tr = X_all[:, tr_slice, :], Y[:, tr_slice]
        X_va, Y_va = X_all[:, va_slice, :], Y[:, va_slice]
        
        # Distribution des classes
        y_tr_np = Y_tr.squeeze().numpy()
        y_va_np = Y_va.squeeze().numpy()
        print(f"    Ratio Train: {y_tr_np.mean():.3f} | Ratio Val: {y_va_np.mean():.3f}")
        
        w1 = (y_tr_np == 0).sum() / (y_tr_np == 1).sum()
        class_weights = torch.FloatTensor([1.0, w1])
        
        model = MambaBlock(d_in=2, d_state=64)
        # AdamW avec weight_decay
        optimizer = optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60, eta_min=1e-5)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        start = time.time()
        for epoch in range(60):
            model.train()
            optimizer.zero_grad()
            logits, deltas = model(X_tr)
            loss, _, _ = compute_loss(logits, Y_tr, deltas, criterion, lambda_var=2.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 
            optimizer.step()
            scheduler.step()
            
        print(f"    Temps: {time.time()-start:.1f}s")
        
        model.eval()
        with torch.no_grad():
            logits_tr, _ = model(X_tr)
            preds_tr = torch.argmax(logits_tr, dim=-1).squeeze().numpy()
            bacc_tr = balanced_accuracy_score(y_tr_np, preds_tr)
            
            logits_va, _ = model(X_va)
            preds_va = torch.argmax(logits_va, dim=-1).squeeze().numpy()
            bacc_va = balanced_accuracy_score(y_va_np, preds_va)
            
        print(f"    BAcc Train: {bacc_tr:.4f} | BAcc Val: {bacc_va:.4f}")
        all_folds_train.append(bacc_tr)
        all_folds_val.append(bacc_va)

    print("\n=== RÉSULTAT FINAL (Moyenne des Folds) ===")
    print(f"Train BAcc: {np.mean(all_folds_train):.4f} (± {np.std(all_folds_train):.4f})")
    print(f"Val BAcc  : {np.mean(all_folds_val):.4f} (± {np.std(all_folds_val):.4f})")
