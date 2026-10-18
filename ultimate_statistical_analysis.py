import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import entropy, spearmanr, pearsonr, shapiro, boxcox, yeojohnson, probplot
from sklearn.feature_selection import mutual_info_classif

print("1. Chargement des données depuis le fichier .npz...")
data = np.load("analysis/cognitif7_b_hidden_states_test.npz")
h_abs = data["h_abs"]
y_test = data["labels"]
preds = data["predictions"]

# ============================================================
# 2. CALCUL DES MÉTRIQUES (Entropie & Information Mutuelle)
# ============================================================
print("2. Calcul de l'entropie et de l'information mutuelle...")
n_bins = 16
x_raw = []
for i in range(h_abs.shape[1]):
    counts, _ = np.histogram(h_abs[:, i], bins=n_bins)
    probs = counts / counts.sum()
    x_raw.append(entropy(probs[probs > 0]))
x_raw = np.array(x_raw)

y_raw = mutual_info_classif(h_abs, y_test, discrete_features=False, random_state=42)

# ============================================================
# 3. TRANSFORMATIONS
# ============================================================
print("3. Application des transformations...")
x_log, y_log = np.log1p(x_raw), np.log1p(y_raw)

# Box-Cox (nécessite > 0)
x_bc, _ = boxcox(x_raw + 1e-6)
y_bc, _ = boxcox(y_raw + 1e-6)

# Yeo-Johnson (accepte 0)
x_yj, _ = yeojohnson(x_raw)
y_yj, _ = yeojohnson(y_raw)

transformations = {
    "Brut": (x_raw, y_raw),
    "log1p": (x_log, y_log),
    "Box-Cox": (x_bc, y_bc),
    "Yeo-Johnson": (x_yj, y_yj),
}

# ============================================================
# 4. TESTS DE NORMALITÉ ET CORRÉLATIONS
# ============================================================
print("4. Tests statistiques (Shapiro, Pearson, Spearman)...")
stats_rows = []

for name, (xx, yy) in transformations.items():
    p_shapiro_x = shapiro(xx).pvalue
    p_shapiro_y = shapiro(yy).pvalue
    r, p_pearson = pearsonr(xx, yy)
    rho, p_spearman = spearmanr(xx, yy)
    
    stats_rows.append({
        "Transformation": name,
        "Shapiro_p_Entropie": p_shapiro_x,
        "Shapiro_p_Importance": p_shapiro_y,
        "Pearson_r": r,
        "Pearson_p": p_pearson,
        "Spearman_rho": rho,
        "Spearman_p": p_spearman
    })

df_stats = pd.DataFrame(stats_rows)
print("\n=== TABLEAU DES CORRÉLATIONS ===")
print(df_stats.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
df_stats.to_csv("analysis/final_stats_correlations.csv", index=False)

# ============================================================
# 5. GRAPHIQUES (QQ-Plots & Bar Chart)
# ============================================================
print("5. Génération des graphiques...")

# --- Figure 1 : QQ-Plots pour l'Entropie ---
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, (name, (xx, _)) in zip(axes.flatten(), transformations.items()):
    probplot(xx, dist="norm", plot=ax)
    ax.set_title(f"Entropie — {name}")
    ax.grid(True, alpha=0.3)
plt.suptitle("Effet des transformations sur la normalité de l'entropie", y=1.02)
plt.tight_layout()
plt.savefig("analysis/qq_entropy.png", dpi=150)
plt.close()

# --- Figure 2 : Bar Chart (Pearson vs Spearman) ---
labels = df_stats["Transformation"]
pearson_vals = df_stats["Pearson_r"]
spearman_vals = df_stats["Spearman_rho"]

x_pos = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x_pos - width/2, pearson_vals, width, label='Pearson r', color='tab:blue')
bars2 = ax.bar(x_pos + width/2, spearman_vals, width, label='Spearman rho', color='tab:orange')

ax.axhline(0, color='black', linewidth=0.8)
ax.bar_label(bars1, fmt="%.2f", padding=3, fontsize=9)
ax.bar_label(bars2, fmt="%.2f", padding=3, fontsize=9)
ax.set_xticks(x_pos)
ax.set_xticklabels(labels)
ax.set_ylabel("Coefficient de corrélation")
ax.set_title("Relation Entropie ↔ Importance selon la transformation")
ax.legend()
ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("analysis/correlation_bar_chart.png", dpi=150)
plt.close()

# --- Figure 3 : Scatter Plot (Yeo-Johnson) ---
plt.figure(figsize=(8, 6))
sc = plt.scatter(x_yj, y_yj, c=y_yj, cmap="viridis", s=60, edgecolors="black", alpha=0.8)
plt.colorbar(sc, label="Importance Mutuelle")
plt.xlabel("Entropie (Yeo-Johnson)")
plt.ylabel("Importance (Yeo-Johnson)")
plt.title("Nuage de points (Transformé Yeo-Johnson)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("analysis/scatter_final.png", dpi=150)
plt.close()

# ============================================================
# 6. RAPPORT FINAL MARKDOWN
# ============================================================
print("6. Génération du rapport Markdown...")
with open("analysis/RAPPORT_FINAL_COGNITIF7B.md", "w", encoding="utf-8") as f:
    f.write("# Rapport Final d'Analyse Statistique — Cognitif7-B\n\n")
    
    f.write("## 1. Performance du Modèle\n")
    f.write("- **Deep Mamba (2 couches) :** Test BAcc = 0.6021\n")
    f.write("- **Baseline (1 couche) :** Test BAcc = 0.7135\n")
    f.write("- **Verdict :** La baseline reste supérieure. Le Deep Mamba a sur-appris.\n\n")
    
    f.write("## 2. Analyse des Corrélations\n")
    f.write("Voici l'impact des transformations mathématiques sur la relation entre l'entropie interne et l'importance de l'état :\n\n")
    f.write(df_stats.to_markdown(index=False, floatfmt=".4f"))
    f.write("\n\n")
    
    f.write("## 3. Interprétation\n")
    f.write("- Si **Pearson** augmente fortement après transformation, la relation était non-linéaire.\n")
    f.write("- Si **Spearman** reste stable et élevé, la relation est monotone (plus l'entropie est grande, plus l'importance est grande, ou inversement).\n")
    f.write("- Si la normalité (Shapiro) est rejetée (p < 0.05), la corrélation de Spearman est la métrique la plus fiable à rapporter.\n\n")
    
    f.write("## 4. Fichiers Générés\n")
    f.write("- `qq_entropy.png` (Graphique de normalité)\n")
    f.write("- `correlation_bar_chart.png` (Comparaison Pearson/Spearman)\n")
    f.write("- `scatter_final.png` (Nuage de points final)\n")

print("\n✅ Analyse terminée ! Regarde le fichier 'analysis/RAPPORT_FINAL_COGNITIF7B.md' pour le résumé.")
