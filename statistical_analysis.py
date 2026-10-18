import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import entropy, spearmanr, pearsonr, shapiro, normaltest
from sklearn.feature_selection import mutual_info_classif

print("Chargement des états cachés...")
try:
    data = np.load("analysis/cognitif7_b_hidden_states_test.npz")
    h_abs = data["h_abs"]
    y_test = data["labels"]
    preds = data["predictions"]
except FileNotFoundError:
    print("Erreur : le fichier .npz n'a pas été trouvé. Avez-vous bien exécuté le script d'analyse précédent ?")
    exit()

# 1. Calcul de l'Entropie Interne (H)
n_bins = 16
internal_entropy = []
for i in range(h_abs.shape[1]):
    counts, _ = np.histogram(h_abs[:, i], bins=n_bins)
    probs = counts / counts.sum()
    internal_entropy.append(entropy(probs[probs > 0]))
internal_entropy = np.array(internal_entropy)

# 2. Calcul de l'Importance Mamba (Information Mutuelle I(H;Y))
mi_scores = mutual_info_classif(h_abs, y_test, discrete_features=False, random_state=42)

# 3. Création du DataFrame
df = pd.DataFrame({
    "state": np.arange(h_abs.shape[1]),
    "entropy": internal_entropy,
    "mutual_info": mi_scores
})

print("\n=== TOP 5 DES ÉTATS PAR INFORMATION MUTUELLE ===")
print(df.sort_values("mutual_info", ascending=False).head(5).to_string(index=False))

# 4. Tests de Normalité (Shapiro-Wilk)
print("\n=== TESTS DE NORMALITÉ (Shapiro-Wilk) ===")
stat_x, p_x = shapiro(df["entropy"])
stat_y, p_y = shapiro(df["mutual_info"])
print(f"Entropie      : W={stat_x:.4f}, p={p_x:.4g}")
print(f"Info Mutuelle : W={stat_y:.4f}, p={p_y:.4g}")

# 5. Corrélations (Pearson vs Spearman)
r_pearson, p_pearson = pearsonr(df["entropy"], df["mutual_info"])
rho_spearman, p_spearman = spearmanr(df["entropy"], df["mutual_info"])

print("\n=== CORRÉLATIONS ===")
print(f"Pearson  (linéaire)  : r = {r_pearson:.4f} (p={p_pearson:.4g})")
print(f"Spearman (monotone)  : rho = {rho_spearman:.4f} (p={p_spearman:.4g})")

if p_y < 0.05 or p_x < 0.05:
    print("=> La normalité est rejetée. La corrélation de Spearman est la plus fiable.")
else:
    print("=> Normalité acceptable. Pearson et Spearman sont valides.")

# 6. Graphique (Scatter Plot)
plt.figure(figsize=(10, 6))
sc = plt.scatter(
    df["entropy"], 
    df["mutual_info"], 
    c=df["mutual_info"], 
    cmap="viridis", 
    s=50, 
    alpha=0.8,
    edgecolors="black"
)
plt.colorbar(sc, label="Information Mutuelle (nats)")
plt.axvline(df["entropy"].median(), color="gray", linestyle="--", alpha=0.5)
plt.axhline(df["mutual_info"].median(), color="gray", linestyle="--", alpha=0.5)

# Annoter les 5 meilleurs états
top_states = df.sort_values("mutual_info", ascending=False).head(5)
for _, row in top_states.iterrows():
    plt.annotate(f"S{int(row['state'])}", 
                 xy=(row["entropy"], row["mutual_info"]),
                 xytext=(5, 5), textcoords="offset points", fontsize=9, fontweight='bold')

plt.xlabel("Entropie Interne (Diversité de l'état)")
plt.ylabel("Information Mutuelle (Importance pour la classe)")
plt.title("Entropie vs Importance des États Cachés (Deep Mamba)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("analysis/cognitif7_b_entropy_vs_importance.png", dpi=180)
print("\nGraphique sauvegardé : analysis/cognitif7_b_entropy_vs_importance.png")
