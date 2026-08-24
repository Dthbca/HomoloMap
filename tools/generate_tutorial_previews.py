"""Generate deterministic visual previews shown on GitHub tutorial pages."""

from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).parents[1]
OUT = ROOT / "tutorials" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
X = pd.read_csv(ROOT / "data/maps/BN/ctype_ratio_BN_23_subclass.csv", index_col=0)
X.index = X.index.astype(int)
rng = np.random.default_rng(42)
position = np.linspace(-1, 1, len(X))
signal = (
    0.8 * (X["Chandelier"] - X["Chandelier"].mean()) / X["Chandelier"].std()
    + 0.5 * (X["Sst"] - X["Sst"].mean()) / X["Sst"].std()
)
Y = pd.DataFrame({
    "example IDP 1": signal + 0.2 * position + rng.normal(0, 0.30, len(X)),
    "example IDP 2": np.sin(np.pi * position) + rng.normal(0, 0.35, len(X)),
}, index=X.index)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 8,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def save(fig, name):
    fig.savefig(OUT / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# 1: alignment and composition QC
fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.4), gridspec_kw={"height_ratios": [3, 1]})
image = axes[0].imshow(X.T, aspect="auto", cmap="viridis", interpolation="none")
axes[0].set(ylabel="Cell-type subclass", title="Aligned BN cell-type composition")
axes[0].set_xticks([])
fig.colorbar(image, ax=axes[0], label="Mapped proportion", fraction=0.025, pad=0.02)
for column, color in zip(Y, ["#2f6690", "#d17b49"]):
    axes[1].plot(Y.index, Y[column], label=column, color=color, linewidth=1.2)
axes[1].set(xlabel="BN region label", ylabel="IDP value")
axes[1].legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.35))
fig.tight_layout()
save(fig, "01_prepare_brain_idps.png")


# 2: ranked association overview (the notebook adds spin-derived p-values)
r = X.apply(lambda values: pearsonr(values, Y.iloc[:, 0]).statistic).sort_values()
fig, ax = plt.subplots(figsize=(6.6, 4.5))
colors = np.where(np.abs(r) >= np.quantile(np.abs(r), 0.8), "#c95d63", "#a8bdc8")
ax.barh(r.index, r, color=colors, edgecolor="none")
ax.axvline(0, color="0.25", linewidth=0.8)
ax.set(xlabel="Pearson correlation", ylabel="Cell-type subclass",
       title="Example spatial association ranking")
fig.tight_layout()
save(fig, "02_spin_test.png")


# 3: held-out total performance and interpretable linear contributions
model = make_pipeline(StandardScaler(), Ridge(alpha=3.0))
folds = KFold(5, shuffle=True, random_state=42)
predicted = cross_val_predict(model, X, Y.iloc[:, 0], cv=folds)
r2 = 1 - np.sum((Y.iloc[:, 0] - predicted) ** 2) / np.sum((Y.iloc[:, 0] - Y.iloc[:, 0].mean()) ** 2)
model.fit(X, Y.iloc[:, 0])
importance = pd.Series(
    np.abs(model.named_steps['ridge'].coef_), index=X.columns
).sort_values().tail(12)
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8), gridspec_kw={"width_ratios": [1, 2.6]})
axes[0].bar(["Example IDP"], [r2], color="#477998", width=0.55)
axes[0].axhline(0, color="0.3", linewidth=0.8)
axes[0].set(ylabel="Out-of-fold R²", title="Total")
axes[1].barh(importance.index, importance, color="#edae49")
axes[1].set(xlabel="Absolute standardized coefficient", title="Individual contributions")
fig.tight_layout()
save(fig, "03_total_contribution_and_shap.png")


# 4: ratio versus CLR sensitivity
positive = X.replace(0, np.nan)
minimum = np.nanmin(positive.to_numpy()) / 2
closed = X.replace(0, minimum)
clr = np.log(closed).sub(np.log(closed).mean(axis=1), axis=0)
ratio_r = X.apply(lambda values: pearsonr(values, Y.iloc[:, 0]).statistic)
clr_r = clr.apply(lambda values: pearsonr(values, Y.iloc[:, 0]).statistic)
fig, ax = plt.subplots(figsize=(4.7, 4.2))
ax.scatter(ratio_r, clr_r, s=30, color="#4c956c", alpha=0.85, edgecolor="white", linewidth=0.4)
limits = [min(ratio_r.min(), clr_r.min()), max(ratio_r.max(), clr_r.max())]
ax.plot(limits, limits, "--", color="0.35", linewidth=0.8)
ax.set(xlabel="Ratio-map correlation", ylabel="CLR correlation",
       title="Example compositional sensitivity", aspect="equal")
fig.tight_layout()
save(fig, "04_clr_sensitivity.png")

print(f"PASS figures=4 output={OUT}")
