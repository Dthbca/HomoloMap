"""Generate deterministic visual previews shown on GitHub tutorial pages."""

from pathlib import Path
import os
import sys

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
sys.path.insert(0, str(ROOT / "src"))
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


# 1: surface previews; the two inputs occupy the same BN regional space.
if os.name == "nt":
    os.environ.setdefault("VTK_DEFAULT_OPENGL_WINDOW", "vtkWin32OpenGLRenderWindow")
from HomoloMap.plotting import plot_left

surface_specs = [
    (Y.iloc[:, 0], "Example brain IDP", "coolwarm", "IDP value",
     "01_brain_idp_surface.png"),
    (X["Sst"], "Example Sst cell map", "YlOrRd", "Mapped proportion",
     "01_cell_map_surface.png"),
]
for values, title, cmap, label, filename in surface_specs:
    fig = plot_left(
        values, atlas="BN", surf="inflated", view="row", cmap=cmap,
        cbar_label=label, title=title, title_fontsize=11,
        figsize=(7.2, 2.5), render_scale=(2, 2),
    )
    save(fig, filename)


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

print(f"PASS figures=5 output={OUT}")
