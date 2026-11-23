"""Figures du README et du rapport (results/figures/*.png).

Libellés en anglais (README et rapport en anglais). Palette catégorielle de référence, ordre fixe :
bleu = modèle principal (B2 quotidien), orange = Chronos-2, vert d'eau = LEAR ; prix réalisé en noir.

Usage : python -m src.figures   (après src.backtest --period test et src.importance)
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.backtest import NAIVE, load_predictions, regime
from src.config import PERIODS, RESULTS, TZ
from src.data import load_dataset
from src.features import add_features
from src import importance

FIG = RESULTS / "figures"

# Palette (instance de référence, mode clair)
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#b9b8b2"
MODEL_COLORS = {"B2_quotidien": BLUE, "Chronos2_fond": ORANGE, "LEAR_quotidien": AQUA,
                "B3_quotidien_hl90_corr": YELLOW}

LABELS = {
    "naive_d7": "Naive D-7", "naive_d1": "Naive D-1", "A1": "A1 calendar", "A2": "A2 + price lags",
    "B1": "B1 + FR fundamentals", "B2": "B2 + gas, CO2, DE", "B2_quotidien": "B2, daily retraining",
    "B2_hebdo": "B2, weekly retraining", "MSTL": "MSTL", "LEAR_quotidien": "LEAR (daily)",
    "Chronos2_prix": "Chronos-2, prices only", "Chronos2_fond": "Chronos-2 + fundamentals",
    "B2_quotidien_corr": "B2 daily + error correction*",
    "B3_quotidien_hl90_corr": "B3 daily + recency + correction*",
}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False, "figure.dpi": 150, "axes.axisbelow": True,
})


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}")


# --------------------------------------------------------------------- 1. merit order
def fig_merit_order(f):
    """Prix en fonction de la charge résiduelle prévue, 2024-2026, avec le SRMC médian."""
    d = f[f["time_local"] >= pd.Timestamp("2024-01-01", tz=TZ)].dropna(subset=["fr_residual_fc", "price_da"])
    x, y = d["fr_residual_fc"] / 1000, d["price_da"].clip(-100, 300)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    hb = ax.hexbin(x, y, gridsize=60, cmap="Blues", mincnt=1, bins="log", linewidths=0)
    bins = pd.cut(x, np.arange(10, 80, 2.5))
    med = y.groupby(bins, observed=True).median()
    ax.plot([b.mid for b in med.index], med.values, color=INK, lw=2, label="Median price")
    ax.axhline(d["srmc_ccgt"].median(), color=ORANGE, lw=2, ls="--", label="Median CCGT SRMC")
    ax.axhline(0, color=INK_2, lw=0.8)
    ax.set_xlabel("French residual load forecast (GW)")
    ax.set_ylabel("Day-ahead price (€/MWh, clipped to [-100, 300])")
    ax.set_title("Merit order: price vs residual load, 2024 – Oct. 2026")
    ax.legend(loc="upper left")
    fig.colorbar(hb, ax=ax, label="Hours (log scale)", pad=0.01)
    _save(fig, "01_merit_order.png")


# --------------------------------------------------------------------- 2. résultats du test
def fig_results(f, preds):
    models = ["naive_d7", "MSTL", "LEAR_quotidien", "Chronos2_prix", "A2", "B1", "B2",
              "B2_hebdo", "B2_quotidien", "B2_quotidien_corr", "B3_quotidien_hl90_corr", "Chronos2_fond"]
    rows = f.loc[preds.index]
    ok = preds[models].notna().all(axis=1) & rows["price_da"].notna()
    err = preds.loc[ok, models].sub(rows.loc[ok, "price_da"], axis=0).abs()
    reg = regime(rows[ok])
    panels = [("All hours", slice(None)), ("Negative-price hours", reg == "negatif"),
              ("Evening peak (18-22h)", reg == "pointe_soir"), ("Other hours", reg == "reste")]
    fig, axes = plt.subplots(1, 4, figsize=(13, 4.9), sharey=True)
    for ax, (title, m) in zip(axes, panels):
        mae = err[m].mean()
        colors = [MODEL_COLORS.get(mo, GRAY) for mo in models]
        ax.barh(range(len(models)), mae.values, color=colors, height=0.7)
        for i, v in enumerate(mae.values):
            ax.text(v + 0.4, i, f"{v:.1f}", va="center", fontsize=8, color=INK_2)
        ax.set_title(title)
        ax.set_xlabel("MAE (€/MWh)")
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, mae.max() * 1.22)
    axes[0].set_yticks(range(len(models)), [LABELS[m] for m in models])
    fig.suptitle("Test period (Nov. 2025 – Oct. 2026): forecast error by model and regime"
                 "  (* post-test extension, chosen on validation)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    _save(fig, "02_test_results.png")


# --------------------------------------------------------------------- 3. rythme de réentraînement
def fig_retraining():
    rows = []
    for period in ("validation", "test"):
        m = pd.read_csv(RESULTS / f"metrics_{period}.csv", header=[0, 1], index_col=0)[("toutes", "MAE")]
        for fam, names in {"LightGBM B2": ["B2", "B2_hebdo", "B2_quotidien"],
                           "LEAR": ["LEAR_mensuel", "LEAR_hebdo", "LEAR_quotidien"]}.items():
            rows += [(period, fam, freq, m[n]) for freq, n in zip(["Monthly", "Weekly", "Daily"], names)]
    d = pd.DataFrame(rows, columns=["period", "model", "freq", "mae"])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    for ax, period in zip(axes, ["validation", "test"]):
        for fam, color in [("LightGBM B2", BLUE), ("LEAR", AQUA)]:
            s = d[(d.period == period) & (d.model == fam)]
            ax.plot(s.freq, s.mae, color=color, lw=2, marker="o", ms=8, label=fam)
            ax.annotate(fam, (2, s.mae.iloc[-1]), xytext=(8, 0), textcoords="offset points",
                        va="center", color=INK_2, fontsize=9)
        ax.set_title(f"{period.capitalize()} ({PERIODS[period][0][:7]} → {PERIODS[period][1][:7]})")
        ax.set_xlim(-0.3, 2.9)
    axes[0].set_ylabel("MAE (€/MWh)")
    axes[0].legend(loc="upper right")
    fig.suptitle("Retraining more often helps both model families", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    _save(fig, "03_retraining.png")


# --------------------------------------------------------------------- 4. une semaine de test
def fig_week(f, preds, start="2026-05-04"):
    a, b = pd.Timestamp(start, tz=TZ), pd.Timestamp(start, tz=TZ) + pd.Timedelta(days=7)
    m = (f["time_local"] >= a) & (f["time_local"] < b)
    t = f.loc[m, "time_local"]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, f.loc[m, "price_da"], color=INK, lw=2.4, label="Actual price")
    for mo in ["B2_quotidien", "Chronos2_fond", "LEAR_quotidien"]:
        ax.plot(t, preds.loc[f.index[m], mo], color=MODEL_COLORS[mo], lw=1.6, label=LABELS[mo])
    ax.axhline(0, color=INK_2, lw=0.8)
    ax.set_ylabel("€/MWh")
    last = b - pd.Timedelta(days=1)
    ax.set_title(f"One test week ({a:%d %b} – {last:%d %b %Y}): actual price vs forecasts")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4)
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.DayLocator(tz=TZ))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d", tz=TZ))
    _save(fig, "04_test_week.png")


# --------------------------------------------------------------------- 5-6. importance
def fig_importance():
    shap, rows = importance.shap_values()
    mag = shap.abs()
    overall = mag.mean().sort_values()
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.barh(overall.index, overall.values, color=BLUE, height=0.7)
    for i, v in enumerate(overall.values):
        ax.text(v + 0.2, i, f"{v:.1f}", va="center", fontsize=8, color=INK_2)
    ax.set_xlabel("Mean |SHAP value| (€/MWh)")
    ax.set_title("What drives the main model (B2), test period")
    ax.grid(axis="y", visible=False)
    _save(fig, "05_importance.png")

    order = overall.index[::-1]
    reg = regime(rows).map({"negatif": "Negative\nhours", "pointe_soir": "Evening\npeak", "reste": "Other\nhours"})
    by_reg = mag.groupby(reg.to_numpy()).mean().T.loc[order, ["Negative\nhours", "Evening\npeak", "Other\nhours"]]
    by_hour = mag.groupby(rows["time_local"].dt.hour.to_numpy()).mean().T.loc[order]
    share_reg = by_reg / by_reg.sum()
    share_hour = by_hour / by_hour.sum()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1, 4]}, sharey=True)
    vmax = max(share_reg.values.max(), share_hour.values.max())
    for ax, s, title in [(axes[0], share_reg, "By regime"), (axes[1], share_hour, "By hour of day")]:
        im = ax.imshow(s.values, aspect="auto", cmap="Blues", vmin=0, vmax=vmax)
        ax.set_xticks(range(s.shape[1]), s.columns, fontsize=8)
        ax.set_title(title)
        ax.grid(False)
    axes[0].set_yticks(range(len(order)), order)
    for i in range(share_reg.shape[0]):
        for j in range(share_reg.shape[1]):
            v = share_reg.values[i, j]
            axes[0].text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=7,
                         color="white" if v > vmax * 0.6 else INK)
    axes[1].set_xlabel("Hour (local time)")
    fig.colorbar(im, ax=axes, label="Share of total |SHAP| in the column", pad=0.01)
    fig.suptitle("Feature importance by regime and hour (share of each column's total)",
                 x=0.01, ha="left", fontweight="bold")
    _save(fig, "06_importance_regime_hour.png")


def main():
    f = add_features(load_dataset())
    preds = load_predictions("test")
    for name, col in NAIVE.items():   # les naïfs ne sont pas sauvés : ce sont des features
        preds[name] = f.loc[preds.index, col]
    fig_merit_order(f)
    fig_results(f, preds)
    fig_retraining()
    fig_week(f, preds)
    fig_importance()


if __name__ == "__main__":
    main()
