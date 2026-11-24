"""README figures (results/figures/*.png).

Reference categorical palette, fixed order: blue = LightGBM (main model), orange = Chronos-2,
aqua = LEAR, yellow = final extension model, gray = other; actual price in black.
Each figure answers one question and varies one thing (information, retraining pace, model, target).

Usage: python -m src.figures   (after the backtests of every period: figures read results/*.csv
and the saved test forecasts; the SHAP figures retrain the main model once)
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.backtest import load_predictions, regime
from src.config import PERIODS, RESULTS, TZ
from src.data import load_dataset
from src.features import add_features
from src import importance

FIG = RESULTS / "figures"

# Palette (reference instance, light mode)
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#b9b8b2"
# Models of the test-week figure: name -> (label, color)
WEEK_MODELS = {"B2_daily": ("LightGBM B2 (daily)", BLUE),
               "Chronos2_fund": ("Chronos-2 + fundamentals", ORANGE),
               "LEAR_B2_daily": ("LEAR-B2 (daily)", AQUA)}

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


# --------------------------------------------------------------------- helpers
def _metrics(period):
    return pd.read_csv(RESULTS / f"metrics_{period}.csv", header=[0, 1], index_col=0)


def _bar_labels(ax, bars, fmt="{:.1f}"):
    for b in bars:
        ax.annotate(fmt.format(b.get_height()), (b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(0, 2), textcoords="offset points", ha="center", va="bottom",
                    fontsize=8, color=INK_2)


# --------------------------------------------------------------------- 01. merit order
def fig_merit_order(f):
    """Price against forecast residual load, 2024-2026, with the median SRMC."""
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


# --------------------------------------------------------------------- 02. information blocks
def fig_information():
    """Ablation: one block of information added at a time (LightGBM, weekly retraining)."""
    steps = [("A1_weekly", "Calendar"), ("A2_weekly", "+ price lags"), ("B1_weekly", "+ FR\nfundamentals"),
             ("B2_weekly", "+ gas, CO2,\nGermany")]
    m = {p: _metrics(p) for p in ("validation", "test")}
    x = np.arange(len(steps))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (reg, title) in zip(axes, [("all", "All hours"), ("negative", "Negative-price hours")]):
        for k, (period, color) in enumerate([("validation", GRAY), ("test", BLUE)]):
            vals = [m[period].loc[s, (reg, "MAE")] for s, _ in steps]
            bars = ax.bar(x + (k - 0.5) * 0.38, vals, width=0.36, color=color, label=period.capitalize())
            _bar_labels(ax, bars)
        ax.set_xticks(x, [lab for _, lab in steps])
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("MAE (€/MWh)")
    axes[0].legend(loc="upper right")
    fig.suptitle("What each block of information is worth (LightGBM, retrained weekly)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    _save(fig, "02_information.png")


# --------------------------------------------------------------------- 03. retraining pace
def fig_retraining():
    """Same information, retraining every month, week or day (LightGBM B2 and LEAR)."""
    rows = []
    for period in ("validation", "test"):
        m = _metrics(period)[("all", "MAE")]
        for fam, names in {"LightGBM B2": ["B2", "B2_weekly", "B2_daily"],
                           "LEAR": ["LEAR_monthly", "LEAR_weekly", "LEAR_daily"]}.items():
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


# --------------------------------------------------------------------- 04. models at equal information
def fig_models():
    """Test year, daily updates: price-only models, then three models with B2's information."""
    rows = [("MSTL", "MSTL", GRAY, 1), ("Chronos2_prices", "Chronos-2", ORANGE, 0.45),
            ("LEAR_B2_daily", "LEAR", AQUA, 1), ("B2_daily", "LightGBM", BLUE, 1),
            ("Chronos2_fund", "Chronos-2", ORANGE, 1)]
    y = np.array([5.0, 4.0, 2.4, 1.4, 0.4])            # top to bottom, with a gap between the two groups
    m = _metrics("test")
    panels = [("all", "All hours"), ("negative", "Negative-price hours"),
              ("evening_peak", "Evening peak (18-22h)"), ("other", "Other hours")]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.8), sharey=True)
    for ax, (reg, title) in zip(axes, panels):
        vals = [m.loc[r[0], (reg, "MAE")] for r in rows]
        for yi, v, (_, _, color, alpha) in zip(y, vals, rows):
            ax.barh(yi, v, color=color, alpha=alpha, height=0.75)
            ax.text(v + 0.4, yi, f"{v:.1f}", va="center", fontsize=8, color=INK_2)
        naive = m.loc["naive_d7", (reg, "MAE")]
        ax.set_xlim(0, max(vals) * 1.25)
        ax.set_ylim(-0.3, 6.2)
        ax.set_title(f"{title}\n(naive D-7: {naive:.1f})")
        ax.set_xlabel("MAE (€/MWh)")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [r[1] for r in rows])
    for yy, lab in [(y[0] + 0.75, "Prices only"), (y[2] + 0.75, "Same information as B2")]:
        axes[0].text(0.01, yy, lab, transform=axes[0].get_yaxis_transform(), ha="left", va="center",
                     fontsize=8.5, fontweight="bold", color=INK_2)
    fig.suptitle("Which model, at equal information? Test year, models updated daily",
                 x=0.01, ha="left", fontweight="bold", y=1.04)
    fig.tight_layout()
    _save(fig, "04_models.png")


# --------------------------------------------------------------------- 05. normalisation by the gas cost
def fig_normalisation():
    """Raw-price target vs price − SRMC vs residual-load-weighted SRMC, by period."""
    targets = [("raw", "Raw price", GRAY), ("full", "Price − SRMC", BLUE), ("rl", "Price − w(RL) × SRMC", "#8fb8ea")]
    names = {"monthly": {"raw": "B2_raw", "full": "B2", "rl": "B2_rl"},
             "weekly": {"raw": "B2_raw_weekly", "full": "B2_weekly", "rl": "B2_rl_weekly"}}
    panels = [("crisis", "all", "monthly", "Gas crisis 2021-22\n(monthly retraining)"),
              ("validation", "all", "weekly", "Validation year"),
              ("test", "all", "weekly", "Test year"),
              ("test", "negative", "weekly", "Test year,\nnegative-price hours")]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.6))
    for ax, (period, reg, pace, title) in zip(axes, panels):
        m = _metrics(period)
        vals = [m.loc[names[pace][t], (reg, "MAE")] for t, _, _ in targets]
        bars = ax.bar(range(3), vals, color=[c for _, _, c in targets], width=0.7)
        _bar_labels(ax, bars)
        ax.set_xticks(range(3), ["raw", "− SRMC", "− w·SRMC"])
        ax.set_ylim(0, max(vals) * 1.18)
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("MAE (€/MWh)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in targets]
    fig.legend(handles, [lab for _, lab, _ in targets], loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Normalising the target by the gas-plant cost: insurance in crises, noise when prices are negative",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    _save(fig, "05_normalisation.png")


# --------------------------------------------------------------------- 06. post-test extension
def fig_extension():
    """From the main model to the final model, one ingredient at a time (test year, daily)."""
    steps = [("B2_daily", "B2 daily\n(main model)", BLUE), ("B2_daily_corr", "+ error\ncorrection", GRAY),
             ("B3_daily_corr", "+ day shape,\nregime (B3)", GRAY), ("B3_daily_hl90_corr", "+ recency\nweights (final)", YELLOW)]
    m = _metrics("test")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (reg, title) in zip(axes, [("all", "All hours"), ("negative", "Negative-price hours")]):
        vals = [m.loc[s, (reg, "MAE")] for s, _, _ in steps]
        bars = ax.bar(range(len(steps)), vals, color=[c for _, _, c in steps], width=0.65)
        _bar_labels(ax, bars)
        ch = m.loc["Chronos2_fund", (reg, "MAE")]
        ax.axhline(ch, color=ORANGE, lw=1.8, ls="--")
        ax.text(1.01, ch, f"Chronos-2\n{ch:.1f}", transform=ax.get_yaxis_transform(), va="center",
                fontsize=8, color=ORANGE, fontweight="bold")
        ax.set_xticks(range(len(steps)), [lab for _, lab, _ in steps])
        ax.set_ylim(0, max(vals + [ch]) * 1.18)
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("MAE (€/MWh)")
    fig.suptitle("Post-test extension: borrowing Chronos-2's ingredients (test year, daily retraining)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    _save(fig, "06_extension.png")


# --------------------------------------------------------------------- 07. one test week
def fig_week(f, preds, start="2026-05-04"):
    """Actual price and three forecasts over one test week."""
    a, b = pd.Timestamp(start, tz=TZ), pd.Timestamp(start, tz=TZ) + pd.Timedelta(days=7)
    m = (f["time_local"] >= a) & (f["time_local"] < b)
    t = f.loc[m, "time_local"]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, f.loc[m, "price_da"], color=INK, lw=2.4, label="Actual price")
    for mo, (label, color) in WEEK_MODELS.items():
        ax.plot(t, preds.loc[f.index[m], mo], color=color, lw=1.6, label=label)
    ax.axhline(0, color=INK_2, lw=0.8)
    ax.set_ylabel("€/MWh")
    last = b - pd.Timedelta(days=1)
    ax.set_title(f"One test week ({a:%d %b} – {last:%d %b %Y}): actual price vs forecasts")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4)
    ax.xaxis.set_major_locator(mdates.DayLocator(tz=TZ))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d", tz=TZ))
    _save(fig, "07_test_week.png")


# --------------------------------------------------------------------- 08-09. feature importance
def fig_importance():
    """SHAP importance of the main model (B2), overall and by regime and hour."""
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
    _save(fig, "08_importance.png")

    order = overall.index[::-1]
    reg = regime(rows).map({"negative": "Negative\nhours", "evening_peak": "Evening\npeak", "other": "Other\nhours"})
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
    _save(fig, "09_importance_regime_hour.png")


def main():
    f = add_features(load_dataset())
    preds = load_predictions("test")
    fig_merit_order(f)
    fig_information()
    fig_retraining()
    fig_models()
    fig_normalisation()
    fig_extension()
    fig_week(f, preds)
    fig_importance()


if __name__ == "__main__":
    main()
