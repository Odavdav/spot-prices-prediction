"""Rapport PDF (report.pdf) construit à partir de results/ : tableaux lus dans les CSV, figures PNG.

Texte en anglais, comme le README. Usage : python -m src.report   (après src.figures)
"""
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

from src.config import ROOT, RESULTS

OUT = ROOT / "report.pdf"
FIG = RESULTS / "figures"

# Police Unicode (€, −, ×, →) si disponible, sinon Helvetica
FONT, FONT_BOLD = "Helvetica", "Helvetica-Bold"
_SUP = Path("/System/Library/Fonts/Supplemental")
if (_SUP / "Arial Unicode.ttf").exists() and (_SUP / "Arial Bold.ttf").exists():
    pdfmetrics.registerFont(TTFont("ArialU", str(_SUP / "Arial Unicode.ttf")))
    pdfmetrics.registerFont(TTFont("ArialB", str(_SUP / "Arial Bold.ttf")))
    italic = "ArialU"
    if (_SUP / "Arial Italic.ttf").exists():
        pdfmetrics.registerFont(TTFont("ArialI", str(_SUP / "Arial Italic.ttf")))
        italic = "ArialI"
    # Famille : nécessaire pour que <b> et <i> dans les paragraphes changent vraiment de police
    from reportlab.pdfbase.pdfmetrics import registerFontFamily
    registerFontFamily("ArialU", normal="ArialU", bold="ArialB", italic=italic, boldItalic="ArialB")
    FONT, FONT_BOLD = "ArialU", "ArialB"

INK, INK_2, GRID, BLUE = colors.HexColor("#0b0b0b"), colors.HexColor("#52514e"), colors.HexColor("#e4e3df"), colors.HexColor("#2a78d6")

base = getSampleStyleSheet()
S = {
    "title": ParagraphStyle("t", parent=base["Title"], fontName=FONT_BOLD, fontSize=19, leading=24, textColor=INK, alignment=0),
    "sub": ParagraphStyle("s", parent=base["Normal"], fontName=FONT, fontSize=10.5, leading=14, textColor=INK_2),
    "h1": ParagraphStyle("h1", parent=base["Heading1"], fontName=FONT_BOLD, fontSize=13.5, leading=17, spaceBefore=12, spaceAfter=6, textColor=INK),
    "h2": ParagraphStyle("h2", parent=base["Heading2"], fontName=FONT_BOLD, fontSize=11, leading=14, spaceBefore=8, spaceAfter=4, textColor=INK),
    "body": ParagraphStyle("b", parent=base["Normal"], fontName=FONT, fontSize=9.5, leading=13.2, alignment=TA_JUSTIFY, spaceAfter=5, textColor=INK),
    "bullet": ParagraphStyle("bl", parent=base["Normal"], fontName=FONT, fontSize=9.5, leading=13, leftIndent=12, bulletIndent=2, spaceAfter=2, textColor=INK),
    "caption": ParagraphStyle("c", parent=base["Normal"], fontName=FONT, fontSize=8, leading=10.5, textColor=INK_2, spaceAfter=8),
    "cell": ParagraphStyle("cell", parent=base["Normal"], fontName=FONT, fontSize=8, leading=10, textColor=INK),
}

NAMES = {
    "naive_d1": "Naive D-1", "naive_d7": "Naive D-7", "A1": "A1: calendar", "A2": "A2: + price lags",
    "B1": "B1: + FR fundamentals", "B2": "B2: + gas, CO2, Germany", "B2_brut": "B2, raw-price target",
    "B2_rl": "B2, residual-load-weighted target",
    "B1_hebdo": "B1, weekly retraining", "B2_hebdo": "B2, weekly retraining",
    "B2_quotidien": "B2, daily retraining (final)", "B2_rl_hebdo": "B2 weighted target, weekly",
    "B2_brut_hebdo": "B2 raw target, weekly", "MSTL": "MSTL", "LEAR_mensuel": "LEAR, monthly",
    "LEAR_hebdo": "LEAR, weekly", "LEAR_quotidien": "LEAR, daily",
    "Chronos2_prix": "Chronos-2, prices only", "Chronos2_fond": "Chronos-2 + fundamentals",
}


def P(text, style="body"):
    return Paragraph(text, S[style])


def bullets(items):
    return [Paragraph(t, S["bullet"], bulletText="•") for t in items]


def table(rows, widths, header_rows=1, highlight=()):
    data = [[Paragraph(str(c), S["cell"]) for c in r] for r in rows]
    t = Table(data, colWidths=widths, repeatRows=header_rows)
    style = [
        ("LINEBELOW", (0, header_rows - 1), (-1, header_rows - 1), 0.8, INK_2),
        ("LINEBELOW", (0, header_rows), (-1, -1), 0.3, GRID),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ]
    for i in highlight:
        style.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#eaf2fc")))
    t.setStyle(TableStyle(style))
    return t


def figure(name, width_cm, caption):
    path = FIG / name
    img = Image(str(path))
    ratio = img.imageHeight / img.imageWidth
    img.drawWidth, img.drawHeight = width_cm * cm, width_cm * cm * ratio
    return KeepTogether([img, P(caption, "caption")])


def metrics_table(period, order):
    m = pd.read_csv(RESULTS / f"metrics_{period}.csv", header=[0, 1], index_col=0)
    rows = [["Model", "MAE all", "rMAE", "MAE negative", "MAE evening peak", "MAE other"]]
    hl = []
    ref = m.loc["naive_d7", ("toutes", "MAE")]
    for name in [o for o in order if o in m.index]:
        r = m.loc[name]
        # rMAE recalculé depuis les MAE (le CSV est arrondi à 3 décimales : éviter un double arrondi)
        rows.append([NAMES.get(name, name), f"{r[('toutes', 'MAE')]:.2f}", f"{r[('toutes', 'MAE')] / ref:.2f}",
                     f"{r[('negatif', 'MAE')]:.2f}", f"{r[('pointe_soir', 'MAE')]:.2f}", f"{r[('reste', 'MAE')]:.2f}"])
        if name in ("B2_quotidien", "Chronos2_fond"):
            hl.append(len(rows) - 1)
    hours = m.xs("heures", axis=1, level=1).iloc[0].astype(int)
    note = (f"{hours['toutes']:,} hours, of which {hours['negatif']} negative-price hours and "
            f"{hours['pointe_soir']:,} evening-peak hours. MAE in €/MWh; rMAE = MAE / MAE of naive D-7.")
    return table(rows, [5.6 * cm] + [2.2 * cm] * 5, highlight=hl), note


def dm_table(period):
    d = pd.read_csv(RESULTS / f"diebold_mariano_{period}.csv", index_col=0)
    rows = [["Comparison (A vs B)", "MAE gap A − B (€/MWh)", "HAC t-stat", "p-value"]]
    for pair, r in d.iterrows():
        a, b = pair.split(" vs ")
        rows.append([f"{NAMES.get(a, a)} vs {NAMES.get(b, b)}", f"{r.ecart_MAE:+.2f}",
                     f"{r.t_HAC:+.2f}", "< 0.001" if r.p_value < 0.001 else f"{r.p_value:.3f}"])
    return table(rows, [9 * cm, 3.4 * cm, 2.3 * cm, 2.3 * cm])


ORDER = ["naive_d1", "naive_d7", "MSTL", "LEAR_mensuel", "LEAR_hebdo", "LEAR_quotidien",
         "Chronos2_prix", "A1", "A2", "B1", "B2", "B2_brut", "B2_rl", "B1_hebdo", "B2_hebdo",
         "B2_rl_hebdo", "B2_brut_hebdo", "B2_quotidien", "Chronos2_fond"]


def build():
    story = [
        P("Forecasting French day-ahead power prices", "title"),
        P("What each block of market information is worth, tested out of sample. "
          "Validation Nov. 2024 – Oct. 2025, test Nov. 2025 – Oct. 2026.", "sub"),
        Spacer(1, 10),
    ]

    # ---------------------------------------------------------------- résumé
    story += [P("Summary", "h1"), P(
        "We forecast the hourly French day-ahead (SDAC) price using only information published before the "
        "auction closes at 12:00 on D-1. A single gradient-boosting model (LightGBM) receives market information "
        "step by step: calendar, price history, French fundamentals (load, wind and solar forecasts, residual "
        "load), then gas and carbon costs (through a combined-cycle gas turbine's short-run marginal cost, SRMC) "
        "and German data. It is compared with naive forecasts, a seasonal decomposition (MSTL), the literature's "
        "linear reference (LEAR) and a zero-shot foundation model (Chronos-2), in a walk-forward backtest. "
        "Differences are tested with Diebold-Mariano tests using a HAC (Newey-West) variance.")]
    story += bullets([
        "The final model (B2, retrained daily) reaches an MAE of <b>17.3 €/MWh</b> on the test year, "
        "44 % of the naive D-7 error, and beats LEAR by 3.2 €/MWh (p < 0.001).",
        "Fundamentals cut the error by 17–18 % beyond price lags, and by 27 % on negative-price hours (validation).",
        "Retraining daily instead of monthly is worth 2.5 €/MWh on the test year: model freshness is a first-order choice.",
        "Normalising the target by the SRMC is insurance: no effect in calm years, −17 % error in the 2021–2022 gas crisis.",
        "Chronos-2, zero-shot but given the same fundamentals, does better still (14.4 €/MWh, p < 0.001), "
        "except on negative-price hours, where the two models are level.",
    ])

    # ---------------------------------------------------------------- données
    story += [P("1. Data and information set", "h1"), P(
        "Every input for delivery day D must be known before 12:00 (Paris time) on D-1. No realised data "
        "(actual generation, physical flows) is used. The dataset covers 1 Oct. 2018, when the German-"
        "Luxembourg bidding zone was created, to 20 Nov. 2026: 71,353 hours. Since 1 Oct. 2025 the market "
        "clears in 15-minute products; we work with hourly averages for a homogeneous history.")]
    story.append(table([
        ["Data", "Source", "Known at decision time"],
        ["French and German-Luxembourg day-ahead prices", "ENTSO-E Transparency", "up to D-1"],
        ["Load forecast D-1 (FR, DE)", "ENTSO-E", "before 10:00 on D-1"],
        ["Wind and solar forecasts D-1 (FR, DE)", "ENTSO-E", "assumed before 12:00 on D-1"],
        ["TTF gas, front month", "Yahoo Finance (TTF=F)", "last close dated ≤ D-2"],
        ["EUA carbon, December contract", "Investing.com (manual CSV)", "last close dated ≤ D-2"],
    ], [6.4 * cm, 4.8 * cm, 4.6 * cm]))
    story += [Spacer(1, 6), P(
        "<b>Leakage control.</b> A register declares when each raw input is published and which inputs, with "
        "which day lag, each feature uses. An automated test checks every declaration against the 12:00 rule, "
        "then recomputes the features after erasing everything published after gate closure, and checks that "
        "day D is unchanged. Test days include clock changes and Easter. A deliberately planted leak makes the "
        "test fail.")]

    # ---------------------------------------------------------------- méthode
    story += [P("2. Method", "h1"), P("2.1 Features and ablation steps", "h2"), table([
        ["Step", "Features added", "Target"],
        ["A1", "hour, weekday, month, French holidays, bridge days, clock-change days", "price"],
        ["A2", "price at the same local hour on D-1, D-2, D-7; mean, min, max of D-1", "price"],
        ["B1", "French load, wind and solar forecasts; residual load; renewables share", "price"],
        ["B2", "SRMC = TTF / 0.55 + EUA × 0.37 + 2; German price D-1; German load and renewables",
         "price − SRMC"],
    ], [1.3 * cm, 11.5 * cm, 3 * cm])]
    story += [Spacer(1, 4), P(
        "In B2 the price lags are also expressed relative to the SRMC, so that inputs and target share the same "
        "scale. Two control variants keep B2's features with a raw-price target (B2 raw) or with "
        "price − w(RL) × SRMC (B2 weighted), where w is the pass-through of gas cost to the French price by "
        "residual-load decile (0.71 to 1.47), estimated on data before the validation year.")]
    story += [P("2.2 Benchmarks", "h2")] + bullets([
        "<b>Naive</b> D-1 and D-7 (same hour yesterday or last week). D-7 is the reference for the rMAE.",
        "<b>MSTL</b>: decomposition of the last 8 weeks of prices into trend, daily and weekly seasonality, re-fitted daily.",
        "<b>LEAR</b> (Lago et al., 2021): one LASSO regression per hour, inputs asinh-transformed, average of models "
        "calibrated on 56, 84, 1,092 and 1,456 days; same information as B1; re-estimated monthly, weekly or daily.",
        "<b>Chronos-2</b> (Ansari et al., 2025): zero-shot transformer, 2,048 hours of context, median forecast; "
        "prices only, or with the same known-in-advance variables as B2. Released on 17 Oct. 2025, it may "
        "have seen the validation year but not the test year, so it is reported on the test year only.",
    ])
    story += [P("2.3 Protocol and statistics", "h2"), P(
        "Walk-forward backtest with an expanding window. At each retraining date (1st of the month, every 7 days "
        "or every day) the model is fitted on all earlier deliveries, then forecasts until the next retraining. "
        "Features are always recomputed from the latest data. LightGBM uses fixed hyperparameters (800 trees, "
        "learning rate 0.03, 63 leaves, absolute-error objective); no tuning was done. All modelling choices were "
        "made on the validation year; the test year was run once. Metrics are the MAE and the rMAE, overall and for "
        "three regimes: negative-price hours, evening peak (18:00–22:00) and other hours.")]
    story += [P(
        "Diebold-Mariano tests compare two models on the daily average of their absolute-error differences "
        "(hours within a day are strongly correlated). The variance is estimated with Newey-West, using "
        "floor(4 (T/100)<super>2/9</super>) = 5 lags for T = 365 days, and p-values come from the normal "
        "distribution. Comparison pairs were fixed before the runs, except where stated.")]

    # ---------------------------------------------------------------- résultats
    story += [PageBreak(), P("3. Results", "h1"), P("3.1 Test year (Nov. 2025 – Oct. 2026)", "h2")]
    t, note = metrics_table("test", ORDER)
    story += [t, P(note, "caption"),
              figure("02_test_results.png", 17, "Figure 1. Test-year MAE by model and regime (subset of models).")]
    story += [P("3.2 Validation year (Nov. 2024 – Oct. 2025)", "h2")]
    t, note = metrics_table("validation", ORDER)
    story += [t, P(note + " Chronos-2 not run on validation (possible contamination).", "caption")]
    story += [P("3.3 Gas crisis (Jul. 2021 – Dec. 2022), monthly retraining", "h2")]
    t, note = metrics_table("crise", ["naive_d1", "naive_d7", "B2", "B2_brut", "B2_rl"])
    story += [t, P(note, "caption"), P(
        "Normalising the target by the SRMC cuts the error by 17 % during the crisis, and the gain grows with "
        "the gas price (16 €/MWh in Q3 2022, when TTF averaged 202 €/MWh). Weighting the normalisation by residual "
        "load does not improve on full normalisation in any period.")]
    story += [figure("03_retraining.png", 15, "Figure 2. Effect of retraining frequency on overall MAE.")]
    story += [P("3.4 Statistical tests", "h2"), P("<b>Test year</b>", "body"), dm_table("test"),
              Spacer(1, 6), P("<b>Validation year</b>", "body"), dm_table("validation"),
              P("Negative gap: model A has the lower error. The pairs B2 weekly vs LEAR daily (validation) and "
                "B2 daily vs Chronos-2 (test) were added after seeing the results, so their p-values are "
                "slightly optimistic.", "caption")]
    story += [figure("04_test_week.png", 16, "Figure 3. One test week: actual price and three forecasts.")]

    # ---------------------------------------------------------------- importance
    story += [P("4. Feature importance", "h1"), P(
        "SHAP values (TreeSHAP) for B2, trained on all data before the test and explained on the test hours. "
        "They decompose each forecast of price − SRMC into additive contributions in €/MWh; the SRMC itself is "
        "added back afterwards and does not appear.")]
    story += [figure("05_importance.png", 13, "Figure 4. Mean absolute SHAP value by feature group.")]
    story += [figure("06_importance_regime_hour.png", 17,
                     "Figure 5. Share of each feature group in total |SHAP|, by regime and hour of day.")]
    story += bullets([
        "Yesterday's price at the same hour dominates in every regime (23–27 % of total importance).",
        "French and German residual load come next, ahead of the raw load, wind and solar forecasts: the model "
        "reasons in terms of what dispatchable plants must cover, across the coupled FR-DE zone.",
        "Negative-price hours rely more on residual load (16 % each for FR and DE, against 11 % at the evening "
        "peak) and less on the calendar (3 % against 7 %).",
        "The evening hours (19:00–21:00) lean most on yesterday's price; midday hours give more weight to the "
        "renewables share, which is the solar effect.",
    ])
    story += [figure("01_merit_order.png", 13, "Figure 6. Price against forecast residual load, 2024 – Oct. 2026.")]

    # ---------------------------------------------------------------- discussion
    story += [P("5. Discussion", "h1")] + bullets([
        "<b>Fundamentals beat history,</b> especially when renewables push prices negative.",
        "<b>Gas sets the French price less and less often:</b> the average spread between price and SRMC fell "
        "from +1–6 €/MWh (2018–2022) to −35 €/MWh (2024–2026). Normalising by gas cost protects against fuel "
        "shocks but explains less of everyday prices.",
        "<b>Model freshness is a first-order choice.</b> Both LightGBM and LEAR gain significantly from "
        "frequent retraining; LEAR, built on short windows, degrades sharply when updated monthly.",
        "<b>Linear vs non-linear.</b> With the same information and cadence, LightGBM beats LEAR on the test "
        "(p = 0.01) but not on validation (p = 0.47). Its edge is largest on negative-price hours.",
        "<b>Foundation models.</b> Chronos-2 re-learns the relationship between price and fundamentals from "
        "the last 12 weeks at every forecast. With the same information it beat LightGBM by 16 % on the test, "
        "but can overshoot extreme renewables hours (Figure 3).",
    ])

    story += [P("6. Limitations", "h1")] + bullets([
        "Wind and solar D-1 forecasts are only required by 18:00 on D-1; we assume availability at 12:00, as is "
        "standard. ENTSO-E keeps only the latest version of each forecast.",
        "Free fuel proxies: TTF front month (Yahoo) and EUA December contract (Investing.com), with contract-roll "
        "jumps. Closes are taken at D-2, which is conservative.",
        "Hourly averages since the move to 15-minute products (Oct. 2025).",
        "No French nuclear availability (as-announced outage notices), the most promising missing input.",
        "Chronos-2's training data is not published; contamination of the test is ruled out by its release "
        "date only, and the model could not be checked on validation.",
        "A single test year, fixed hyperparameters, and some comparisons chosen after seeing the results.",
        "The gas pass-through weights w(RL) drift over time (0.26 at low residual load in 2024, against "
        "0.5–0.8 before).",
    ])

    story += [P("References", "h1")] + bullets([
        "Lago, J., Marcjasz, G., De Schutter, B., Weron, R. (2021). Forecasting day-ahead electricity prices: "
        "a review of state-of-the-art algorithms, best practices and an open-access benchmark. "
        "<i>Applied Energy</i> 293, 116983.",
        "Diebold, F. X., Mariano, R. S. (1995). Comparing predictive accuracy. "
        "<i>Journal of Business &amp; Economic Statistics</i> 13(3), 253–263.",
        "Newey, W. K., West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and "
        "autocorrelation consistent covariance matrix. <i>Econometrica</i> 55(3), 703–708.",
        "Ansari, A. F., Shchur, O., et al. (2025). Chronos-2: from univariate to universal forecasting. arXiv:2510.15821.",
        "Ke, G., et al. (2017). LightGBM: a highly efficient gradient boosting decision tree. <i>NeurIPS</i>.",
        "Lundberg, S. M., Lee, S.-I. (2017). A unified approach to interpreting model predictions. <i>NeurIPS</i>.",
        "Bandara, K., Hyndman, R. J., Bergmeir, C. (2021). MSTL: a seasonal-trend decomposition algorithm for "
        "time series with multiple seasonal patterns. arXiv:2107.13462.",
    ])
    story += [P("Acknowledgements", "h1"), P(
        "Code and analysis were developed with the help of Claude (Anthropic), used as an AI coding assistant. "
        "Research design, data choices and interpretation of the results are my own.")]

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont(FONT, 7.5)
        canvas.setFillColor(INK_2)
        canvas.drawString(2 * cm, 1.2 * cm, "French day-ahead power price forecasting")
        canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"{doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                            title="Forecasting French day-ahead power prices", author="",
                            subject="Day-ahead price forecasting, ablation and benchmarks")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"Écrit : {OUT}")


if __name__ == "__main__":
    build()
