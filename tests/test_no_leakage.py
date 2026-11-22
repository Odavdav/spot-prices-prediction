"""Tests anti-fuite : aucune feature du jour J ne doit utiliser une information publiée après la
fermeture de l'enchère SDAC (12h00 en J-1).

Les tests tournent sur un jeu synthétique : ils ne dépendent pas de data/ (non versionné).
Lancer depuis la racine : python -m pytest
"""
import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from src.features import (ALL_FEATURES, FEATURE_REGISTRY, RAW_PUBLICATION, STEPS, TZ, add_features,
                          get_X, is_known_at_gate_closure, mask_unpublished)


@pytest.fixture(scope="module")
def synthetic() -> pd.DataFrame:
    """Données horaires aléatoires autour du passage à l'heure d'été (31/03/2024) et de Pâques, avec
    assez d'historique (6 semaines) pour que les fenêtres glissantes de 14 et 28 jours soient remplies."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-02-15", "2024-04-12", freq="h", tz="UTC")
    df = pd.DataFrame(index=idx)
    for col in RAW_PUBLICATION:
        df[col] = rng.normal(100, 30, len(idx))
    df["time_local"] = idx.tz_convert(TZ)
    # Gaz et carbone : une valeur par jour local, comme dans le vrai jeu
    day = df["time_local"].dt.tz_localize(None).dt.normalize()
    for col in ("ttf_d2", "eua_d2"):
        daily = pd.Series(rng.normal(50, 10, day.nunique()), index=day.unique())
        df[col] = daily.reindex(day.to_numpy()).to_numpy()
    return df


# ------------------------------------------------------------------ 1. le registre est complet et sûr
def test_every_feature_is_registered():
    missing = [f for f in ALL_FEATURES if f not in FEATURE_REGISTRY]
    assert not missing, f"Features sans entrée dans FEATURE_REGISTRY : {missing}"


def test_registered_sources_exist():
    unknown = {c for cols, _ in FEATURE_REGISTRY.values() for c in cols if c not in RAW_PUBLICATION}
    assert not unknown, f"Colonnes sans heure de publication dans RAW_PUBLICATION : {unknown}"


@pytest.mark.parametrize("feature", ALL_FEATURES)
def test_declared_feature_is_known_at_gate_closure(feature):
    cols, lag_days = FEATURE_REGISTRY[feature]
    late = [c for c in cols if not is_known_at_gate_closure(c, lag_days)]
    assert not late, f"{feature} utilise {late} (J-{lag_days}), publié(s) après 12h en J-1"


# ------------------------------------------------------------------ 2. le code respecte le registre
DELIVERY_DAYS = ["2024-03-29",   # vendredi saint
                 "2024-03-31",   # passage à l'heure d'été (23 h)
                 "2024-04-01",   # lundi de Pâques, lendemain du changement d'heure
                 "2024-04-08"]   # J-7 tombe sur le changement d'heure


@pytest.mark.parametrize("delivery_day", DELIVERY_DAYS)
def test_features_unchanged_when_future_is_erased(synthetic, delivery_day):
    """Effacer tout ce qui est publié après la fermeture ne doit rien changer aux features de J."""
    full = add_features(synthetic)
    seen = add_features(mask_unpublished(synthetic, delivery_day))
    on_day = (full["time_local"].dt.strftime("%Y-%m-%d") == delivery_day).to_numpy()
    assert on_day.sum() in (23, 24, 25)
    pdt.assert_frame_equal(full.loc[on_day, ALL_FEATURES], seen.loc[on_day, ALL_FEATURES],
                           check_dtype=False)
    for step in STEPS:  # y compris les features normalisées par le SRMC
        pdt.assert_frame_equal(get_X(full, step)[on_day], get_X(seen, step)[on_day], check_dtype=False)


def test_masking_catches_a_leaky_feature(synthetic):
    """Contrôle du test lui-même : une feature qui lit le prix de l'heure précédente doit être détectée."""
    day = "2024-04-08"
    leaky_full = synthetic["price_da"].shift(1)
    leaky_seen = mask_unpublished(synthetic, day)["price_da"].shift(1)
    on_day = (synthetic["time_local"].dt.strftime("%Y-%m-%d") == day).to_numpy()
    assert not leaky_full[on_day].equals(leaky_seen[on_day])


# ------------------------------------------------------------------ 3. lags à heure locale fixe
def test_lags_follow_local_hour_across_dst(synthetic):
    f = add_features(synthetic).set_index("time_local")
    price = synthetic.set_index("time_local")["price_da"]
    ts = lambda s: pd.Timestamp(s, tz=TZ)
    # Lundi 01/04 à 3h (heure d'été) -> dimanche 31/03 à 3h (heure d'été) : 23 h d'écart, pas 24
    assert f.loc[ts("2024-04-01 03:00"), "price_d1"] == price[ts("2024-03-31 03:00")]
    # 2h n'existe pas le 31/03 : pas de lag J-1 à 2h le 01/04
    assert np.isnan(f.loc[ts("2024-04-01 02:00"), "price_d1"])
    # J-7 traverse le changement d'heure
    assert f.loc[ts("2024-04-05 18:00"), "price_d7"] == price[ts("2024-03-29 18:00")]
    # Moyenne de J-1 = moyenne de la journée locale précédente (23 h le 31/03)
    expected = price[(price.index >= ts("2024-03-31")) & (price.index < ts("2024-04-01"))].mean()
    assert f.loc[ts("2024-04-01 12:00"), "price_d1_mean"] == pytest.approx(expected)


def test_error_correction_uses_only_past_prices():
    """La correction du jour J ne doit dépendre ni du prix de J ni d'aucun jour suivant."""
    from src.correction import error_correction
    rng = np.random.default_rng(1)
    idx = pd.date_range("2024-01-01", "2024-06-30", freq="h", tz="UTC")
    t = pd.Series(idx.tz_convert(TZ), index=idx)
    price = pd.Series(rng.normal(80, 20, len(idx)), index=idx)
    base = (price + rng.normal(5, 10, len(idx))).rename("base")
    start, end, day = "2024-05-01", "2024-06-01", "2024-05-15"
    full = error_correction(base, price, t, start, end)
    after = (t >= pd.Timestamp(day, tz=TZ)).to_numpy()
    price2 = price.copy()
    price2[after] = rng.normal(500, 100, after.sum())     # prix de J et suivants bouleversés
    base2 = base.copy()
    later = (t >= pd.Timestamp(day, tz=TZ) + pd.Timedelta(days=1)).to_numpy()
    base2[later] = 0.0                                      # prévisions des jours suivants aussi
    seen = error_correction(base2, price2, t, start, end)
    on_day = (full.index.tz_convert(TZ).strftime("%Y-%m-%d") == day)
    pdt.assert_series_equal(full[on_day], seen[on_day])
    assert not full.equals(seen)                            # le test a bien du mordant


def test_calendar_flags():
    idx = pd.date_range("2024-05-08", "2024-05-12", freq="h", tz=TZ)
    df = pd.DataFrame({c: 1.0 for c in RAW_PUBLICATION}, index=idx)
    df["time_local"] = idx
    f = add_features(df).groupby(df["time_local"].dt.strftime("%Y-%m-%d"))[
        ["is_holiday", "is_bridge"]].first()
    assert f.loc["2024-05-09", "is_holiday"] == 1   # Ascension (jeudi)
    assert f.loc["2024-05-10", "is_bridge"] == 1    # vendredi de pont
    assert f.loc["2024-05-08", "is_holiday"] == 1   # 8 mai
    assert f.loc["2024-05-10", "is_holiday"] == 0
