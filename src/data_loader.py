from pathlib import Path
import pandas as pd
import numpy as np

from config import MarketConfig


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"


def ensure_directories():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_fred_series(series_id: str) -> pd.Series:
    """
    Scarica una serie da FRED e la restituisce come pandas Series.
    Versione robusta: gestisce DATE o OBSERVATION_DATE.
    """
    url = FRED_URL.format(series_id=series_id)
    df = pd.read_csv(url)

    # Pulizia nomi colonne
    df.columns = [str(col).strip().upper() for col in df.columns]

    print(f"\nLoading series: {series_id}")
    print("Columns found:", list(df.columns))

    # Colonna data: FRED può usare DATE oppure OBSERVATION_DATE
    if "DATE" in df.columns:
        date_col = "DATE"
    elif "OBSERVATION_DATE" in df.columns:
        date_col = "OBSERVATION_DATE"
    else:
        raise ValueError(
            f"La serie {series_id} non contiene una colonna data valida. "
            f"Colonne trovate: {list(df.columns)}"
        )

    # Colonna valori
    value_col = series_id.upper()
    if value_col not in df.columns:
        possible_value_cols = [c for c in df.columns if c != date_col]
        if len(possible_value_cols) == 1:
            value_col = possible_value_cols[0]
        else:
            raise ValueError(
                f"Non riesco a identificare la colonna valori per {series_id}. "
                f"Colonne trovate: {list(df.columns)}"
            )

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    series = df.set_index(date_col)[value_col].sort_index().dropna()
    series.name = series_id
    return series


def save_raw_series(series: pd.Series, filename: str):
    """
    Salva la serie raw in CSV.
    """
    out = series.reset_index()
    out.columns = ["date", "value"]
    out.to_csv(RAW_DIR / filename, index=False)


def build_factor_dataset(cfg: MarketConfig) -> pd.DataFrame:
    """
    Costruisce il dataset mensile dei fattori di rischio:
    - fx: EUR per USD
    - copper: USD per ton
    - eur_rate: decimal
    - usd_rate: decimal

    E aggiunge:
    - log returns per FX e copper
    - differenze per i tassi
    """
    ensure_directories()

    # 1. Scarico serie raw
    fx_usd_per_eur = load_fred_series(cfg.fx_series_id)          # USD per EUR
    copper_usd_ton = load_fred_series(cfg.copper_series_id)      # USD per ton
    eur_rate_pct = load_fred_series(cfg.eur_rate_series_id)      # %
    usd_rate_pct = load_fred_series(cfg.usd_rate_series_id)      # %

    # 2. Salvo raw
    save_raw_series(fx_usd_per_eur, "fx_usd_per_eur_raw.csv")
    save_raw_series(copper_usd_ton, "copper_usd_ton_raw.csv")
    save_raw_series(eur_rate_pct, "eur_rate_pct_raw.csv")
    save_raw_series(usd_rate_pct, "usd_rate_pct_raw.csv")

    # 3. Unisco tutto in un unico DataFrame
    df = pd.concat(
        [fx_usd_per_eur, copper_usd_ton, eur_rate_pct, usd_rate_pct],
        axis=1
    )
    df.columns = ["usd_per_eur", "copper_usd_ton", "eur_rate_pct", "usd_rate_pct"]

    # 4. Porto tutto a frequenza mensile (ultimo dato del mese)
    monthly = df.resample("ME").last().ffill()

    # 5. Tengo solo la finestra storica desiderata
    end_date = monthly.index.max()
    start_date = end_date - pd.DateOffset(years=cfg.estimation_window_years)
    monthly = monthly.loc[monthly.index >= start_date].copy()

    # 6. Converto il cambio da USD per EUR a EUR per USD
    monthly["fx_eur_per_usd"] = 1.0 / monthly["usd_per_eur"]

    # 7. Converto i tassi in decimal
    monthly["eur_rate"] = monthly["eur_rate_pct"] / 100.0
    monthly["usd_rate"] = monthly["usd_rate_pct"] / 100.0

    # 8. Creo colonne finali pulite
    factors = monthly[["fx_eur_per_usd", "copper_usd_ton", "eur_rate", "usd_rate"]].copy()
    factors.columns = ["fx", "copper", "eur_rate", "usd_rate"]

    # 9. Variabili da usare nelle simulazioni
    factors["fx_logret"] = np.log(factors["fx"] / factors["fx"].shift(1))
    factors["copper_logret"] = np.log(factors["copper"] / factors["copper"].shift(1))
    factors["eur_rate_diff"] = factors["eur_rate"].diff()
    factors["usd_rate_diff"] = factors["usd_rate"].diff()

    factors = factors.dropna().copy()

    # 10. Salvo dataset finale
    factors.to_csv(PROCESSED_DIR / "factors_monthly.csv", index=True)

    return factors


def main():
    cfg = MarketConfig()
    factors = build_factor_dataset(cfg)

    print("=== DATA LOADER CHECK ===")
    print(f"Rows: {len(factors)}")
    print(f"Columns: {list(factors.columns)}")
    print("\nLast 5 rows:")
    print(factors.tail())

    print("\nSaved files:")
    print(f"- Raw data folder: {RAW_DIR}")
    print(f"- Processed dataset: {PROCESSED_DIR / 'factors_monthly.csv'}")


if __name__ == "__main__":
    main()