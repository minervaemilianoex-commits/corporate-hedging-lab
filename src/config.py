from dataclasses import dataclass


@dataclass
class CompanyConfig:
    # Flussi operativi su orizzonte 3 mesi
    rev_eur: float = 9_750_000
    rev_usd: float = 4_125_000
    cost_eur: float = 7_250_000
    cost_usd: float = 1_500_000

    # Commodity exposure: quantità fisica equivalente
    commodity_qty_ton: float = 222.222222

    # Stock di debito
    debt_notional_eur: float = 22_000_000
    debt_spread: float = 0.015  # 1.5%

    # Orizzonte del rischio
    horizon_years: float = 0.25  # 3 mesi


@dataclass
class MarketConfig:
    # Convenzioni
    fx_quote: str = "EUR per USD"

    # Livelli iniziali
    fx_spot: float = 0.92
    copper_price_usd_ton: float = 9_000
    eur_rate: float = 0.025
    usd_rate: float = 0.045

    # Serie FRED
    fx_series_id: str = "DEXUSEU"
    copper_series_id: str = "PCOPPUSDM"
    eur_rate_series_id: str = "IR3TIB01EZM156N"
    usd_rate_series_id: str = "SOFR"

    # Finestra storica
    estimation_window_years: int = 5

    # Simulazione
    confidence: float = 0.95
    n_sims: int = 20_000
    seed: int = 42


@dataclass
class HedgeConfig:
    fx_hedge_ratio: float = 0.70
    commodity_hedge_ratio: float = 0.60
    rate_hedge_ratio: float = 0.75

    collar_put_pct: float = 0.97
    collar_call_pct: float = 1.05

    cap_strike_addon: float = 0.005    # 50 bps
    swap_fixed_addon: float = 0.0025   # 25 bps