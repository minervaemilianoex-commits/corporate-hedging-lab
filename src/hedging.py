import numpy as np
import pandas as pd
from scipy.stats import norm

from config import CompanyConfig, HedgeConfig


def fx_forward_rate(spot_fx: float, r_eur: float, r_usd: float, horizon_years: float) -> float:
    """
    Forward FX teorico con convenzione FX = EUR per USD.
    """
    return spot_fx * np.exp((r_eur - r_usd) * horizon_years)


def commodity_forward_price(spot_commodity: float, r_usd: float, horizon_years: float) -> float:
    """
    Forward commodity teorico molto semplificato.

    Assunzione:
    - commodity quotata in USD
    - nessun costo di storage / convenience yield esplicito
    """
    return spot_commodity * np.exp(r_usd * horizon_years)


def swap_fixed_rate(spot_eur_rate: float, swap_fixed_addon: float) -> float:
    """
    Tasso fisso swap semplificato:
    tasso EUR corrente + piccolo add-on.
    """
    return spot_eur_rate + swap_fixed_addon


def rate_cap_strike(spot_eur_rate: float, cap_strike_addon: float) -> float:
    """
    Strike del cap:
    tasso EUR corrente + add-on.
    """
    return spot_eur_rate + cap_strike_addon


def bachelier_caplet_price(
    forward_rate: float,
    strike: float,
    normal_vol: float,
    horizon_years: float,
    discount_rate: float,
    notional: float
) -> float:
    """
    Prezzo semplificato di un caplet con formula di Bachelier.

    È una scelta coerente con il fatto che nel modello stimiamo
    la volatilità come std delle differenze dei tassi (normal vol),
    non come log-volatilità.

    Formula:
    price = DF * N * T * [ (F-K) * Phi(d) + sigma * sqrt(T) * phi(d) ]
    con d = (F-K) / (sigma * sqrt(T))
    """
    T = horizon_years
    sigma_n = max(normal_vol, 1e-8)

    discount_factor = np.exp(-discount_rate * T)
    sigma_root_t = sigma_n * np.sqrt(T)

    if sigma_root_t <= 1e-12:
        intrinsic = max(forward_rate - strike, 0.0)
        return discount_factor * notional * T * intrinsic

    d = (forward_rate - strike) / sigma_root_t

    price = (
        discount_factor
        * notional
        * T
        * ((forward_rate - strike) * norm.cdf(d) + sigma_root_t * norm.pdf(d))
    )

    return max(price, 0.0)


def expected_net_usd_exposure(company_cfg: CompanyConfig, market_state: dict) -> float:
    """
    Calcola l'esposizione netta attesa in USD.

    L'esposizione è definita come:
    ricavi USD - costi USD - costo commodity in USD.

    Nota importante:
    il segno dell'esposizione può cambiare in base ai prezzi correnti della commodity.
    - se il valore è positivo, l'azienda è net long USD
    - se il valore è negativo, l'azienda è net short USD
    """
    commodity_cost_usd = company_cfg.commodity_qty_ton * market_state["copper"]

    net_usd = (
        company_cfg.rev_usd
        - company_cfg.cost_usd
        - commodity_cost_usd
    )
    return net_usd


def apply_fx_forward_hedge(
    simulated_df: pd.DataFrame,
    company_cfg: CompanyConfig,
    hedge_cfg: HedgeConfig,
    market_state: dict
) -> tuple[pd.DataFrame, dict]:
    """
    Applica una copertura FX forward su una quota dell'esposizione netta attesa in USD.

    Logica del modello:
    - se l'esposizione netta USD è positiva, l'azienda è net long USD
      e la copertura equivale economicamente a vendere USD forward
    - se l'esposizione netta USD è negativa, l'azienda è net short USD
      e la copertura equivale economicamente a comprare USD forward

    Nel codice questo è gestito automaticamente dal segno del notional:
    - hedge_notional_usd > 0  -> posizione corta in USD forward
    - hedge_notional_usd < 0  -> posizione lunga in USD forward

    Convenzione payoff in EUR:
    payoff = hedge_notional_usd * (F - S_T)
    """
    out = simulated_df.copy()

    net_usd = expected_net_usd_exposure(company_cfg, market_state)

    # Il segno del notional incorpora automaticamente il verso della copertura:
    # positivo = vendita USD forward
    # negativo = acquisto USD forward
    hedge_notional_usd = hedge_cfg.fx_hedge_ratio * net_usd

    forward_rate = fx_forward_rate(
        spot_fx=market_state["fx"],
        r_eur=market_state["eur_rate"],
        r_usd=market_state["usd_rate"],
        horizon_years=company_cfg.horizon_years,
    )

    out["fx_forward_payoff"] = hedge_notional_usd * (forward_rate - out["fx"])
    out["ebit_fx_forward"] = out["ebit"] + out["fx_forward_payoff"]
    out["cash_flow_fx_forward"] = out["cash_flow"] + out["fx_forward_payoff"]

    # Payoff del forward valutato nel base case corrente
    base_payoff = hedge_notional_usd * (forward_rate - market_state["fx"])

    info = {
        "expected_net_usd": net_usd,
        "hedge_notional_usd": hedge_notional_usd,
        "forward_rate": forward_rate,
        "base_payoff_eur": base_payoff,
    }

    return out, info

def apply_fx_collar_hedge(
    simulated_df: pd.DataFrame,
    company_cfg: CompanyConfig,
    hedge_cfg: HedgeConfig,
    market_state: dict
) -> tuple[pd.DataFrame, dict]:
    """
    Applica una copertura FX collar su una quota dell'esposizione netta attesa in USD.

    Logica:
    - il collar impone un tasso minimo e massimo sul cambio della quota hedgiata
    - con convenzione FX = EUR per USD, il cambio effettivo sulla quota coperta diventa:
      clamp(S_T, put_strike, call_strike)

    Questo vale sia se l'azienda è net long USD sia se è net short USD:
    il segno del notional gestisce automaticamente il verso economico della copertura.

    Assunzione semplificativa:
    - collar trattato come zero-cost, quindi senza premio netto iniziale
    """
    out = simulated_df.copy()

    net_usd = expected_net_usd_exposure(company_cfg, market_state)
    hedge_notional_usd = hedge_cfg.fx_hedge_ratio * net_usd

    put_strike = hedge_cfg.collar_put_pct * market_state["fx"]
    call_strike = hedge_cfg.collar_call_pct * market_state["fx"]

    out["fx_collar_effective_fx"] = out["fx"].clip(lower=put_strike, upper=call_strike)

    out["fx_collar_payoff"] = (
        hedge_notional_usd
        * (out["fx_collar_effective_fx"] - out["fx"])
    )

    out["ebit_fx_collar"] = out["ebit"] + out["fx_collar_payoff"]
    out["cash_flow_fx_collar"] = out["cash_flow"] + out["fx_collar_payoff"]

    base_effective_fx = min(max(market_state["fx"], put_strike), call_strike)
    base_payoff = hedge_notional_usd * (base_effective_fx - market_state["fx"])

    info = {
        "expected_net_usd": net_usd,
        "hedge_notional_usd": hedge_notional_usd,
        "put_strike": put_strike,
        "call_strike": call_strike,
        "base_payoff_eur": base_payoff,
    }

    return out, info

def apply_commodity_forward_hedge(
    simulated_df: pd.DataFrame,
    company_cfg: CompanyConfig,
    hedge_cfg: HedgeConfig,
    market_state: dict
) -> tuple[pd.DataFrame, dict]:
    """
    Applica una copertura commodity forward su una quota del fabbisogno fisico di rame.

    Logica economica:
    - l'azienda compra commodity, quindi soffre se il prezzo del rame sale
    - la copertura equivale a comprare forward una quota della quantità attesa
    - payoff in USD = hedge_qty_ton * (S_T - F)

    Il payoff viene poi convertito in EUR usando il cambio simulato.
    """
    out = simulated_df.copy()

    hedge_qty_ton = hedge_cfg.commodity_hedge_ratio * company_cfg.commodity_qty_ton

    forward_price = commodity_forward_price(
        spot_commodity=market_state["copper"],
        r_usd=market_state["usd_rate"],
        horizon_years=company_cfg.horizon_years,
    )

    out["commodity_forward_payoff_usd"] = hedge_qty_ton * (out["copper"] - forward_price)
    out["commodity_forward_payoff"] = out["commodity_forward_payoff_usd"] * out["fx"]

    out["ebit_commodity_forward"] = out["ebit"] + out["commodity_forward_payoff"]
    out["cash_flow_commodity_forward"] = out["cash_flow"] + out["commodity_forward_payoff"]

    base_payoff_usd = hedge_qty_ton * (market_state["copper"] - forward_price)
    base_payoff_eur = base_payoff_usd * market_state["fx"]

    info = {
        "hedge_qty_ton": hedge_qty_ton,
        "forward_price_usd_ton": forward_price,
        "base_payoff_usd": base_payoff_usd,
        "base_payoff_eur": base_payoff_eur,
    }

    return out, info


def apply_rate_swap_hedge(
    simulated_df: pd.DataFrame,
    company_cfg: CompanyConfig,
    hedge_cfg: HedgeConfig,
    market_state: dict
) -> tuple[pd.DataFrame, dict]:
    """
    Applica una copertura via interest rate swap su una quota del debito floating.

    Logica:
    - l'azienda paga tasso variabile sul debito
    - con lo swap paga fisso e riceve floating sulla quota hedgiata
    - payoff = notional_hedged * T * (r_sim - r_fixed)

    Se il tasso simulato sale sopra il fisso, il payoff è positivo e compensa
    il maggior costo del debito floating.

    Nota:
    lo swap impatta il cash flow, non l'EBIT.
    """
    out = simulated_df.copy()

    hedge_notional_eur = hedge_cfg.rate_hedge_ratio * company_cfg.debt_notional_eur
    fixed_rate = swap_fixed_rate(
        spot_eur_rate=market_state["eur_rate"],
        swap_fixed_addon=hedge_cfg.swap_fixed_addon,
    )

    out["rate_swap_payoff"] = (
        hedge_notional_eur
        * company_cfg.horizon_years
        * (out["eur_rate"] - fixed_rate)
    )

    out["ebit_rate_swap"] = out["ebit"]
    out["cash_flow_rate_swap"] = out["cash_flow"] + out["rate_swap_payoff"]

    base_payoff = (
        hedge_notional_eur
        * company_cfg.horizon_years
        * (market_state["eur_rate"] - fixed_rate)
    )

    info = {
        "hedge_notional_eur": hedge_notional_eur,
        "fixed_rate": fixed_rate,
        "base_payoff_eur": base_payoff,
    }

    return out, info


def apply_rate_cap_hedge(
    simulated_df: pd.DataFrame,
    company_cfg: CompanyConfig,
    hedge_cfg: HedgeConfig,
    market_state: dict,
    normal_vol: float
) -> tuple[pd.DataFrame, dict]:
    """
    Applica una copertura via rate cap su una quota del debito floating.

    Logica:
    - l'azienda resta esposta al floating rate
    - riceve protezione solo se il tasso supera lo strike del cap
    - in cambio paga un premio iniziale

    Payoff del cap:
    payoff = notional_hedged * T * max(r_sim - strike, 0)

    Il premio viene trattato come costo sul cash flow iniziale del base case
    e su ogni scenario simulato.
    """
    out = simulated_df.copy()

    hedge_notional_eur = hedge_cfg.rate_hedge_ratio * company_cfg.debt_notional_eur
    strike_rate = rate_cap_strike(
        spot_eur_rate=market_state["eur_rate"],
        cap_strike_addon=hedge_cfg.cap_strike_addon,
    )

    premium_eur = bachelier_caplet_price(
        forward_rate=market_state["eur_rate"],
        strike=strike_rate,
        normal_vol=normal_vol,
        horizon_years=company_cfg.horizon_years,
        discount_rate=market_state["eur_rate"],
        notional=hedge_notional_eur,
    )

    out["rate_cap_payoff"] = (
        hedge_notional_eur
        * company_cfg.horizon_years
        * np.maximum(out["eur_rate"] - strike_rate, 0.0)
    )

    out["ebit_rate_cap"] = out["ebit"]
    out["cash_flow_rate_cap"] = out["cash_flow"] + out["rate_cap_payoff"] - premium_eur

    base_payoff = (
        hedge_notional_eur
        * company_cfg.horizon_years
        * max(market_state["eur_rate"] - strike_rate, 0.0)
        - premium_eur
    )

    info = {
        "hedge_notional_eur": hedge_notional_eur,
        "strike_rate": strike_rate,
        "normal_vol": normal_vol,
        "premium_eur": premium_eur,
        "base_payoff_eur": base_payoff,
    }

    return out, info

def apply_layered_hedge(
    simulated_df: pd.DataFrame,
    company_cfg: CompanyConfig,
    hedge_cfg: HedgeConfig,
    market_state: dict
) -> tuple[pd.DataFrame, dict]:
    """
    Applica una layered hedge completa composta da:
    - FX forward
    - commodity forward
    - rate swap

    Nota importante:
    - FX e commodity impattano EBIT e cash flow
    - il rate swap impatta solo il cash flow
    """
    out = simulated_df.copy()

    out, fx_info = apply_fx_forward_hedge(
        simulated_df=out,
        company_cfg=company_cfg,
        hedge_cfg=hedge_cfg,
        market_state=market_state,
    )

    out, commodity_info = apply_commodity_forward_hedge(
        simulated_df=out,
        company_cfg=company_cfg,
        hedge_cfg=hedge_cfg,
        market_state=market_state,
    )

    out, rate_info = apply_rate_swap_hedge(
        simulated_df=out,
        company_cfg=company_cfg,
        hedge_cfg=hedge_cfg,
        market_state=market_state,
    )

    # Payoff operativo: FX + commodity
    out["layered_operating_payoff"] = (
        out["fx_forward_payoff"] + out["commodity_forward_payoff"]
    )

    # Payoff totale cash flow: FX + commodity + rate swap
    out["layered_total_payoff"] = (
        out["fx_forward_payoff"]
        + out["commodity_forward_payoff"]
        + out["rate_swap_payoff"]
    )

    out["ebit_layered"] = out["ebit"] + out["layered_operating_payoff"]
    out["cash_flow_layered"] = out["cash_flow"] + out["layered_total_payoff"]

    layered_base_ebit_payoff_eur = (
        fx_info["base_payoff_eur"] + commodity_info["base_payoff_eur"]
    )

    layered_base_cash_flow_payoff_eur = (
        fx_info["base_payoff_eur"]
        + commodity_info["base_payoff_eur"]
        + rate_info["base_payoff_eur"]
    )

    info = {
        "fx_base_payoff_eur": fx_info["base_payoff_eur"],
        "commodity_base_payoff_eur": commodity_info["base_payoff_eur"],
        "rate_base_payoff_eur": rate_info["base_payoff_eur"],
        "layered_base_ebit_payoff_eur": layered_base_ebit_payoff_eur,
        "layered_base_cash_flow_payoff_eur": layered_base_cash_flow_payoff_eur,
    }

    return out, info

def apply_optional_layered_hedge(
    simulated_df: pd.DataFrame,
    company_cfg: CompanyConfig,
    hedge_cfg: HedgeConfig,
    market_state: dict,
    normal_vol: float
) -> tuple[pd.DataFrame, dict]:
    """
    Applica una layered hedge opzionale composta da:
    - FX collar
    - commodity forward
    - rate cap

    Logica:
    - FX collar e commodity forward impattano EBIT e cash flow
    - il rate cap impatta solo il cash flow
    - il premio del cap viene sottratto in ogni scenario simulato
    """
    out = simulated_df.copy()

    out, fx_collar_info = apply_fx_collar_hedge(
        simulated_df=out,
        company_cfg=company_cfg,
        hedge_cfg=hedge_cfg,
        market_state=market_state,
    )

    out, commodity_info = apply_commodity_forward_hedge(
        simulated_df=out,
        company_cfg=company_cfg,
        hedge_cfg=hedge_cfg,
        market_state=market_state,
    )

    out, rate_cap_info = apply_rate_cap_hedge(
        simulated_df=out,
        company_cfg=company_cfg,
        hedge_cfg=hedge_cfg,
        market_state=market_state,
        normal_vol=normal_vol,
    )

    # Payoff operativo: FX collar + commodity forward
    out["optional_layered_operating_payoff"] = (
        out["fx_collar_payoff"] + out["commodity_forward_payoff"]
    )

    # Payoff totale di cash flow:
    # FX collar + commodity forward + rate cap payoff netto del premio
    out["optional_layered_total_payoff"] = (
        out["fx_collar_payoff"]
        + out["commodity_forward_payoff"]
        + out["rate_cap_payoff"]
        - rate_cap_info["premium_eur"]
    )

    out["ebit_optional_layered"] = (
        out["ebit"] + out["optional_layered_operating_payoff"]
    )

    out["cash_flow_optional_layered"] = (
        out["cash_flow"] + out["optional_layered_total_payoff"]
    )

    optional_layered_base_ebit_payoff_eur = (
        fx_collar_info["base_payoff_eur"] + commodity_info["base_payoff_eur"]
    )

    optional_layered_base_cash_flow_payoff_eur = (
        fx_collar_info["base_payoff_eur"]
        + commodity_info["base_payoff_eur"]
        + rate_cap_info["base_payoff_eur"]
    )

    info = {
        "fx_collar_base_payoff_eur": fx_collar_info["base_payoff_eur"],
        "commodity_base_payoff_eur": commodity_info["base_payoff_eur"],
        "rate_cap_base_payoff_eur": rate_cap_info["base_payoff_eur"],
        "rate_cap_premium_eur": rate_cap_info["premium_eur"],
        "optional_layered_base_ebit_payoff_eur": optional_layered_base_ebit_payoff_eur,
        "optional_layered_base_cash_flow_payoff_eur": optional_layered_base_cash_flow_payoff_eur,
    }

    return out, info

   