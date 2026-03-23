from pathlib import Path
import numpy as np
import pandas as pd

from config import CompanyConfig, MarketConfig, HedgeConfig
from config import CompanyConfig, MarketConfig, HedgeConfig
from hedging import (
    apply_fx_forward_hedge,
    apply_fx_collar_hedge,
    apply_commodity_forward_hedge,
    apply_rate_swap_hedge,
    apply_rate_cap_hedge,
    apply_layered_hedge,
    apply_optional_layered_hedge,
)


BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"


def load_factor_dataset() -> pd.DataFrame:
    """
    Carica il dataset dei fattori già processato.
    """
    path = PROCESSED_DIR / "factors_monthly.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df


def get_latest_market_state(factors: pd.DataFrame) -> dict:
    """
    Estrae l'ultimo stato osservato dei fattori di mercato.
    """
    last = factors.iloc[-1]
    return {
        "fx": float(last["fx"]),
        "copper": float(last["copper"]),
        "eur_rate": float(last["eur_rate"]),
        "usd_rate": float(last["usd_rate"]),
    }


def simulate_market_states(
    factors: pd.DataFrame,
    market_cfg: MarketConfig,
    company_cfg: CompanyConfig,
    latest_state: dict
) -> pd.DataFrame:
    """
    Simula i fattori di rischio su orizzonte 3 mesi usando:
    - log returns per FX e copper
    - differenze semplici per i tassi
    """
    np.random.seed(market_cfg.seed)

    shock_cols = ["fx_logret", "copper_logret", "eur_rate_diff", "usd_rate_diff"]
    hist = factors[shock_cols].copy()

    mu = hist.mean().values
    cov = hist.cov().values

    months = int(round(company_cfg.horizon_years * 12)) # 3 mesi
    mu_h = mu * months
    cov_h = cov * months

    sims = np.random.multivariate_normal(mu_h, cov_h, size=market_cfg.n_sims)
    sim_df = pd.DataFrame(sims, columns=shock_cols)

    # Ricostruzione livelli finali
    sim_df["fx"] = latest_state["fx"] * np.exp(sim_df["fx_logret"])
    sim_df["copper"] = latest_state["copper"] * np.exp(sim_df["copper_logret"])

    sim_df["eur_rate"] = latest_state["eur_rate"] + sim_df["eur_rate_diff"]
    sim_df["usd_rate"] = latest_state["usd_rate"] + sim_df["usd_rate_diff"]

    # Limiti ragionevoli per evitare valori estremi
    sim_df["eur_rate"] = sim_df["eur_rate"].clip(lower=-0.02, upper=0.20)
    sim_df["usd_rate"] = sim_df["usd_rate"].clip(lower=-0.02, upper=0.20)

    return sim_df


def compute_base_case(
    company_cfg: CompanyConfig,
    market_state: dict
) -> dict:
    """
    Calcola EBIT e cash flow del base case.
    """
    commodity_cost_usd = company_cfg.commodity_qty_ton * market_state["copper"]

    revenue_total_eur = (
        company_cfg.rev_eur
        + company_cfg.rev_usd * market_state["fx"]
    )

    operating_costs_total_eur = (
        company_cfg.cost_eur
        + company_cfg.cost_usd * market_state["fx"]
        + commodity_cost_usd * market_state["fx"]
    )

    ebit = revenue_total_eur - operating_costs_total_eur

    interest_cost = (
        company_cfg.debt_notional_eur
        * (market_state["eur_rate"] + company_cfg.debt_spread)
        * company_cfg.horizon_years
    )

    cash_flow = ebit - interest_cost

    return {
        "commodity_cost_usd": commodity_cost_usd,
        "revenue_total_eur": revenue_total_eur,
        "operating_costs_total_eur": operating_costs_total_eur,
        "ebit": ebit,
        "interest_cost": interest_cost,
        "cash_flow": cash_flow,
    }


def compute_simulated_pnl(
    sim_df: pd.DataFrame,
    company_cfg: CompanyConfig
) -> pd.DataFrame:
    """
    Ricostruisce EBIT e cash flow per ogni scenario simulato.
    """
    out = sim_df.copy()

    out["commodity_cost_usd"] = company_cfg.commodity_qty_ton * out["copper"]

    out["revenue_total_eur"] = (
        company_cfg.rev_eur
        + company_cfg.rev_usd * out["fx"]
    )

    out["operating_costs_total_eur"] = (
        company_cfg.cost_eur
        + company_cfg.cost_usd * out["fx"]
        + out["commodity_cost_usd"] * out["fx"]
    )

    out["ebit"] = out["revenue_total_eur"] - out["operating_costs_total_eur"]

    out["interest_cost"] = (
        company_cfg.debt_notional_eur
        * (out["eur_rate"] + company_cfg.debt_spread)
        * company_cfg.horizon_years
    )

    out["cash_flow"] = out["ebit"] - out["interest_cost"]

    return out


def compute_risk_metrics(
    simulated_df: pd.DataFrame,
    base_case: dict,
    confidence: float
) -> dict:
    """
    Calcola EaR, CFaR e VaR sul delta cash flow.
    """
    alpha = 1.0 - confidence

    ebit_q = simulated_df["ebit"].quantile(alpha)
    cf_q = simulated_df["cash_flow"].quantile(alpha)

    delta_cf = simulated_df["cash_flow"] - base_case["cash_flow"]
    delta_cf_q = delta_cf.quantile(alpha)

    metrics = {
        "base_ebit": base_case["ebit"],
        "base_cash_flow": base_case["cash_flow"],
        "mean_ebit": simulated_df["ebit"].mean(),
        "mean_cash_flow": simulated_df["cash_flow"].mean(),
        "std_cash_flow": simulated_df["cash_flow"].std(),
        "ear_95": base_case["ebit"] - ebit_q,
        "cfar_95": base_case["cash_flow"] - cf_q,
        "var_delta_cf_95": -delta_cf_q,
        "p5_ebit": ebit_q,
        "p5_cash_flow": cf_q,
    }

    return metrics


def main():
    company_cfg = CompanyConfig()
    market_cfg = MarketConfig()
    hedge_cfg = HedgeConfig()

    factors = load_factor_dataset()
    latest_state = get_latest_market_state(factors)

    # Volatilità normale del tasso EUR, annualizzata
    rate_normal_vol = factors["eur_rate_diff"].std() * np.sqrt(12)

    # =========================
    # UNHEDGED
    # =========================
    base_case = compute_base_case(company_cfg, latest_state)
    sim_states = simulate_market_states(factors, market_cfg, company_cfg, latest_state)
    simulated_df = compute_simulated_pnl(sim_states, company_cfg)
    metrics = compute_risk_metrics(simulated_df, base_case, market_cfg.confidence)

    simulated_df.to_csv(PROCESSED_DIR / "simulated_unhedged.csv", index=False)

    # =========================
    # FX FORWARD
    # =========================
    fx_hedged_df, fx_info = apply_fx_forward_hedge(
        simulated_df=simulated_df,
        company_cfg=company_cfg,
        hedge_cfg=hedge_cfg,
        market_state=latest_state,
    )

    fx_base_case = base_case.copy()
    fx_base_case["ebit"] = base_case["ebit"] + fx_info["base_payoff_eur"]
    fx_base_case["cash_flow"] = base_case["cash_flow"] + fx_info["base_payoff_eur"]

    fx_eval_df = fx_hedged_df[["ebit_fx_forward", "cash_flow_fx_forward"]].rename(
        columns={
            "ebit_fx_forward": "ebit",
            "cash_flow_fx_forward": "cash_flow",
        }
    )

    fx_metrics = compute_risk_metrics(fx_eval_df, fx_base_case, market_cfg.confidence)
    fx_hedged_df.to_csv(PROCESSED_DIR / "simulated_fx_forward.csv", index=False)

    # =========================
    # FX COLLAR
    # =========================
    fx_collar_df, fx_collar_info = apply_fx_collar_hedge(
        simulated_df=simulated_df,
        company_cfg=company_cfg,
        hedge_cfg=hedge_cfg,
        market_state=latest_state,
    )

    fx_collar_base_case = base_case.copy()
    fx_collar_base_case["ebit"] = base_case["ebit"] + fx_collar_info["base_payoff_eur"]
    fx_collar_base_case["cash_flow"] = base_case["cash_flow"] + fx_collar_info["base_payoff_eur"]

    fx_collar_eval_df = fx_collar_df[["ebit_fx_collar", "cash_flow_fx_collar"]].rename(
        columns={
            "ebit_fx_collar": "ebit",
            "cash_flow_fx_collar": "cash_flow",
        }
    )

    fx_collar_metrics = compute_risk_metrics(
        fx_collar_eval_df, fx_collar_base_case, market_cfg.confidence
    )
    fx_collar_df.to_csv(PROCESSED_DIR / "simulated_fx_collar.csv", index=False)

    # =========================
    # COMMODITY FORWARD
    # =========================
    commodity_hedged_df, commodity_info = apply_commodity_forward_hedge(
        simulated_df=simulated_df,
        company_cfg=company_cfg,
        hedge_cfg=hedge_cfg,
        market_state=latest_state,
    )

    commodity_base_case = base_case.copy()
    commodity_base_case["ebit"] = base_case["ebit"] + commodity_info["base_payoff_eur"]
    commodity_base_case["cash_flow"] = base_case["cash_flow"] + commodity_info["base_payoff_eur"]

    commodity_eval_df = commodity_hedged_df[
        ["ebit_commodity_forward", "cash_flow_commodity_forward"]
    ].rename(
        columns={
            "ebit_commodity_forward": "ebit",
            "cash_flow_commodity_forward": "cash_flow",
        }
    )

    commodity_metrics = compute_risk_metrics(
        commodity_eval_df, commodity_base_case, market_cfg.confidence
    )
    commodity_hedged_df.to_csv(PROCESSED_DIR / "simulated_commodity_forward.csv", index=False)

    # =========================
    # RATE SWAP
    # =========================
    rate_hedged_df, rate_info = apply_rate_swap_hedge(
        simulated_df=simulated_df,
        company_cfg=company_cfg,
        hedge_cfg=hedge_cfg,
        market_state=latest_state,
    )

    rate_base_case = base_case.copy()
    rate_base_case["ebit"] = base_case["ebit"]
    rate_base_case["cash_flow"] = base_case["cash_flow"] + rate_info["base_payoff_eur"]

    rate_eval_df = rate_hedged_df[["ebit_rate_swap", "cash_flow_rate_swap"]].rename(
        columns={
            "ebit_rate_swap": "ebit",
            "cash_flow_rate_swap": "cash_flow",
        }
    )

    rate_metrics = compute_risk_metrics(
        rate_eval_df, rate_base_case, market_cfg.confidence
    )
    rate_hedged_df.to_csv(PROCESSED_DIR / "simulated_rate_swap.csv", index=False)

    # =========================
    # RATE CAP
    # =========================
    rate_cap_df, rate_cap_info = apply_rate_cap_hedge(
        simulated_df=simulated_df,
        company_cfg=company_cfg,
        hedge_cfg=hedge_cfg,
        market_state=latest_state,
        normal_vol=rate_normal_vol,
    )

    rate_cap_base_case = base_case.copy()
    rate_cap_base_case["ebit"] = base_case["ebit"]
    rate_cap_base_case["cash_flow"] = base_case["cash_flow"] + rate_cap_info["base_payoff_eur"]

    rate_cap_eval_df = rate_cap_df[["ebit_rate_cap", "cash_flow_rate_cap"]].rename(
        columns={
            "ebit_rate_cap": "ebit",
            "cash_flow_rate_cap": "cash_flow",
        }
    )

    rate_cap_metrics = compute_risk_metrics(
        rate_cap_eval_df, rate_cap_base_case, market_cfg.confidence
    )
    rate_cap_df.to_csv(PROCESSED_DIR / "simulated_rate_cap.csv", index=False)

    # =========================
    # LAYERED HEDGE COMPLETA (FX + COMMODITY + RATE SWAP)
    # =========================
    layered_df, layered_info = apply_layered_hedge(
        simulated_df=simulated_df,
        company_cfg=company_cfg,
        hedge_cfg=hedge_cfg,
        market_state=latest_state,
    )

    layered_base_case = base_case.copy()
    layered_base_case["ebit"] = (
        base_case["ebit"] + layered_info["layered_base_ebit_payoff_eur"]
    )
    layered_base_case["cash_flow"] = (
        base_case["cash_flow"] + layered_info["layered_base_cash_flow_payoff_eur"]
    )

    layered_eval_df = layered_df[["ebit_layered", "cash_flow_layered"]].rename(
        columns={
            "ebit_layered": "ebit",
            "cash_flow_layered": "cash_flow",
        }
    )

    layered_metrics = compute_risk_metrics(
        layered_eval_df, layered_base_case, market_cfg.confidence
    )
    layered_df.to_csv(PROCESSED_DIR / "simulated_layered.csv", index=False)

    # =========================
    # OPTIONAL LAYERED HEDGE (FX COLLAR + COMMODITY + RATE CAP)
    # =========================
    optional_layered_df, optional_layered_info = apply_optional_layered_hedge(
        simulated_df=simulated_df,
        company_cfg=company_cfg,
        hedge_cfg=hedge_cfg,
        market_state=latest_state,
        normal_vol=rate_normal_vol,
    )

    optional_layered_base_case = base_case.copy()
    optional_layered_base_case["ebit"] = (
        base_case["ebit"] + optional_layered_info["optional_layered_base_ebit_payoff_eur"]
    )
    optional_layered_base_case["cash_flow"] = (
        base_case["cash_flow"] + optional_layered_info["optional_layered_base_cash_flow_payoff_eur"]
    )

    optional_layered_eval_df = optional_layered_df[
        ["ebit_optional_layered", "cash_flow_optional_layered"]
    ].rename(
        columns={
            "ebit_optional_layered": "ebit",
            "cash_flow_optional_layered": "cash_flow",
        }
    )

    optional_layered_metrics = compute_risk_metrics(
        optional_layered_eval_df,
        optional_layered_base_case,
        market_cfg.confidence,
    )

    optional_layered_df.to_csv(
        PROCESSED_DIR / "simulated_optional_layered.csv",
        index=False,
    )

    # =========================
    # OUTPUT
    # =========================
    print("=== RISK ENGINE CHECK ===")

    print("\nLatest market state:")
    for k, v in latest_state.items():
        print(f"{k}: {v:,.6f}")

    print("\nBase case:")
    for k, v in base_case.items():
        if "usd" in k:
            print(f"{k}: {v:,.0f} USD")
        else:
            print(f"{k}: {v:,.0f} EUR")

    print("\nRisk metrics - Unhedged:")
    for k, v in metrics.items():
        print(f"{k}: {v:,.2f}")

    print("\nFX forward hedge info:")
    for k, v in fx_info.items():
        if "rate" in k:
            print(f"{k}: {v:,.6f}")
        elif "usd" in k:
            print(f"{k}: {v:,.0f} USD")
        else:
            print(f"{k}: {v:,.2f}")

    print("\nRisk metrics - FX forward:")
    for k, v in fx_metrics.items():
        print(f"{k}: {v:,.2f}")
    
    print("\nFX collar hedge info:")
    for k, v in fx_collar_info.items():
        if "usd" in k:
            print(f"{k}: {v:,.0f} USD")
        else:
            print(f"{k}: {v:,.6f}")

    print("\nRisk metrics - FX collar:")
    for k, v in fx_collar_metrics.items():
        print(f"{k}: {v:,.2f}")

    print("\nCommodity forward hedge info:")
    for k, v in commodity_info.items():
        if "ton" in k:
            print(f"{k}: {v:,.4f}")
        elif "usd" in k:
            print(f"{k}: {v:,.2f} USD")
        else:
            print(f"{k}: {v:,.2f}")

    print("\nRisk metrics - Commodity forward:")
    for k, v in commodity_metrics.items():
        print(f"{k}: {v:,.2f}")

    print("\nRate swap hedge info:")
    for k, v in rate_info.items():
        if "rate" in k:
            print(f"{k}: {v:,.6f}")
        elif "eur" in k:
            print(f"{k}: {v:,.2f} EUR")
        else:
            print(f"{k}: {v:,.2f}")

    print("\nRisk metrics - Rate swap:")
    for k, v in rate_metrics.items():
        print(f"{k}: {v:,.2f}")

    print("\nRate cap hedge info:")
    for k, v in rate_cap_info.items():
        if "rate" in k or "vol" in k:
            print(f"{k}: {v:,.6f}")
        elif "eur" in k:
            print(f"{k}: {v:,.2f} EUR")
        else:
            print(f"{k}: {v:,.2f}")

    print("\nRisk metrics - Rate cap:")
    for k, v in rate_cap_metrics.items():
        print(f"{k}: {v:,.2f}")

    print("\nLayered hedge info:")
    for k, v in layered_info.items():
        print(f"{k}: {v:,.2f}")

    print("\nRisk metrics - Layered hedge:")
    for k, v in layered_metrics.items():
        print(f"{k}: {v:,.2f}")

    print("\nOptional layered hedge info:")
    for k, v in optional_layered_info.items():
        print(f"{k}: {v:,.2f}")

    print("\nRisk metrics - Optional layered hedge:")
    for k, v in optional_layered_metrics.items():
        print(f"{k}: {v:,.2f}")

    comparison = pd.DataFrame(
        {
            "Unhedged": {
                "EaR_95": metrics["ear_95"],
                "CFaR_95": metrics["cfar_95"],
                "Std_CF": metrics["std_cash_flow"],
            },
            "FX Forward": {
                "EaR_95": fx_metrics["ear_95"],
                "CFaR_95": fx_metrics["cfar_95"],
                "Std_CF": fx_metrics["std_cash_flow"],
            },
            "FX Collar": {
                "EaR_95": fx_collar_metrics["ear_95"],
                "CFaR_95": fx_collar_metrics["cfar_95"],
                "Std_CF": fx_collar_metrics["std_cash_flow"],
            },
            "Commodity Forward": {
                "EaR_95": commodity_metrics["ear_95"],
                "CFaR_95": commodity_metrics["cfar_95"],
                "Std_CF": commodity_metrics["std_cash_flow"],
            },
            "Rate Swap": {
                "EaR_95": rate_metrics["ear_95"],
                "CFaR_95": rate_metrics["cfar_95"],
                "Std_CF": rate_metrics["std_cash_flow"],
            },
            "Rate Cap": {
                "EaR_95": rate_cap_metrics["ear_95"],
                "CFaR_95": rate_cap_metrics["cfar_95"],
                "Std_CF": rate_cap_metrics["std_cash_flow"],
            },
            "Layered Hedge": {
                "EaR_95": layered_metrics["ear_95"],
                "CFaR_95": layered_metrics["cfar_95"],
                "Std_CF": layered_metrics["std_cash_flow"],
            },
             "Optional Layered": {
                "EaR_95": optional_layered_metrics["ear_95"],
                "CFaR_95": optional_layered_metrics["cfar_95"],
                "Std_CF": optional_layered_metrics["std_cash_flow"],
            },
        }
    ).T

    comparison = comparison.round(2)
    comparison.to_csv(PROCESSED_DIR / "hedge_comparison.csv")

    print("\nConfronto sintetico:")
    print(comparison)

    print("\nSimulated dataset preview:")
    preview = simulated_df[["fx", "copper", "eur_rate", "cash_flow"]].head().copy()
    preview["cash_flow"] = preview["cash_flow"].map(lambda x: f"{x:,.2f}")
    preview["fx"] = preview["fx"].map(lambda x: f"{x:.4f}")
    preview["copper"] = preview["copper"].map(lambda x: f"{x:,.4f}")
    preview["eur_rate"] = preview["eur_rate"].map(lambda x: f"{x:.4f}")
    print(preview)

if __name__ == "__main__":
    main()