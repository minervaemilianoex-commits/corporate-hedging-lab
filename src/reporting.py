from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
CHARTS_DIR = BASE_DIR / "outputs" / "charts"
REPORTS_DIR = BASE_DIR / "outputs" / "reports"


def ensure_output_directories():
    """
    Crea le cartelle di output se non esistono.
    """
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_hedge_comparison() -> pd.DataFrame:
    """
    Carica la tabella finale di confronto delle strategie.
    """
    path = PROCESSED_DIR / "hedge_comparison.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"File non trovato: {path}\n"
            "Prima esegui python src/risk_engine.py"
        )

    df = pd.read_csv(path, index_col=0)
    return df


def clean_hedge_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pulisce e arrotonda la tabella finale.
    """
    out = df.copy()

    desired_order = [
        "Unhedged",
        "FX Forward",
        "FX Collar",
        "Commodity Forward",
        "Rate Swap",
        "Rate Cap",
        "Layered Hedge",
        "Optional Layered",
    ]

    existing_order = [name for name in desired_order if name in out.index]
    out = out.loc[existing_order]

    out = out.round(2)

    return out


def save_clean_table(df: pd.DataFrame):
    """
    Salva la tabella pulita nella cartella reports.
    """
    output_path = REPORTS_DIR / "hedge_comparison_clean.csv"
    df.to_csv(output_path)


def plot_metric_comparison(df: pd.DataFrame, metric: str, filename: str, title: str, ylabel: str):
    """
    Crea e salva un grafico a barre per una metrica specifica.
    """
    plt.figure(figsize=(10, 6))
    plt.bar(df.index, df[metric])
    plt.title(title)
    plt.xlabel("Strategia")
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    output_path = CHARTS_DIR / filename
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def load_simulation_file(filename: str) -> pd.DataFrame:
    """
    Carica un file di simulazione dalla cartella data/processed.
    """
    path = PROCESSED_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"File non trovato: {path}\n"
            "Prima esegui python src/risk_engine.py"
        )

    return pd.read_csv(path)


def plot_cashflow_distributions():
    """
    Crea un confronto delle distribuzioni del cash flow per le strategie principali.
    """
    unhedged = load_simulation_file("simulated_unhedged.csv")
    commodity = load_simulation_file("simulated_commodity_forward.csv")
    layered = load_simulation_file("simulated_layered.csv")
    optional_layered = load_simulation_file("simulated_optional_layered.csv")

    plt.figure(figsize=(10, 6))

    plt.hist(
        unhedged["cash_flow"],
        bins=50,
        alpha=0.5,
        label="Unhedged",
        density=True,
    )

    plt.hist(
        commodity["cash_flow_commodity_forward"],
        bins=50,
        alpha=0.5,
        label="Commodity Forward",
        density=True,
    )

    plt.hist(
        layered["cash_flow_layered"],
        bins=50,
        alpha=0.5,
        label="Layered Hedge",
        density=True,
    )

    plt.hist(
        optional_layered["cash_flow_optional_layered"],
        bins=50,
        alpha=0.5,
        label="Optional Layered",
        density=True,
    )

    plt.title("Distribuzione del cash flow simulato")
    plt.xlabel("Cash flow (EUR)")
    plt.ylabel("Densità")
    plt.legend()
    plt.tight_layout()

    output_path = CHARTS_DIR / "cashflow_distributions.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_fx_block(df: pd.DataFrame):
    """
    Confronto grafico del blocco FX:
    Unhedged vs FX Forward vs FX Collar
    """
    fx_df = df.loc[["Unhedged", "FX Forward", "FX Collar"]]

    plt.figure(figsize=(9, 6))
    plt.bar(fx_df.index, fx_df["CFaR_95"])
    plt.title("FX block comparison - CFaR 95%")
    plt.xlabel("Strategia")
    plt.ylabel("CFaR 95% (EUR)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    output_path = CHARTS_DIR / "fx_block_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_rate_block(df: pd.DataFrame):
    """
    Confronto grafico del blocco tassi:
    Unhedged vs Rate Swap vs Rate Cap
    """
    rate_df = df.loc[["Unhedged", "Rate Swap", "Rate Cap"]]

    plt.figure(figsize=(9, 6))
    plt.bar(rate_df.index, rate_df["CFaR_95"])
    plt.title("Rate block comparison - CFaR 95%")
    plt.xlabel("Strategia")
    plt.ylabel("CFaR 95% (EUR)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    output_path = CHARTS_DIR / "rate_block_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_final_recommendation(df: pd.DataFrame):
    """
    Grafico finale decisionale:
    Unhedged vs Commodity Forward vs Layered Hedge vs Optional Layered
    """
    final_df = df.loc[
        ["Unhedged", "Commodity Forward", "Layered Hedge", "Optional Layered"]
    ]

    plt.figure(figsize=(9, 6))
    plt.bar(final_df.index, final_df["CFaR_95"])
    plt.title("Final recommendation comparison - CFaR 95%")
    plt.xlabel("Strategia")
    plt.ylabel("CFaR 95% (EUR)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    output_path = CHARTS_DIR / "final_recommendation_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    ensure_output_directories()

    comparison = load_hedge_comparison()
    comparison_clean = clean_hedge_comparison(comparison)

    save_clean_table(comparison_clean)

    plot_metric_comparison(
        comparison_clean,
        metric="CFaR_95",
        filename="comparison_cfar.png",
        title="Confronto CFaR 95% per strategia",
        ylabel="CFaR 95% (EUR)",
    )

    plot_metric_comparison(
        comparison_clean,
        metric="EaR_95",
        filename="comparison_ear.png",
        title="Confronto EaR 95% per strategia",
        ylabel="EaR 95% (EUR)",
    )

    plot_metric_comparison(
        comparison_clean,
        metric="Std_CF",
        filename="comparison_stdcf.png",
        title="Confronto deviazione standard cash flow",
        ylabel="Std cash flow (EUR)",
    )

    plot_cashflow_distributions()
    plot_fx_block(comparison_clean)
    plot_rate_block(comparison_clean)
    plot_final_recommendation(comparison_clean)

    print("=== REPORTING STEP 7 CHECK ===")
    print("\nTabella finale pulita:")
    print(comparison_clean)

    print("\nFile salvati in:")
    print(REPORTS_DIR / "hedge_comparison_clean.csv")
    print(CHARTS_DIR / "comparison_cfar.png")
    print(CHARTS_DIR / "comparison_ear.png")
    print(CHARTS_DIR / "comparison_stdcf.png")
    print(CHARTS_DIR / "cashflow_distributions.png")
    print(CHARTS_DIR / "fx_block_comparison.png")
    print(CHARTS_DIR / "rate_block_comparison.png")
    print(CHARTS_DIR / "final_recommendation_comparison.png")


if __name__ == "__main__":
    main()