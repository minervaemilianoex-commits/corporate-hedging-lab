from config import CompanyConfig, MarketConfig, HedgeConfig


def main():
    company = CompanyConfig()
    market = MarketConfig()
    hedge = HedgeConfig()  # tenuto per coerenza di struttura

    # Costo commodity in USD = quantità * prezzo del rame
    commodity_cost_usd = company.commodity_qty_ton * market.copper_price_usd_ton

    # Esposizione netta in USD
    net_usd_exposure = (
        company.rev_usd
        - company.cost_usd
        - commodity_cost_usd
    )

    # Ricavi totali in EUR
    revenue_total_eur = (
        company.rev_eur
        + company.rev_usd * market.fx_spot
    )

    # Costi operativi totali in EUR
    operating_costs_total_eur = (
        company.cost_eur
        + company.cost_usd * market.fx_spot
        + commodity_cost_usd * market.fx_spot
    )

    # EBIT
    ebit = revenue_total_eur - operating_costs_total_eur

    # Interest cost sul debito floating per l'orizzonte di rischio
    interest_cost = (
        company.debt_notional_eur
        * (market.eur_rate + company.debt_spread)
        * company.horizon_years
    )

    # Cash flow before hedging
    cash_flow = ebit - interest_cost

    # Quick check aggiuntivo
    net_usd_exposure_pct = net_usd_exposure / company.rev_usd

    print("=== BASE CASE CHECK ===")
    print(f"Commodity cost: {commodity_cost_usd:,.0f} USD")
    print(f"Net USD exposure: {net_usd_exposure:,.0f} USD")
    print(f"Net USD exposure as % of USD revenue: {net_usd_exposure_pct:.1%}")
    print(f"Revenue total (EUR eq.): {revenue_total_eur:,.0f} EUR")
    print(f"Operating costs total (EUR eq.): {operating_costs_total_eur:,.0f} EUR")
    print(f"EBIT before hedging: {ebit:,.0f} EUR")
    print(f"Interest cost over horizon: {interest_cost:,.0f} EUR")
    print(f"Cash flow before hedging: {cash_flow:,.0f} EUR")


if __name__ == "__main__":
    main()