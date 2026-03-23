# Corporate Hedging Lab

A Python-based multi-risk hedging project designed to simulate and compare hedging strategies for an industrial European company exposed to **FX risk, commodity risk, and interest rate risk**.

The project was developed as a practical case study to replicate the type of reasoning used in a **Global Markets / Solutions / Derivatives** environment, combining quantitative modelling, market intuition, and risk management logic.

---

## Project objective

The goal of the project is to design and evaluate a hedging framework for a semi-real industrial company with the following characteristics:

- revenues partly denominated in **USD**
- operating costs partly denominated in **USD**
- commodity purchases linked to **industrial metals**
- floating-rate debt in **EUR**

The model measures the impact of market shocks on operating and cash-flow performance and compares alternative hedging strategies using:

- **EaR (Earnings at Risk)**
- **CFaR (Cash Flow at Risk)**
- **cash flow volatility**

The final aim is to understand which hedging structure is most effective depending on the company’s risk profile and management priorities.

---

## Company case

The case is built around a semi-real industrial company inspired by a European precision manufacturing / industrial components business.

Main assumptions:

- EUR revenues + USD export revenues
- USD operating costs
- USD-denominated industrial metals purchases (copper used as proxy)
- EUR floating-rate debt
- 3-month risk horizon

This setup creates a realistic interaction between:

- FX exposure
- commodity exposure
- interest rate exposure

---

## Tools used

The project was built entirely in **Python**.

Main libraries:

- `pandas` for data handling
- `numpy` for numerical calculations
- `matplotlib` for charts
- `scipy` for simple option-style pricing logic
- `pathlib` for file management

Data sources:

- **FRED** for macro and market time series
- processed simulation datasets generated directly by the project

---

## Folder structure

```text
corporate-hedging-lab
├── data
│   ├── raw
│   └── processed
├── outputs
│   ├── charts
│   └── reports
├── src
│   ├── config.py
│   ├── data_loader.py
│   ├── hedging.py
│   ├── main.py
│   ├── reporting.py
│   └── risk_engine.py
├── .gitignore
└── README.md

Folder description
data/raw
raw downloaded market series
data/processed
cleaned factor datasets and simulated strategy outputs
outputs/charts
generated visualizations for the report
outputs/reports
final comparison tables and report-ready outputs
src/config.py
project parameters and assumptions
src/data_loader.py
download and preprocessing of historical market data
src/hedging.py
implementation of hedging instruments and layered structures
src/risk_engine.py
Monte Carlo simulation and risk metrics engine
src/reporting.py
chart and table generation for reporting
Strategies implemented

The project compares both single-instrument hedges and layered hedging structures.

FX strategies
FX Forward
FX Collar
Commodity strategies
Commodity Forward
Interest rate strategies
Rate Swap
Rate Cap
Integrated strategies
Layered Hedge
FX Forward
Commodity Forward
Rate Swap
Optional Layered Hedge
FX Collar
Commodity Forward
Rate Cap
Benchmark
Unhedged
Methodology

The workflow of the project is the following:

Define company assumptions and market conventions
Download historical market data
Build a clean factor dataset
Estimate historical volatilities and co-movements
Simulate 3-month market scenarios
Compute unhedged EBIT and cash flow under each scenario
Apply alternative hedging strategies
Compare downside risk and cash flow stability

The model focuses on practical corporate hedging logic rather than purely theoretical pricing.

Risk metrics used

The main metrics used to compare strategies are:

EaR 95% (Earnings at Risk)
downside risk on EBIT
CFaR 95% (Cash Flow at Risk)
downside risk on cash flow
Std CF
standard deviation of cash flow across simulated scenarios

These metrics are used to evaluate both:

operating risk reduction
total cash flow stabilization
How to run the project

Make sure Python is installed and required libraries are available.

1. Build / refresh market data
python src/data_loader.py
2. Run the simulation and hedging engine
python src/risk_engine.py

This step generates:

simulated unhedged dataset
simulated datasets for each hedging strategy
comparison table in data/processed
3. Generate charts and final tables
python src/reporting.py

This step generates:

comparison charts
cash flow distribution chart
FX block comparison
rate block comparison
final recommendation comparison
clean report table in outputs/reports
Main results

The key findings of the project are the following:

1. Commodity exposure is the dominant risk driver

The strongest reduction in risk comes from the commodity forward hedge, showing that the company’s downside risk is driven mainly by USD-denominated commodity costs rather than by pure FX exposure.

2. FX hedges are not sufficient on their own

Both FX Forward and FX Collar are relatively weak in the current market setup.
The collar performs better than the forward, but neither materially improves the risk profile versus the unhedged case.

3. Rate hedges have secondary impact

The Rate Swap provides a small improvement in cash flow downside protection, while the Rate Cap is almost inactive under the current strike and horizon assumptions.

4. The layered linear hedge is strongest on cash-flow tail protection

The structure combining:

FX Forward
Commodity Forward
Rate Swap

achieves the best result on CFaR, making it the most protective solution against severe cash-flow downside.

5. The optional layered hedge is more flexible

The structure combining:

FX Collar
Commodity Forward
Rate Cap

delivers a very strong result as well, with slightly less protection on extreme downside but a more flexible and less rigid hedging profile.

Interpretation

The project shows that hedging effectiveness depends on the true economic source of risk, not on the sophistication of the instrument alone.

A more complex instrument is not automatically better if it is not aligned with the dominant exposure.

In this case:

the commodity hedge drives most of the benefit
FX protection is only complementary
interest-rate protection refines the cash-flow profile but is not the main driver

This is the main strategic insight of the project.

Why this project is relevant

This project was built to demonstrate skills relevant to roles in:

Global Markets
Structured Solutions
Equity / Rates / Cross-Asset Sales
Risk Management
Quantitative Finance
Corporate Derivatives Advisory

It combines:

market intuition
corporate hedging logic
Python implementation
scenario simulation
risk metric interpretation
communication of results in a report-ready format
Future improvements

Possible next developments include:

calibration of more realistic option premia
dynamic hedge ratios
scenario-dependent exposures
richer commodity basket instead of a single copper proxy
dashboard version of the project
optimization layer for hedge ratio selection
Author

Emilia / Minervaemilianoex-commits
Finance student project focused on derivatives, risk management, and corporate hedging applications.