# Sales Performance Dashboard — End-to-End Data Analysis

A complete end-to-end sales analytics pipeline built with Python. Generates a synthetic Superstore-style dataset, performs data cleaning, calculates business KPIs, produces 10 publication-quality visualisations, and outputs an automated insights report with business recommendations.

## Results

| KPI | Value |
|---|---|
| Total Revenue | $8.55M |
| Total Profit | $1.35M |
| Profit Margin | 15.8% |
| Total Orders | 4,970 |
| Avg Order Value | $1,721 |
| YoY Growth 2022 | +53.0% |
| YoY Growth 2023 | +1.2% |

## Key Findings

- West region drives highest revenue. South region underperforms.
- Technology category generates maximum revenue across all segments.
- High discounts (20%+) reduce average profit by 71% vs low-discount orders.
- Q4 consistently outperforms all other quarters across 3 years.
- First Class shipping yields the highest profit margin.

## Pipeline

```
1. Data Generation   -> 5,000 synthetic orders with realistic distributions
2. Data Cleaning     -> Handle duplicates, missing values, type casting
3. KPI Calculation   -> Revenue, profit, margin, AOV, YoY growth
4. Visualisation     -> 10 charts covering all business dimensions
5. Insights Report   -> Automated business recommendations
6. Export            -> Clean CSV dataset + all chart PNGs
```

## Charts Generated

| File | Description |
|---|---|
| 01_kpi_cards.png | Executive KPI summary |
| 02_monthly_revenue.png | Monthly trend with 3-month moving average |
| 03_category_segment.png | Revenue by category and customer segment |
| 04_regional_performance.png | Revenue, profit, margin by region |
| 05_subcategory_profit.png | Profit breakdown by sub-category |
| 06_discount_profit.png | Discount rate vs profit impact scatter |
| 07_yoy_comparison.png | Year-over-year monthly comparison |
| 08_quarterly_performance.png | Quarterly revenue and profit bars |
| 09_shipping_analysis.png | Ship mode share and margin analysis |
| 10_correlation_heatmap.png | Feature correlation heatmap |

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.8+ | Core language |
| Pandas | Data manipulation and cleaning |
| NumPy | Numerical operations |
| Matplotlib | All visualisations |
| CSV | Dataset export |

## Project Structure

```
sales-dashboard/
    sales_dashboard.py    Main analysis pipeline
    requirements.txt      Python dependencies
    README.md             Project documentation
    outputs/
        01_kpi_cards.png
        02_monthly_revenue.png
        03_category_segment.png
        04_regional_performance.png
        05_subcategory_profit.png
        06_discount_profit.png
        07_yoy_comparison.png
        08_quarterly_performance.png
        09_shipping_analysis.png
        10_correlation_heatmap.png
        insights_report.txt
        superstore_clean.csv
```

## Quick Start

```bash
git clone https://github.com/[your-username]/sales-dashboard.git
cd sales-dashboard
pip install -r requirements.txt
python sales_dashboard.py
```

All charts and the insights report will be saved to the outputs folder.

## Business Recommendations

Based on the analysis the following priorities were identified:

1. Cap discount policy at 15% maximum to protect profit margins.
2. Focus marketing budget on West region and Consumer segment for highest ROI.
3. Investigate South region underperformance through pricing or distribution audit.
4. Build Q4 readiness strategy by increasing inventory and campaign spend in September.
5. Incentivize customers toward First Class shipping through loyalty programs.

## Author

Chandan Thakur
- GitHub: github.com/[your-username]
- LinkedIn: linkedin.com/in/chandanthakur
- Behance: behance.net/outcastthakur
- Email: thakurchandan07c@gmail.com

## License

MIT License
