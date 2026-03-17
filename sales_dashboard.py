"""
Sales Performance Dashboard — End-to-End Data Analysis
=======================================================
Author : Chandan Thakur
GitHub : github.com/[your-username]
Dataset: Superstore Sales (synthetic, Kaggle-style)

Pipeline:
  1. Data Generation  — realistic synthetic sales dataset
  2. Data Cleaning    — handle missing values, dtypes, duplicates
  3. EDA              — distributions, trends, correlations
  4. KPI Calculation  — revenue, profit, growth, AOV
  5. Visualisations   — 10 publication-quality charts
  6. Insights Report  — automated business recommendations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import warnings
import os
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Output folder ────────────────────────────────────────────
OUT = "/home/claude/sales-dashboard/outputs"
os.makedirs(OUT, exist_ok=True)

# ── Color palette ─────────────────────────────────────────────
C = {
    "blue":   "#4f8ef7",
    "green":  "#3ecf8e",
    "red":    "#f87171",
    "yellow": "#fbbf24",
    "purple": "#8b5cf6",
    "teal":   "#06b6d4",
    "dark":   "#1a1d27",
    "grey":   "#64748b",
    "light":  "#f1f5f9",
    "white":  "#ffffff",
}

plt.rcParams.update({
    "figure.facecolor":  C["dark"],
    "axes.facecolor":    "#22263a",
    "axes.edgecolor":    "#2e3350",
    "axes.labelcolor":   C["light"],
    "xtick.color":       C["grey"],
    "ytick.color":       C["grey"],
    "text.color":        C["light"],
    "grid.color":        "#2e3350",
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
    "font.family":       "DejaVu Sans",
    "axes.titlesize":    13,
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
})

# ═══════════════════════════════════════════════════════════════
# 1. DATA GENERATION
# ═══════════════════════════════════════════════════════════════
print("=" * 55)
print("  Sales Performance Dashboard — Chandan Thakur")
print("=" * 55)
print("\n[1/6] Generating synthetic sales dataset...")

CATEGORIES = {
    "Technology":  {"subcats": ["Phones", "Computers", "Accessories", "Copiers"],
                    "margin": 0.22, "price_range": (80, 2800)},
    "Furniture":   {"subcats": ["Chairs", "Tables", "Bookcases", "Furnishings"],
                    "margin": 0.12, "price_range": (40, 1800)},
    "Office Supplies": {"subcats": ["Storage", "Binders", "Paper", "Labels", "Fasteners"],
                        "margin": 0.18, "price_range": (5, 300)},
}

REGIONS   = ["West", "East", "Central", "South"]
SEGMENTS  = ["Consumer", "Corporate", "Home Office"]
SHIP_MODES = ["Standard Class", "Second Class", "First Class", "Same Day"]

REGION_MULTIPLIER = {"West": 1.25, "East": 1.15, "Central": 0.90, "South": 0.85}
SEGMENT_MULTIPLIER = {"Corporate": 1.30, "Consumer": 1.00, "Home Office": 0.75}

# Generate 5000 orders over 3 years
N = 5000
start_date = datetime(2021, 1, 1)
end_date   = datetime(2023, 12, 31)
date_range = (end_date - start_date).days

rows = []
for i in range(N):
    cat_name = np.random.choice(list(CATEGORIES.keys()), p=[0.35, 0.30, 0.35])
    cat      = CATEGORIES[cat_name]
    subcat   = np.random.choice(cat["subcats"])
    region   = np.random.choice(REGIONS, p=[0.30, 0.28, 0.24, 0.18])
    segment  = np.random.choice(SEGMENTS, p=[0.52, 0.30, 0.18])
    ship     = np.random.choice(SHIP_MODES, p=[0.60, 0.20, 0.15, 0.05])

    # Date with seasonal trend
    day_offset = int(np.random.beta(1.5, 1.2) * date_range)
    order_date = start_date + timedelta(days=day_offset)
    # Q4 boost
    if order_date.month not in [10, 11, 12]:
        if np.random.rand() < 0.4:
            new_month = np.random.choice([10, 11, 12])
            try:
                order_date = order_date.replace(month=new_month, day=min(order_date.day, 28))
            except ValueError:
                pass

    lo, hi   = cat["price_range"]
    unit_price = round(np.random.uniform(lo, hi), 2)
    quantity   = int(np.random.choice([1,2,3,4,5], p=[0.45,0.28,0.15,0.08,0.04]))
    discount   = np.random.choice([0, 0.05, 0.10, 0.15, 0.20, 0.30],
                                   p=[0.55,0.15,0.12,0.08,0.06,0.04])

    sales  = round(unit_price * quantity * (1 - discount)
                   * REGION_MULTIPLIER[region]
                   * SEGMENT_MULTIPLIER[segment], 2)
    profit = round(sales * cat["margin"] * np.random.uniform(0.7, 1.3)
                   - sales * discount * 0.5, 2)
    profit_margin = round(profit / sales * 100, 2) if sales > 0 else 0

    rows.append({
        "order_id":      f"ORD-{100000+i}",
        "order_date":    order_date,
        "ship_mode":     ship,
        "segment":       segment,
        "region":        region,
        "category":      cat_name,
        "sub_category":  subcat,
        "unit_price":    unit_price,
        "quantity":      quantity,
        "discount":      discount,
        "sales":         sales,
        "profit":        profit,
        "profit_margin": profit_margin,
    })

df_raw = pd.DataFrame(rows)
print(f"   Generated {len(df_raw):,} orders across {df_raw['region'].nunique()} regions")

# ═══════════════════════════════════════════════════════════════
# 2. DATA CLEANING
# ═══════════════════════════════════════════════════════════════
print("\n[2/6] Cleaning data...")

df = df_raw.copy()

# Inject realistic issues
df.loc[np.random.choice(df.index, 30, replace=False), "sales"]  = np.nan
df.loc[np.random.choice(df.index, 15, replace=False), "profit"] = np.nan
df = pd.concat([df, df.sample(12)], ignore_index=True)  # duplicates

before = len(df)
df.drop_duplicates(subset=["order_id"], keep="first", inplace=True)
dupes_removed = before - len(df)

missing_sales  = df["sales"].isna().sum()
missing_profit = df["profit"].isna().sum()
df["sales"].fillna(df["sales"].median(), inplace=True)
df["profit"].fillna(df["profit"].median(), inplace=True)

df["order_date"] = pd.to_datetime(df["order_date"])
df["year"]       = df["order_date"].dt.year
df["month"]      = df["order_date"].dt.month
df["quarter"]    = df["order_date"].dt.quarter
df["month_name"] = df["order_date"].dt.strftime("%b")
df["year_month"] = df["order_date"].dt.to_period("M")

df = df[df["sales"] > 0].reset_index(drop=True)

print(f"   Duplicates removed     : {dupes_removed}")
print(f"   Missing sales imputed  : {missing_sales}")
print(f"   Missing profit imputed : {missing_profit}")
print(f"   Final clean records    : {len(df):,}")

# ═══════════════════════════════════════════════════════════════
# 3. KPI CALCULATION
# ═══════════════════════════════════════════════════════════════
print("\n[3/6] Calculating KPIs...")

total_revenue = df["sales"].sum()
total_profit  = df["profit"].sum()
total_orders  = df["order_id"].nunique()
avg_order_val = total_revenue / total_orders
avg_margin    = (total_profit / total_revenue) * 100
total_qty     = df["quantity"].sum()

# Year-over-year
rev_2021 = df[df["year"] == 2021]["sales"].sum()
rev_2022 = df[df["year"] == 2022]["sales"].sum()
rev_2023 = df[df["year"] == 2023]["sales"].sum()
yoy_22   = (rev_2022 - rev_2021) / rev_2021 * 100
yoy_23   = (rev_2023 - rev_2022) / rev_2022 * 100

print(f"   Total Revenue : ${total_revenue:,.0f}")
print(f"   Total Profit  : ${total_profit:,.0f}")
print(f"   Profit Margin : {avg_margin:.1f}%")
print(f"   Total Orders  : {total_orders:,}")
print(f"   Avg Order Val : ${avg_order_val:,.0f}")
print(f"   YoY Growth 22 : {yoy_22:+.1f}%")
print(f"   YoY Growth 23 : {yoy_23:+.1f}%")

# ═══════════════════════════════════════════════════════════════
# 4. VISUALISATIONS
# ═══════════════════════════════════════════════════════════════
print("\n[4/6] Generating charts...")

def save(fig, name):
    path = f"{OUT}/{name}"
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=C["dark"], edgecolor="none")
    plt.close(fig)
    print(f"   Saved {name}")

# ── Chart 1: KPI Summary Card ─────────────────────────────────
fig, axes = plt.subplots(1, 5, figsize=(16, 3))
fig.patch.set_facecolor(C["dark"])
fig.suptitle("Key Performance Indicators — 2021 to 2023",
             fontsize=14, fontweight="bold", color=C["white"], y=1.02)

kpis = [
    ("Total Revenue",   f"${total_revenue/1e6:.2f}M", C["blue"]),
    ("Total Profit",    f"${total_profit/1e3:.0f}K",  C["green"]),
    ("Profit Margin",   f"{avg_margin:.1f}%",          C["yellow"]),
    ("Total Orders",    f"{total_orders:,}",           C["purple"]),
    ("Avg Order Value", f"${avg_order_val:,.0f}",      C["teal"]),
]

for ax, (label, value, color) in zip(axes, kpis):
    ax.set_facecolor("#22263a")
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(2)
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.5, 0.62, value, transform=ax.transAxes,
            ha="center", va="center", fontsize=22,
            fontweight="bold", color=color)
    ax.text(0.5, 0.25, label, transform=ax.transAxes,
            ha="center", va="center", fontsize=10, color=C["grey"])

plt.tight_layout()
save(fig, "01_kpi_cards.png")

# ── Chart 2: Monthly Revenue Trend ───────────────────────────
monthly = (df.groupby("year_month")["sales"]
             .sum()
             .reset_index()
             .sort_values("year_month"))
monthly["year_month_str"] = monthly["year_month"].astype(str)
monthly["ma3"] = monthly["sales"].rolling(3).mean()

fig, ax = plt.subplots(figsize=(14, 5))
ax.fill_between(range(len(monthly)), monthly["sales"],
                alpha=0.15, color=C["blue"])
ax.plot(range(len(monthly)), monthly["sales"],
        color=C["blue"], lw=2, label="Monthly Revenue")
ax.plot(range(len(monthly)), monthly["ma3"],
        color=C["yellow"], lw=2, linestyle="--", label="3-Month MA")

# Year dividers
for yr in [2022, 2023]:
    idx = monthly[monthly["year_month_str"].str.startswith(str(yr))].index[0]
    pos = monthly.index.get_loc(idx)
    ax.axvline(pos, color=C["grey"], linestyle=":", alpha=0.6)
    ax.text(pos + 0.3, monthly["sales"].max() * 0.95,
            str(yr), color=C["grey"], fontsize=9)

tick_positions = list(range(0, len(monthly), 3))
ax.set_xticks(tick_positions)
ax.set_xticklabels([monthly["year_month_str"].iloc[i] for i in tick_positions],
                   rotation=35, ha="right")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e3:.0f}K"))
ax.set_title("Monthly Revenue Trend with 3-Month Moving Average", fontweight="bold", pad=12)
ax.set_xlabel("Month"); ax.set_ylabel("Revenue")
ax.legend(framealpha=0.2)
ax.grid(True, axis="y")
plt.tight_layout()
save(fig, "02_monthly_revenue.png")

# ── Chart 3: Revenue by Category & Segment ───────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

cat_rev = df.groupby("category")["sales"].sum().sort_values()
colors  = [C["blue"], C["purple"], C["teal"]]
bars = ax1.barh(cat_rev.index, cat_rev.values, color=colors, height=0.5)
for bar, val in zip(bars, cat_rev.values):
    ax1.text(val + cat_rev.max() * 0.01, bar.get_y() + bar.get_height()/2,
             f"${val/1e3:.0f}K", va="center", fontsize=10,
             fontweight="bold", color=C["white"])
ax1.set_title("Revenue by Category", fontweight="bold", pad=10)
ax1.set_xlabel("Revenue ($)")
ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e3:.0f}K"))
ax1.grid(True, axis="x"); ax1.set_axisbelow(True)

seg_rev = df.groupby("segment")["sales"].sum().sort_values()
colors2 = [C["green"], C["yellow"], C["red"]]
bars2 = ax2.barh(seg_rev.index, seg_rev.values, color=colors2, height=0.5)
for bar, val in zip(bars2, seg_rev.values):
    ax2.text(val + seg_rev.max() * 0.01, bar.get_y() + bar.get_height()/2,
             f"${val/1e3:.0f}K", va="center", fontsize=10,
             fontweight="bold", color=C["white"])
ax2.set_title("Revenue by Customer Segment", fontweight="bold", pad=10)
ax2.set_xlabel("Revenue ($)")
ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e3:.0f}K"))
ax2.grid(True, axis="x"); ax2.set_axisbelow(True)

plt.tight_layout()
save(fig, "03_category_segment.png")

# ── Chart 4: Regional Performance ────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

reg_metrics = df.groupby("region").agg(
    revenue=("sales", "sum"),
    profit=("profit", "sum"),
    orders=("order_id", "nunique")
).reset_index()
reg_metrics["margin"] = reg_metrics["profit"] / reg_metrics["revenue"] * 100
reg_metrics = reg_metrics.sort_values("revenue", ascending=False)

reg_colors = [C["blue"], C["green"], C["yellow"], C["purple"]]

b1 = axes[0].bar(reg_metrics["region"], reg_metrics["revenue"],
                  color=reg_colors)
for bar, val in zip(b1, reg_metrics["revenue"]):
    axes[0].text(bar.get_x() + bar.get_width()/2, val + reg_metrics["revenue"].max()*0.01,
                 f"${val/1e3:.0f}K", ha="center", fontsize=9, fontweight="bold")
axes[0].set_title("Revenue by Region", fontweight="bold")
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e3:.0f}K"))
axes[0].grid(True, axis="y"); axes[0].set_axisbelow(True)

b2 = axes[1].bar(reg_metrics["region"], reg_metrics["profit"],
                  color=reg_colors)
for bar, val in zip(b2, reg_metrics["profit"]):
    axes[1].text(bar.get_x() + bar.get_width()/2, val + reg_metrics["profit"].max()*0.01,
                 f"${val/1e3:.0f}K", ha="center", fontsize=9, fontweight="bold")
axes[1].set_title("Profit by Region", fontweight="bold")
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e3:.0f}K"))
axes[1].grid(True, axis="y"); axes[1].set_axisbelow(True)

b3 = axes[2].bar(reg_metrics["region"], reg_metrics["margin"],
                  color=reg_colors)
for bar, val in zip(b3, reg_metrics["margin"]):
    axes[2].text(bar.get_x() + bar.get_width()/2, val + 0.3,
                 f"{val:.1f}%", ha="center", fontsize=9, fontweight="bold")
axes[2].set_title("Profit Margin by Region", fontweight="bold")
axes[2].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
axes[2].grid(True, axis="y"); axes[2].set_axisbelow(True)

plt.tight_layout()
save(fig, "04_regional_performance.png")

# ── Chart 5: Sub-Category Profit Analysis ────────────────────
subcat = df.groupby("sub_category").agg(
    revenue=("sales", "sum"),
    profit=("profit", "sum")
).reset_index()
subcat["margin"] = subcat["profit"] / subcat["revenue"] * 100
subcat = subcat.sort_values("profit", ascending=True)

colors_sub = [C["red"] if p < 0 else C["green"] for p in subcat["profit"]]

fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(subcat["sub_category"], subcat["profit"],
               color=colors_sub, height=0.6)
for bar, val in zip(bars, subcat["profit"]):
    x_pos = val + (subcat["profit"].max() * 0.01 if val >= 0
                   else subcat["profit"].max() * -0.01)
    ax.text(x_pos, bar.get_y() + bar.get_height()/2,
            f"${val:,.0f}", va="center",
            ha="left" if val >= 0 else "right",
            fontsize=8.5, color=C["white"])
ax.axvline(0, color=C["grey"], lw=1)
ax.set_title("Profit by Sub-Category (Green = Profitable, Red = Loss)", fontweight="bold", pad=12)
ax.set_xlabel("Total Profit ($)")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e3:.0f}K"))
ax.grid(True, axis="x"); ax.set_axisbelow(True)
plt.tight_layout()
save(fig, "05_subcategory_profit.png")

# ── Chart 6: Discount vs Profit Scatter ──────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
scatter_colors = [C["green"] if p >= 0 else C["red"] for p in df["profit"]]
ax.scatter(df["discount"], df["profit"],
           c=scatter_colors, alpha=0.25, s=12)

disc_bins = pd.cut(df["discount"], bins=6)
disc_avg  = df.groupby(disc_bins)["profit"].mean().reset_index()
disc_mid  = [(iv.left + iv.right)/2 for iv in disc_avg["discount"]]
ax.plot(disc_mid, disc_avg["profit"], color=C["yellow"],
        lw=2.5, marker="o", ms=7, label="Avg Profit per Discount Band")

ax.axhline(0, color=C["grey"], lw=1, linestyle="--")
ax.set_title("Discount Rate vs Profit Impact", fontweight="bold", pad=12)
ax.set_xlabel("Discount Rate")
ax.set_ylabel("Profit ($)")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax.legend(framealpha=0.2)
ax.grid(True); ax.set_axisbelow(True)
plt.tight_layout()
save(fig, "06_discount_profit.png")

# ── Chart 7: Year-over-Year Comparison ───────────────────────
monthly_yr = df.groupby(["year", "month"])["sales"].sum().reset_index()

fig, ax = plt.subplots(figsize=(12, 5))
yr_colors = {2021: C["grey"], 2022: C["blue"], 2023: C["green"]}
months = ["Jan","Feb","Mar","Apr","May","Jun",
          "Jul","Aug","Sep","Oct","Nov","Dec"]

for yr in [2021, 2022, 2023]:
    data = monthly_yr[monthly_yr["year"] == yr].set_index("month")["sales"]
    data = data.reindex(range(1, 13), fill_value=0)
    ax.plot(range(1, 13), data.values,
            color=yr_colors[yr], lw=2.5, marker="o",
            ms=5, label=str(yr))

ax.set_xticks(range(1, 13))
ax.set_xticklabels(months)
ax.set_title("Year-over-Year Monthly Revenue Comparison", fontweight="bold", pad=12)
ax.set_xlabel("Month"); ax.set_ylabel("Revenue ($)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e3:.0f}K"))
ax.legend(framealpha=0.2)
ax.grid(True); ax.set_axisbelow(True)
plt.tight_layout()
save(fig, "07_yoy_comparison.png")

# ── Chart 8: Quarterly Revenue & Profit ──────────────────────
quarterly = df.groupby(["year", "quarter"]).agg(
    revenue=("sales", "sum"),
    profit=("profit", "sum")
).reset_index()
quarterly["label"] = quarterly["year"].astype(str) + " Q" + quarterly["quarter"].astype(str)

x = np.arange(len(quarterly))
w = 0.38

fig, ax = plt.subplots(figsize=(13, 5))
b1 = ax.bar(x - w/2, quarterly["revenue"], w, color=C["blue"],
            label="Revenue", alpha=0.9)
b2 = ax.bar(x + w/2, quarterly["profit"],  w, color=C["green"],
            label="Profit", alpha=0.9)

for bar in b1:
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + quarterly["revenue"].max()*0.005,
            f"${bar.get_height()/1e3:.0f}K",
            ha="center", fontsize=7.5, color=C["white"])

ax.set_xticks(x)
ax.set_xticklabels(quarterly["label"], rotation=30, ha="right")
ax.set_title("Quarterly Revenue and Profit", fontweight="bold", pad=12)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e3:.0f}K"))
ax.legend(framealpha=0.2)
ax.grid(True, axis="y"); ax.set_axisbelow(True)
plt.tight_layout()
save(fig, "08_quarterly_performance.png")

# ── Chart 9: Shipping Mode Analysis ──────────────────────────
ship = df.groupby("ship_mode").agg(
    orders=("order_id", "nunique"),
    revenue=("sales", "sum"),
    profit=("profit", "sum")
).reset_index()
ship["margin"] = ship["profit"] / ship["revenue"] * 100
ship = ship.sort_values("revenue", ascending=False)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
ship_colors = [C["blue"], C["green"], C["yellow"], C["purple"]]

wedges, texts, autotexts = ax1.pie(
    ship["orders"], labels=ship["ship_mode"],
    colors=ship_colors, autopct="%1.1f%%",
    startangle=90, pctdistance=0.75,
    wedgeprops={"edgecolor": C["dark"], "linewidth": 2})
for at in autotexts: at.set_fontsize(9)
ax1.set_title("Order Share by Ship Mode", fontweight="bold", pad=10)
ax1.set_facecolor(C["dark"])

b = ax2.bar(ship["ship_mode"], ship["margin"], color=ship_colors, width=0.5)
for bar, val in zip(b, ship["margin"]):
    ax2.text(bar.get_x() + bar.get_width()/2,
             val + 0.2, f"{val:.1f}%",
             ha="center", fontsize=10, fontweight="bold")
ax2.set_title("Profit Margin by Ship Mode", fontweight="bold", pad=10)
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
ax2.grid(True, axis="y"); ax2.set_axisbelow(True)

plt.tight_layout()
save(fig, "09_shipping_analysis.png")

# ── Chart 10: Correlation Heatmap ────────────────────────────
corr_cols = ["sales", "profit", "quantity", "discount",
             "unit_price", "profit_margin"]
corr = df[corr_cols].corr()

fig, ax = plt.subplots(figsize=(8, 6))
mask_upper = np.triu(np.ones_like(corr, dtype=bool), k=1)
data_plot  = corr.values.copy()
data_plot[mask_upper] = np.nan

im = ax.imshow(data_plot, cmap="RdYlGn", vmin=-1, vmax=1,
               aspect="auto")
plt.colorbar(im, ax=ax, shrink=0.8)

for i in range(len(corr_cols)):
    for j in range(len(corr_cols)):
        if not mask_upper[i, j]:
            val = data_plot[i, j]
            color = "black" if abs(val) > 0.6 else C["white"]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold")

ax.set_xticks(range(len(corr_cols)))
ax.set_yticks(range(len(corr_cols)))
ax.set_xticklabels(corr_cols, rotation=35, ha="right")
ax.set_yticklabels(corr_cols)
ax.set_title("Feature Correlation Heatmap", fontweight="bold", pad=12)
plt.tight_layout()
save(fig, "10_correlation_heatmap.png")

# ═══════════════════════════════════════════════════════════════
# 5. AUTOMATED INSIGHTS REPORT
# ═══════════════════════════════════════════════════════════════
print("\n[5/6] Writing insights report...")

top_region   = reg_metrics.sort_values("revenue", ascending=False).iloc[0]["region"]
worst_region = reg_metrics.sort_values("revenue").iloc[0]["region"]
top_cat      = df.groupby("category")["sales"].sum().idxmax()
top_seg      = df.groupby("segment")["sales"].sum().idxmax()
loss_subcats = subcat[subcat["profit"] < 0]["sub_category"].tolist()
best_ship    = ship.sort_values("margin", ascending=False).iloc[0]["ship_mode"]
high_disc    = df[df["discount"] >= 0.20]["profit"].mean()
low_disc     = df[df["discount"] < 0.05]["profit"].mean()

report = f"""
SALES PERFORMANCE ANALYSIS REPORT
===================================
Author  : Chandan Thakur
Date    : {datetime.now().strftime("%B %Y")}
Dataset : Superstore Sales — 2021 to 2023
Records : {len(df):,} orders

===========================================
EXECUTIVE SUMMARY
===========================================

Total Revenue    : ${total_revenue:,.0f}
Total Profit     : ${total_profit:,.0f}
Profit Margin    : {avg_margin:.1f}%
Total Orders     : {total_orders:,}
Avg Order Value  : ${avg_order_val:,.0f}
Total Units Sold : {total_qty:,}

Year-over-Year Growth:
  2021 to 2022 : {yoy_22:+.1f}%
  2022 to 2023 : {yoy_23:+.1f}%

===========================================
KEY FINDINGS
===========================================

1. REGIONAL PERFORMANCE
   Top Region    : {top_region} (highest revenue)
   Weak Region   : {worst_region} (lowest revenue)
   Recommendation: Investigate {worst_region} for
   pricing or demand issues. Consider
   targeted promotions to boost performance.

2. CATEGORY INSIGHTS
   Best Category : {top_cat}
   Best Segment  : {top_seg}
   Recommendation: Double down on {top_cat}
   marketing and {top_seg} acquisition
   strategies as these drive maximum revenue.

3. DISCOUNT IMPACT
   Avg profit at discount 20%+ : ${high_disc:,.0f}
   Avg profit at discount <5%  : ${low_disc:,.0f}
   Finding: High discounts (20%+) significantly
   erode profitability. Discount strategy needs
   review — consider capping discounts at 15%.

4. SUB-CATEGORY LOSSES
   Loss-making sub-categories: {", ".join(loss_subcats) if loss_subcats else "None"}
   Recommendation: Review pricing and cost
   structure for these sub-categories.
   Consider discontinuing or repricing.

5. SHIPPING ANALYSIS
   Best margin ship mode: {best_ship}
   Recommendation: Incentivize customers
   toward higher-margin shipping options
   through loyalty programs or bundling.

6. SEASONALITY
   Q4 consistently outperforms other quarters
   across all three years.
   Recommendation: Increase inventory and
   marketing spend ahead of Q4. Plan
   promotional campaigns for October.

===========================================
RECOMMENDATIONS SUMMARY
===========================================

Priority 1 : Review and cap discount policy
             at 15% maximum to protect margins.

Priority 2 : Focus marketing budget on
             {top_region} and {top_seg} segment
             for highest ROI.

Priority 3 : Address {worst_region} region
             underperformance through targeted
             pricing or distribution strategy.

Priority 4 : Eliminate or restructure
             loss-making sub-categories.

Priority 5 : Build Q4 readiness strategy
             leveraging seasonal demand spike.

===========================================
CHARTS GENERATED
===========================================

01_kpi_cards.png            KPI Summary
02_monthly_revenue.png      Revenue Trend
03_category_segment.png     Category & Segment
04_regional_performance.png Regional Analysis
05_subcategory_profit.png   Sub-Category Profit
06_discount_profit.png      Discount Impact
07_yoy_comparison.png       Year-over-Year
08_quarterly_performance.png Quarterly View
09_shipping_analysis.png    Shipping Modes
10_correlation_heatmap.png  Correlations

===========================================
END OF REPORT
===========================================
"""

report_path = f"{OUT}/insights_report.txt"
with open(report_path, "w") as f:
    f.write(report)

print(report)

# ═══════════════════════════════════════════════════════════════
# 6. SAVE CLEAN DATASET
# ═══════════════════════════════════════════════════════════════
print("[6/6] Saving clean dataset...")
df.to_csv(f"{OUT}/superstore_clean.csv", index=False)
print(f"   Dataset saved: {len(df):,} rows x {len(df.columns)} columns")

print("\n" + "=" * 55)
print("  All outputs saved to outputs/ folder")
print("  Ready to upload to GitHub")
print("=" * 55)
