"""Generate the deterministic sample datasets in sample_data/.

sales.csv      - ~4,000 order lines, Jan 2024 - Dec 2025, with seasonality, a declining product,
                 and deliberately injected anomalies (huge orders, negative quantities, a spike day,
                 duplicate rows, a few missing values) so every feature can be demonstrated.
customers.csv  - customer master data (segment, country) for multi-file joins.
products.csv   - product catalogue (category, unit cost) for margin analysis.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "sample_data"
rng = np.random.default_rng(42)

PRODUCTS = [
    # product, category, price, unit_cost, popularity, yearly trend
    ("Laptop Pro 14", "Electronics", 1299.0, 910.0, 1.0, 0.10),
    ("Wireless Mouse", "Accessories", 29.0, 11.0, 2.6, 0.05),
    ("Mechanical Keyboard", "Accessories", 89.0, 41.0, 1.6, 0.12),
    ("4K Monitor", "Electronics", 399.0, 270.0, 1.1, 0.08),
    ("USB-C Hub", "Accessories", 49.0, 18.0, 2.0, 0.15),
    ("Noise-Cancel Headphones", "Audio", 249.0, 140.0, 1.2, 0.20),
    ("Bluetooth Speaker", "Audio", 79.0, 38.0, 1.3, -0.05),
    ("Tablet 10", "Electronics", 499.0, 360.0, 0.9, -0.45),  # underperformer: steep decline
    ("Webcam HD", "Accessories", 69.0, 30.0, 1.0, -0.30),     # underperformer
    ("Smartwatch S", "Wearables", 199.0, 120.0, 1.1, 0.25),
]
REGIONS = {"North": 1.0, "South": 0.8, "East": 1.25, "West": 1.1, "Central": 0.55}
SEGMENTS = ["Enterprise", "SMB", "Consumer"]
COUNTRIES = ["India", "United Kingdom", "United States", "Germany", "UAE", "Singapore"]
CHANNELS = ["Online", "Retail", "Partner"]


def make_customers(n: int = 80) -> pd.DataFrame:
    first = ["Apex", "Blue", "Nova", "Prime", "Zen", "Urban", "Vertex", "Orbit", "Peak", "Swift", "Lumen", "Core"]
    second = ["Systems", "Retail", "Labs", "Traders", "Solutions", "Industries", "Digital", "Partners"]
    names = set()
    while len(names) < n:
        names.add(f"{rng.choice(first)} {rng.choice(second)}")
    names = sorted(names)
    df = pd.DataFrame({
        "customer_id": [f"C{i:03d}" for i in range(1, n + 1)],
        "customer_name": names,
        "segment": rng.choice(SEGMENTS, n, p=[0.25, 0.45, 0.30]),
        "country": rng.choice(COUNTRIES, n),
        "signup_date": pd.to_datetime("2022-01-01") + pd.to_timedelta(rng.integers(0, 700, n), unit="D"),
    })
    # Pareto weights: a handful of customers drive most revenue.
    df["_weight"] = rng.pareto(1.6, n) + 0.2
    return df


def make_sales(customers: pd.DataFrame, n: int = 4000) -> pd.DataFrame:
    start, end = pd.Timestamp("2024-01-01"), pd.Timestamp("2025-12-31")
    days = pd.date_range(start, end, freq="D")
    t = np.asarray((days - start).days, dtype=float) / 365.0
    month = np.asarray(days.month)
    season = 1 + 0.25 * np.sin(2 * np.pi * (month - 3) / 12) + np.where(np.isin(month, [11, 12]), 0.35, 0)
    day_w = season * (1 + 0.15 * t)
    dates = rng.choice(days, n, p=day_w / day_w.sum())

    prod_idx = []
    for d in dates:
        yrs = (pd.Timestamp(d) - start).days / 365.0
        w = np.array([p[4] * max(0.05, 1 + p[5] * yrs) for p in PRODUCTS])
        prod_idx.append(rng.choice(len(PRODUCTS), p=w / w.sum()))
    prod_idx = np.array(prod_idx)

    reg_names = list(REGIONS)
    reg_w = np.array(list(REGIONS.values()))
    cust_w = customers["_weight"].to_numpy()

    qty = rng.integers(1, 6, n) + (rng.random(n) < 0.08) * rng.integers(5, 20, n)
    price = np.array([PRODUCTS[i][2] for i in prod_idx])
    discount = rng.choice([0, 0, 0, 0.05, 0.1, 0.15], n)

    df = pd.DataFrame({
        "order_id": [f"ORD-{100000 + i}" for i in range(n)],
        "order_date": pd.to_datetime(dates),
        "region": rng.choice(reg_names, n, p=reg_w / reg_w.sum()),
        "channel": rng.choice(CHANNELS, n, p=[0.5, 0.3, 0.2]),
        "customer_id": rng.choice(customers["customer_id"], n, p=cust_w / cust_w.sum()),
        "product": [PRODUCTS[i][0] for i in prod_idx],
        "category": [PRODUCTS[i][1] for i in prod_idx],
        "quantity": qty,
        "unit_price": price,
        "discount": discount,
    })
    df["revenue"] = (df["quantity"] * df["unit_price"] * (1 - df["discount"])).round(2)
    df = df.sort_values("order_date").reset_index(drop=True)

    # ---- injected anomalies (documented in README) -------------------------------------
    big = [400, 1900, 3300]
    df.loc[big, "quantity"] = [120, 95, 150]
    df.loc[big, "product"] = "Laptop Pro 14"
    df.loc[big, "category"] = "Electronics"
    df.loc[big, "unit_price"] = 1299.0
    df.loc[big, "discount"] = 0.0
    df.loc[big, "revenue"] = df.loc[big, "quantity"] * 1299.0

    neg = [777, 2555]
    df.loc[neg, "quantity"] = -df.loc[neg, "quantity"].abs()
    df.loc[neg, "revenue"] = (df.loc[neg, "quantity"] * df.loc[neg, "unit_price"]).round(2)

    df.loc[1234, "unit_price"] = 12.99  # mis-keyed unit price (breaks the fixed product price)
    df.loc[1234, "revenue"] = round(df.loc[1234, "quantity"] * 12.99, 2)

    # flash-sale spike on 2025-07-15
    spike = df.sample(60, random_state=7).copy()
    spike["order_date"] = pd.Timestamp("2025-07-15")
    spike["order_id"] = [f"ORD-{900000 + i}" for i in range(len(spike))]
    df = pd.concat([df, spike])

    df = pd.concat([df, df.iloc[[50, 51, 2000]]])  # duplicate rows
    df.loc[df.sample(25, random_state=3).index, "channel"] = np.nan  # missing values
    return df.sort_values("order_date").reset_index(drop=True)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    customers = make_customers()
    sales = make_sales(customers)
    products = pd.DataFrame(
        [(p[0], p[1], p[2], p[3]) for p in PRODUCTS], columns=["product", "category", "list_price", "unit_cost"]
    )
    sales.to_csv(OUT / "sales.csv", index=False, date_format="%Y-%m-%d")
    customers.drop(columns="_weight").to_csv(OUT / "customers.csv", index=False, date_format="%Y-%m-%d")
    products.to_csv(OUT / "products.csv", index=False)
    print(f"sales: {len(sales)} rows, customers: {len(customers)}, products: {len(products)} -> {OUT}")


if __name__ == "__main__":
    main()
