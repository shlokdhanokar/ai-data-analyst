import pytest

from analyst.data_loader import DataStore, DataValidationError, UnsafeQueryError, load_csv, sanitize_identifier


def test_loads_and_normalises_columns():
    t = load_csv(b"Order Date,Total Sales ($),Region\n2024-01-05,\"$1,200.50\",North\n2024-02-01,$300,South\n", "My Sales.csv")
    assert t.name == "my_sales"
    assert list(t.df.columns) == ["order_date", "total_sales", "region"]
    assert t.df["total_sales"].tolist() == [1200.5, 300.0]
    assert str(t.df["order_date"].dtype).startswith("datetime64")
    assert any("snake_case" in w for w in t.warnings)


def test_semicolon_delimiter_and_latin1():
    data = "name;city;amount\nJosé;Málaga;10\nAna;Sevilla;20\n".encode("latin-1")
    t = load_csv(data, "people.csv")
    assert t.df.shape == (2, 3)
    assert t.df["amount"].sum() == 30


@pytest.mark.parametrize("data,name,msg", [
    (b"", "empty.csv", "empty"),
    (b"a,b\n1,2", "data.xlsx", "not a CSV"),
    (b"a,b\n", "headers_only.csv", "no data"),
])
def test_rejects_invalid_files(data, name, msg):
    with pytest.raises(DataValidationError, match=msg):
        load_csv(data, name)


def test_rejects_oversized_file():
    with pytest.raises(DataValidationError, match="limit"):
        load_csv(b"a\n" + b"1\n" * 600_000, "big.csv", max_mb=1)


def test_duplicate_columns_and_table_names_are_deduped():
    t = load_csv(b"x,X,x \n1,2,3\n", "t.csv", existing={"t"})
    assert t.name == "t_2"
    assert list(t.df.columns) == ["x", "x_1", "x_2"]


def test_sanitize_identifier():
    assert sanitize_identifier("2024 Sales!") == "col_2024_sales"
    assert sanitize_identifier("   ") == "col"


def test_store_query_and_fingerprint(store):
    before = store.fingerprint
    df = store.query("SELECT region, SUM(revenue) AS r FROM sales GROUP BY 1 ORDER BY 2 DESC LIMIT 1")
    assert df.iloc[0]["region"] == "East"
    store.remove("sales")
    assert store.fingerprint != before


@pytest.mark.parametrize("sql", [
    "DROP TABLE sales",
    "SELECT 1; DROP TABLE sales",
    "INSERT INTO sales VALUES (1)",
    "SELECT * FROM read_csv('/etc/passwd')",
    "COPY sales TO 'out.csv'",
])
def test_store_blocks_unsafe_sql(store, sql):
    with pytest.raises((UnsafeQueryError, ValueError)):
        store.query(sql)


def test_store_bad_column_gives_helpful_error(store):
    with pytest.raises(ValueError, match="SQL error"):
        store.query("SELECT nope FROM sales")


def test_schema_summary_mentions_tables_and_types(store):
    text = store.schema_summary()
    assert "TABLE sales" in text and "revenue: float" in text and "order_date: datetime" in text


def test_empty_store():
    assert DataStore().schema_summary() == "No tables loaded."
