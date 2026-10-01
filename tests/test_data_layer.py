"""Tests for services/data: lake, warehouse, feature store, quality, catalog, ETL."""
import pytest

from services.data import (
    LocalDataLake,
    DuckDBWarehouse,
    InMemoryFeatureStore,
    MetadataCatalog,
    DatasetMetadata,
    ETLStep,
    ETLPipeline,
    Expectation,
    run_expectations,
)
from services.data.feature_store import FeatureVector
from services.data.quality import expect_not_null, expect_between, expect_in_set


def test_local_data_lake_put_get_list_delete(tmp_path):
    lake = LocalDataLake(str(tmp_path))
    lake.put("raw/file.txt", b"hello")
    assert lake.get("raw/file.txt") == b"hello"
    assert lake.exists("raw/file.txt")
    assert lake.list("raw/") == ["raw/file.txt"]
    lake.delete("raw/file.txt")
    assert not lake.exists("raw/file.txt")


def test_local_data_lake_rejects_path_escape(tmp_path):
    lake = LocalDataLake(str(tmp_path))
    with pytest.raises(ValueError):
        lake.put("../escape.txt", b"nope")


def test_duckdb_warehouse_executes_query():
    warehouse = DuckDBWarehouse()
    if not warehouse.is_available():
        pytest.skip("duckdb not installed")
    rows = warehouse.execute("SELECT 1 AS a, 2 AS b")
    assert rows == [{"a": 1, "b": 2}]


def test_feature_store_online_and_historical_reads():
    store = InMemoryFeatureStore()
    store.write(
        "user_features",
        [
            FeatureVector(entity_id="u1", features={"age": 30}, event_timestamp=1.0),
            FeatureVector(entity_id="u1", features={"age": 31}, event_timestamp=2.0),
            FeatureVector(entity_id="u2", features={"age": 40}, event_timestamp=1.0),
        ],
    )
    online = store.read_online("user_features", ["u1", "u2"])
    assert online["u1"]["age"] == 31
    assert online["u2"]["age"] == 40
    history = store.read_historical("user_features", ["u1"])
    assert len(history) == 2


def test_metadata_catalog_register_search_lineage():
    catalog = MetadataCatalog()
    upstream = catalog.register(DatasetMetadata(urn="urn:raw", name="raw", owner="team-a", tags=("pii",)))
    catalog.register(
        DatasetMetadata(
            urn="urn:curated", name="curated", owner="team-a", tags=("gold",), lineage=("urn:raw",)
        )
    )
    assert catalog.search_by_tag("pii") == [upstream]
    lineage = catalog.upstream_lineage("urn:curated")
    assert lineage == [upstream]


def test_etl_pipeline_runs_in_dependency_order():
    def extract(ctx):
        return [1, 2, 3]

    def transform(ctx):
        return [x * 2 for x in ctx["extract"]]

    def load(ctx):
        return sum(ctx["transform"])

    pipeline = ETLPipeline(
        [
            ETLStep("extract", extract),
            ETLStep("transform", transform, depends_on=("extract",)),
            ETLStep("load", load, depends_on=("transform",)),
        ]
    )
    results = pipeline.run()
    assert all(r.success for r in results.values())
    assert results["load"].output == 12


def test_etl_pipeline_detects_cycles():
    with pytest.raises(ValueError):
        ETLPipeline([ETLStep("a", lambda ctx: None, depends_on=("missing",))])


def test_data_quality_expectations():
    rows = [{"age": 10}, {"age": None}, {"age": 200}]
    report = run_expectations(
        rows,
        [expect_not_null("age"), expect_between("age", 0, 120)],
    )
    assert report.total_rows == 3
    assert not report.success
    by_column = {r.column: r for r in report.results}
    assert by_column["age"].failed_values  # some rows fail both expectations
