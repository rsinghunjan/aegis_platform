"""Warehouse connector abstraction (BigQuery, Snowflake, DuckDB, Trino)."""
from __future__ import annotations

import abc
from typing import Any, Dict, List, Optional


class WarehouseConnector(abc.ABC):
    name: str = "warehouse"

    @abc.abstractmethod
    def execute(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Run a SQL query and return rows as a list of dicts."""
        raise NotImplementedError

    def is_available(self) -> bool:
        return True


class DuckDBWarehouse(WarehouseConnector):
    """DuckDB-backed warehouse connector.

    DuckDB is an embedded OLAP engine with no external services required,
    making it the practical default for local development, tests, and
    small-scale deployments while sharing an interface with hosted
    warehouses (BigQuery, Snowflake, Trino).
    """

    name = "duckdb"

    def __init__(self, database: str = ":memory:"):
        self.database = database
        self._connection = None

    def is_available(self) -> bool:
        try:
            import duckdb  # noqa: F401
        except ImportError:
            return False
        return True

    def _conn(self):
        if self._connection is None:
            import duckdb

            self._connection = duckdb.connect(self.database)
        return self._connection

    def execute(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        if not self.is_available():
            raise RuntimeError("duckdb is not installed")
        conn = self._conn()
        cursor = conn.execute(query, params or {})
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None


class BigQueryWarehouse(WarehouseConnector):
    name = "bigquery"

    def __init__(self, project: str, dataset: Optional[str] = None):
        self.project = project
        self.dataset = dataset

    def is_available(self) -> bool:
        try:
            from google.cloud import bigquery  # noqa: F401
        except ImportError:
            return False
        return True

    def execute(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        if not self.is_available():
            raise RuntimeError("google-cloud-bigquery is not installed")
        from google.cloud import bigquery

        client = bigquery.Client(project=self.project)
        job = client.query(query)
        return [dict(row.items()) for row in job.result()]


class SnowflakeWarehouse(WarehouseConnector):
    name = "snowflake"

    def __init__(self, account: str, user: str, password: str, warehouse: str, database: str):
        self.account = account
        self.user = user
        self.password = password
        self.warehouse = warehouse
        self.database = database

    def is_available(self) -> bool:
        try:
            import snowflake.connector  # noqa: F401
        except ImportError:
            return False
        return True

    def execute(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        if not self.is_available():
            raise RuntimeError("snowflake-connector-python is not installed")
        import snowflake.connector

        conn = snowflake.connector.connect(
            account=self.account,
            user=self.user,
            password=self.password,
            database=self.database,
        )
        try:
            cursor = conn.cursor(snowflake.connector.DictCursor)
            cursor.execute(query, params or {})
            return cursor.fetchall()
        finally:
            conn.close()
