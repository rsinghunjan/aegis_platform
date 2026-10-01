  1
  2
  3
  4
  5
  6
  7
  8
  9
 10
 11
 12
 13
 14
 15
 16
 17
 18
 19
 20
 21
 22
 23
 24
 25
 26
 27
 28
 29
 30
"""
Feast feature definitions for the Aegis demo (S3 offline store + Redis online store).

Entity: user_id
FeatureView: user_features
"""
from datetime import timedelta
from feast import Entity, Feature, FeatureView, ValueType, FileSource

# FileSource now points to S3 path (created by ingest_to_s3.py)
user_events = FileSource(
    path="s3://feast-offline/aegis/datasets/user_events/latest/user_events.parquet",
    event_timestamp_column="event_ts",
    created_timestamp_column="created_ts",
)

user = Entity(name="user_id", value_type=ValueType.INT64, description="User id")

user_features_view = FeatureView(
    name="user_features",
    entities=["user_id"],
    ttl=timedelta(days=7),
    features=[
        Feature(name="avg_session_length", dtype=ValueType.FLOAT),
        Feature(name="last_purchase_amount", dtype=ValueType.FLOAT),
        Feature(name="purchase_count_30d", dtype=ValueType.INT32),
    ],
    online=True,
    input=user_events,
)
feast/feature_repo/features.py
