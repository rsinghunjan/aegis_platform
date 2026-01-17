# Multimodal Dataset Manifest (schema)
This manifest describes how HIL/ingested multimodal datasets are stored and referenced.

Fields:
- dataset_id: unique id
- description: short text
- created_at: ISO8601 ts
- modalities: list of modalities (e.g., camera, lidar, imu, audio, gt)
- objects: list of objects stored with keys and URIs
  - id
  - uri (s3:// or gs://)
  - modality
  - timestamp (ISO8601)
  - schema (fields)
- recordings: list of recording sessions
  - session_id
  - device_list
  - sample_rate_hz
  - start_ts, end_ts
  - manifest_uri (reference to this session's manifest)
- provenance: pointer to ingestion evidence and HIL campaign id

Example snippet:
```json
{
 "dataset_id":"multimodal-2026-01-20-veh1",
 "description":"Vehicle pose + imu + camera dataset for SSM and fusion",
 "created_at":"2026-01-20T12:00:00Z",
 "modalities":["camera","imu","gt"],
 "recordings":[
   {"session_id":"sess-001","device_list":["cam-front","imu1","mocap"],"sample_rate_hz":50,
    "start_ts":"2026-01-19T08:00:00Z","end_ts":"2026-01-19T08:05:00Z","manifest_uri":"s3://aegis-datasets/.../sess-001.json"}
 ]
}
