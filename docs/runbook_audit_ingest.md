 35
 36
 37
 38
 39
 40
 41
 42
 43
 44
 45
 46
 47
 48
 49
 50
 51
 52
 53
 54
 55
 56
 57
 58
 59
 60
 61
 62
 63
 64
 65
 66
 67
 68
 69
 70
 71
 72
 73
 74
 75
 76
 77
 78
 79
 80
 81
```markdown
   - Verify audit_event writes succeed; monitor AEGIS_AUDIT_WRITE_ERRORS metric.

5. Migrate existing logs
   - Run s3_migration.py to upload logs from logs/safety_audit.log to S3 (idempotent).
   - Verify objects appear with correct keys and sizes.

6. Ingestion into OpenSearch/ELK
   - Option A: Deploy Logstash with pipeline.conf (logstash/pipeline.conf) and configure credentials/hosts.
   - Option B: Use Lambda triggered by S3 events to parse objects and index into OpenSearch (for serverless).
   - Option C: Use AWS OpenSearch ingestion tools (Kinesis + Lambda) if on AWS.

7. RBAC & access control
   - OpenSearch: create role(s) for auditors and ops (infra/opensearch_role.json).
   - Limit console / dashboard access via SSO (e.g., Cognito / SAML) and map to OpenSearch roles.

8. Monitoring & alerts
   - Alert if AEGIS_AUDIT_WRITE_ERRORS increases.
   - Alert if ingestion pipeline lag > threshold (S3 objects older than X not indexed).
   - Alert on failed S3 lifecycle transitions or KMS errors.

9. Recovery & forensic copy
   - Keep a separate archival bucket with versioning enabled if retention needs to be extended after an incident.
   - To preserve evidence, copy relevant S3 objects to immutable storage (separate bucket with stricter access).

Operational notes & security best practices
- Do not expose raw audit S3 bucket to wide roles. Provide a small ingestion role and a separate read-only role for auditors.
- Use SSE-KMS with a customer-managed key and restrict key usage to known service principals.
- Use S3 bucket policies to restrict access only to the ingestion service and audit writer.
- For compliance, enable S3 access logging to a separate bucket and set retention for access logs.

Troubleshooting
- "Audit write failed" in app logs:
  - Check IAM credentials / role attached to the instance/pod.
  - Check KMS permissions.
  - Check bucket policy and block public access settings.
- No recent objects in OpenSearch:
  - Check ingestion pipeline logs (Logstash or Lambda).
  - Verify S3 notifications (SNS/SQS/Lambda) are configured correctly.

Appendix: Common commands
- List recent objects:
  aws s3 ls s3://aegis-audit-prod/aegis/audit/2025/12/01/ --recursive
- Get an object:
  aws s3 cp s3://aegis-audit-prod/aegis/audit/2025/12/01/167xxx-abc.json .
- Apply lifecycle via CLI:
  aws s3api put-bucket-lifecycle-configuration --bucket aegis-audit-prod --lifecycle-configuration file://infra/s3_lifecycle.xml
```
