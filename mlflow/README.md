# MLflow Integration for Aegis

This folder provides MLflow integration for the Aegis multimodal AI system, including:
- MLflow server deployment (local and Kubernetes)
- Training scripts with MLflow logging
- Model artifact packaging and Vault transit signing
- CI/CD workflows for experiment tracking and model promotion

## Prerequisites

- Docker & Docker Compose (v2 plugin preferred)
- Python 3.9+
- HashiCorp Vault (for artifact signing)
- Kubernetes cluster (for production deployment)

## Quick Start (Local Development)

### 1. Start the MLflow Stack

```bash
docker compose -f docker/docker-compose.mlflow.yml up -d
```

Wait for services:
- MLflow UI: http://localhost:5000
- MinIO console: http://localhost:9000 (user/minioadmin)

### 2. Environment Variables

Set the following environment variables for training scripts:

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin
export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
```

## Kubernetes Deployment

Deploy MLflow to Kubernetes using the provided manifest:

```bash
kubectl apply -f mlflow/k8s/mlflow-deployment.yaml
```

**Placeholders to replace:**
- `REPLACE_WITH_BUCKET` - S3 bucket for artifact storage

## Training Scripts

### DeepSpeed Training with MLflow

```bash
python training/train_deepspeed.py \
  --out-dir ./model_registry/demo-models/cifar_deepspeed/0.1 \
  --max-epochs 3
```

### Select Best Run from Experiment

```bash
python scripts/select_best_mlflow_run.py \
  --experiment-name aegis-demo \
  --metric val/accuracy \
  --maximize
```

## Distributed Training with Argo Workflows

Submit a distributed DeepSpeed training job:

```bash
argo submit argo/workflows/distributed_training_deepspeed.yaml \
  -p experiment-name=aegis-demo \
  -p mlflow-tracking-uri=http://mlflow.aegis.svc.cluster.local:5000 \
  -p object-store-bucket=your-bucket-name
```

## Model Packaging and Signing

### Vault Transit Signing

The system uses HashiCorp Vault transit engine for artifact signing:

```bash
./scripts/package_and_sign_vault.sh ./model_dir ./output.tar.gz
```

**Required environment variables:**
- `VAULT_ADDR` - Vault server address
- `VAULT_TOKEN` - Vault authentication token

### GitHub Actions Workflow

Use the `mlflow_pack_sign.yml` workflow for CI/CD:

```yaml
# Trigger manually or via workflow_call
uses: ./.github/workflows/mlflow_pack_sign.yml
with:
  experiment-name: aegis-demo
  metric: train/dummy_loss
  selection-mode: minimize
```

## SageMaker Integration

For AWS SageMaker training jobs, use the adapter:

```bash
python adapters/sagemaker/capture_and_sign.py \
  --model-dir /opt/ml/model \
  --output-bucket s3://your-bucket/model-artifacts \
  --vault-addr https://vault.example.com \
  --vault-key aegis-cosign
```

## Configuration Placeholders

The following placeholders should be replaced with actual values:

| Placeholder | Description |
|------------|-------------|
| `REPLACE_WITH_BUCKET` | S3 bucket name for model artifacts |
| `VAULT_ADDR` | HashiCorp Vault server address |
| `VAULT_AUDIENCE` | JWT audience for Vault OIDC authentication |

## Testing

### Test Plan

1. **Deploy MLflow**: Start local stack or deploy to Kubernetes
2. **Run Training**: Execute DeepSpeed training script
3. **Verify Logging**: Check MLflow UI for logged metrics/artifacts
4. **Run Argo Workflow**: Submit workflow in staging environment
5. **Verify Artifacts**: Check S3 bucket for packaged artifacts and signatures
6. **Run GitHub Action**: Trigger `mlflow_pack_sign.yml` manually
7. **Verify Signatures**: Confirm signature verification passes

## Related Files

- `argo/workflows/distributed_training_deepspeed.yaml` - Argo workflow for distributed training
- `training/train_deepspeed.py` - DeepSpeed training script
- `scripts/select_best_mlflow_run.py` - MLflow experiment query tool
- `scripts/package_and_sign_vault.sh` - Vault transit signing script
- `conversion/convert_to_onnx.py` - ONNX model conversion
- `adapters/sagemaker/capture_and_sign.py` - SageMaker integration
