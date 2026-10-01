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
 31
 32
 33
 34
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
```markdown
Training job templates & orchestration — quickstart (Aegis)

What I delivered
- docker/Dockerfile.training : base image to build training image (CPU example; swap to CUDA base image for GPU nodes)
- examples/train_ddp.py : canonical PyTorch training script supporting DDP (use torchrun for distributed launches)
- k8s/job-training.yaml : Kubernetes Job manifest for single-node training
- k8s/pytorchjob-training.yaml : Kubeflow PyTorchJob manifest for distributed training (requires PyTorch operator installed)
- examples/hpo_ray_tune.py : Ray Tune-based HPO example to run hyperparameter search locally or in a cluster
- Documentation describing how to build/run locally and on k8s

Quickstart (local single-node)
1. Build image (optional for local python run):
   docker build -t ghcr.io/your-org/aegis-training:latest -f docker/Dockerfile.training .

2. Run locally (no container):
   python examples/train_ddp.py --epochs 2 --batch-size 64

3. Run with torchrun (simulate 2 GPUs on a single machine with 2 processes):
   torchrun --nproc_per_node=2 examples/train_ddp.py --epochs 3 --batch-size 64 --ddp

Kubernetes (single node)
1. Push training image to registry (e.g., ghcr.io/your-org/aegis-training:latest)
2. kubectl apply -f k8s/job-training.yaml
   - Monitor logs: kubectl logs job/aegis-training-job -f

Kubeflow PyTorchJob (recommended for multi-node GPU clusters)
1. Ensure the Kubeflow PyTorch Operator is installed in your cluster.
2. kubectl apply -f k8s/pytorchjob-training.yaml
3. Monitor pods and logs:
   - kubectl get pytorchjobs
   - kubectl logs -l pytorch-job-name=aegis-pytorchjob --all-containers

Ray Tune HPO (local)
1. pip install ray[tune]
2. python examples/hpo_ray_tune.py
3. Check ray_results/ for trial outputs and best config printed at the end.

Notes & adaptation points
- Image: switch Dockerfile base to a CUDA-enabled base (nvidia/cuda) for real GPU nodes and install matching torch wheels.
- Use S3 or PVC-backed dataset mounts in production clusters to avoid re-downloading datasets each trial.
- For production HPO at scale, consider using Ray on a K8s cluster (Ray operator) or a managed Ray cluster.
- Integrate MLflow tracking by setting MLFLOW_TRACKING_URI env in k8s manifests and enabling mlflow logging in scripts.
- For PyTorchJob, adjust resource requests/limits and the number of worker replicas to match available GPUs/nodes.

Acceptance criteria
- You can run the training script locally and produce a checkpoint artifact.
- You can run torchrun for multi-process DDP on a single machine.
- You can launch the k8s job and see logs from the training container.
- You can create a PyTorchJob and have master/worker pods coordinate training.
- You can run the Ray Tune script locally and see trial results.

Next steps I recommend (pick one)
- Add a CI job that builds the training image and runs a short smoke test using a small dataset.
- Add a PersistentVolumeClaim + k8s manifest for dataset caching to reduce download time in cluster runs.
- Integrate MLflow logging for every training run and register produced model artifacts via the MLflow-to-registry script.
