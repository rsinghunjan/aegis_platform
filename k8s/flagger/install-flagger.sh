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
#!/usr/bin/env bash
# Quick setup notes to install Flagger + nginx provider + Prometheus (Helm)
# Run on a machine with kubectl configured for your cluster.

set -euo pipefail

# 1) Install Prometheus (if not present)
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
helm upgrade --install prometheus prometheus-community/prometheus --namespace monitoring --create-namespace

# 2) Install NGINX Ingress (if not present)
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm repo update
helm upgrade --install ingress-nginx ingress-nginx/ingress-nginx --namespace ingress-nginx --create-namespace

# 3) Install Flagger + CRDs
helm repo add flagger https://flagger.app
helm repo update
helm upgrade -i flagger flagger/flagger \
  --namespace flagger-system --create-namespace \
  --set prometheus.install=false \
  --set meshProvider=nginx

echo "Flagger installed. Confirm Prometheus is reachable by Flagger (service.monitoring/prometheus)."
echo "Adjust Flagger config if your Prometheus service name/namespace differs."
k8s/flagger/install-flagger.sh
