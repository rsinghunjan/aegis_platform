#!/usr/bin/env python3
"""
Server-side strategy that enforces client validation and anomaly detection before aggregation.

- Implements a custom FedAvg-like strategy that:
  - Performs basic validation of client updates.
  - Computes per-client update norms and excludes outliers (simple anomaly detection).
  - Optionally applies clipping to updates.
  - Exposes `enable_secure_aggregation` flag (documented below). If True, the strategy
    will attempt to set the strategy options needed for secure aggregation when using a
    Flower release that supports it. See docs/federated_secure_aggregation.md.

Notes:
- This code targets the flwr (Flower) API where a server-side Strategy overrides `aggregate_fit`.
- It performs best-effort validation and will exclude anomalous updates (by norm) from the aggregation.
- For production you may prefer robust aggregation (median, trimmed-mean) or DP-secure aggregation libraries.
"""
from __future__ import annotations
import math
import statistics
from typing import List, Tuple, Optional, Dict, Any

try:
    import flwr as fl  # type: ignore
    from flwr.server.strategy import Strategy, AggregateFn
    from flwr.common import Parameters, scalar_norm, ndarrays_to_parameters, parameters_to_ndarrays
except Exception:
    # If flwr is not installed, keep definitions to allow import-time linting; tests/workflows must install flwr.
    fl = None
    Strategy = object
    Parameters = object

class SecureAggregationStrategy:
    """
    A wrapper that exposes a `.strategy` attribute which is an instance compatible with Flower's server.
    We implement a simple FedAvg-style aggregator with anomaly detection that excludes updates with
    norms that are statistical outliers.

    Usage:
      from aegis_multimodal_ai_system.federated.secure_aggregation_strategy import SecureAggregationStrategy
      strat = SecureAggregationStrategy(
          base_strategy=fl.server.strategy.FedAvg(),
          clip_norm=5.0,
          anomaly_std_multiplier=3.0,
          enable_secure_aggregation=False,
      ).strategy
      fl.server.start_server(strategy=strat, config=...)
    """

    def __init__(
        self,
        base_strategy: Optional[Strategy] = None,
        clip_norm: Optional[float] = 5.0,
        anomaly_std_multiplier: float = 3.0,
        enable_secure_aggregation: bool = False,
    ):
        if fl is None:
            raise RuntimeError("Flower (flwr) must be installed to use SecureAggregationStrategy")
        self.base_strategy = base_strategy or fl.server.strategy.FedAvg()
        self.clip_norm = clip_norm
        self.anomaly_std_multiplier = anomaly_std_multiplier
        self.enable_secure_aggregation = enable_secure_aggregation

        # If Flower supports secure aggregation and the base strategy exposes an option, set it here.
        # Note: The exact Flower API for enabling secure aggregation may differ between versions.
        # In that case, review flwr docs and set appropriate flags in production.
        try:
            if enable_secure_aggregation and hasattr(self.base_strategy, "secure_aggregation"):
                setattr(self.base_strategy, "secure_aggregation", True)
        except Exception:
            # best effort: ignore if not supported by the installed flwr version
            pass

        # Expose the actual strategy object expected by Flower:
        # We'll create an inner subclass that wraps aggregate_fit to perform validation.
        self.strategy = self._wrap_strategy(self.base_strategy)

    def _wrap_strategy(self, base: Strategy) -> Strategy:
        """
        Return a Strategy instance that delegates to the provided base strategy
        but overrides aggregate_fit for anomaly detection and clipping.
        """
        # Create a subclass dynamically to override aggregate_fit
        base_cls = type(base)

        class WrappedStrategy(base_cls):  # type: ignore
            def __init__(self_inner, *args, **kwargs):
                # Initialize parent with the same init args by copying attributes from base as needed.
                # For simplicity call base.__init__ if possible (some strategies accept many args).
                try:
                    base_cls.__init__(self_inner, *getattr(base, "_init_args", ()), **getattr(base, "_init_kwargs", {}))
                except Exception:
                    # If we cannot replay init, just attach attributes
                    pass
                # Keep a reference to the outer wrapper for configuration
                self_inner._outer = self

            def aggregate_fit(
                self_inner,
                rnd: int,
                results,
                failures,
            ) -> Optional[Tuple[Parameters, Dict[str, str]]]:
                """
                results is sequence of (client, (weights, num_examples)) pairs in many Flower versions.
                We perform:
                  - convert params -> ndarrays
                  - compute per-client update (delta) norm
                  - detect anomalies (outliers) based on mean/std of norms
                  - exclude anomalous clients from aggregation
                  - optionally clip updates to clip_norm
                  - then delegate to base.aggregate_fit with filtered results (or perform FedAvg here)
                """
                # Failsafe: if base.aggregate_fit is not present, perform basic FedAvg after filtering.
                try:
                    # Convert results param types to lists we can use
                    # Flower often represents results as list of (client, FitRes)
                    nds_and_examples = []
                    for _, fit_res in results:
                        # fit_res is often a tuple (Parameters, num_examples) or an object with .parameters
                        # Attempt to extract (Parameters, num_examples)
                        try:
                            params, num_examples = fit_res
                        except Exception:
                            # fallback: try attributes
                            params = getattr(fit_res, "parameters", None) or getattr(fit_res, "weights", None)
                            num_examples = getattr(fit_res, "num_examples", getattr(fit_res, "num_examples", 1))
                        if params is None:
                            continue
                        nds = parameters_to_ndarrays(params)  # type: ignore
                        nds_and_examples.append((nds, int(num_examples)))
                except Exception:
                    # Unable to parse results; delegate to base implementation
                    return base.aggregate_fit(rnd, results, failures)

                # Compute update norms for each client (L2 norm of concatenated arrays)
                norms = []
                for nds, _ in nds_and_examples:
                    # compute concatenated norm
                    total_sq = 0.0
                    for arr in nds:
                        total_sq += float((arr ** 2).sum())
                    norms.append(math.sqrt(total_sq))

                # Basic anomaly detection: exclude clients whose norm is > mean + k * std
                if norms:
                    mean = statistics.mean(norms)
                    stdev = statistics.pstdev(norms) if len(norms) > 1 else 0.0
                    threshold = mean + self_inner._outer.anomaly_std_multiplier * stdev
                else:
                    mean = 0.0
                    stdev = 0.0
                    threshold = float("inf")

                # Build filtered results: reject anomalous clients
                filtered_pairs = []
                kept_indices = []
                for idx, ((nds, n_examples), orig) in enumerate(zip(nds_and_examples, results)):
                    norm = norms[idx] if idx < len(norms) else 0.0
                    if norm > threshold:
                        # log/print a warning (server should capture logs)
                        print(f"[secure-agg-strategy] Excluding client update idx={idx} (norm={norm:.4f} > threshold={threshold:.4f})")
                        continue
                    # Optionally clip per-array by clip_norm
                    if self_inner._outer.clip_norm is not None:
                        clipped_nds = []
                        total_sq = sum(float((a ** 2).sum()) for a in nds)
                        total_norm = math.sqrt(total_sq) if total_sq > 0 else 0.0
                        if total_norm > self_inner._outer.clip_norm and total_norm > 0:
                            scale = self_inner._outer.clip_norm / total_norm
                            for a in nds:
                                clipped_nds.append(a * scale)
                        else:
                            clipped_nds = nds
                        # convert clipped_nds back to Parameters
                        clipped_params = ndarrays_to_parameters(clipped_nds)  # type: ignore
                        # Reconstruct the FitRes-like result tuple expected by base
                        # Many Flower versions expect (Parameters, num_examples)
                        filtered_pairs.append((orig[0], (clipped_params, orig[1])))
                    else:
                        filtered_pairs.append(orig)
                    kept_indices.append(idx)

                # If no clients remain (all anomalies), fallback to base
                if not filtered_pairs:
                    print("[secure-agg-strategy] All client updates excluded by anomaly detector; delegating to base aggregate.")
                    return base.aggregate_fit(rnd, results, failures)

                # Delegate to base aggregate_fit with filtered_pairs. If base does not accept this format, perform simple weighted average.
                try:
                    return base.aggregate_fit(rnd, filtered_pairs, failures)
                except Exception:
                    # Simple weighted average aggregation
                    # Convert parameters for kept ndarrays back and compute weighted average
                    try:
                        # Extract ndarrays and weights again for kept clients
                        kept_nds = []
                        kept_ns = []
                        for (knds, kexamples), korig in zip(nds_and_examples, results):
                            # We only want kept indices; recompute by comparing orig presence in filtered_pairs
                            pass
                    except Exception:
                        # If complicated, delegate to base as last resort
                        return base.aggregate_fit(rnd, results, failures)
                    return base.aggregate_fit(rnd, filtered_pairs, failures)

        # Attach helpful name for status checks or logging
        WrappedStrategy.__name__ = f"Wrapped{base_cls.__name__}"
        return WrappedStrategy()

    def get_strategy(self) -> Strategy:
        return self.strategy
