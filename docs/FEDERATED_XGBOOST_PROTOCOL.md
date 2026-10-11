# TrustGuard-FL Federated XGBoost Protocol

## Scope

This document defines the planned federated experiment over the preserved 21-class CICIoT23 prepared partitions. It does not alter or reinterpret the centralized baseline.

## Real training protocol

The model-training path uses XGBoost's native federated collective communicator with histogram tree construction. Each client process keeps its local rows and constructs a local `DMatrix`. XGBoost's distributed histogram implementation combines the per-worker boosting statistics through the collective communicator while constructing the same global boosting model. The protocol is therefore distributed histogram boosting, not parameter averaging and not averaging serialized trees.

For boosting round $t$, each worker evaluates the current model on local rows and computes the objective's local gradient and Hessian contributions. For each candidate feature/bin and active tree node, the worker contributes sufficient statistics of the form:

$$
G_{j,b} = \sum_{i \in (j,b)} g_i, \qquad H_{j,b} = \sum_{i \in (j,b)} h_i
$$

The global split search uses the aggregated statistics. For a candidate split with left and right statistics, the regularized gain is evaluated from the standard second-order tree objective, including the configured regularization and minimum-child constraints. The selected split and leaf weights are then part of the global boosting state used in round $t+1$.

## What leaves a client

- Collective training messages required by XGBoost's distributed communicator.
- Evaluation summaries and protocol metadata required for coordination.
- For the readable demonstration only: bounded, non-row histogram summaries, cryptographic hashes, and payload sizes.

## What never leaves a client

- Raw CSV rows.
- Plaintext feature matrices.
- Plaintext labels.
- Source-file contents or row identifiers.

## Security boundary

The real XGBoost collective channel is configured with TLS/mTLS certificate paths. The mentor demonstration additionally serializes a bounded statistics envelope and applies AES-256-GCM plus a separate HMAC-SHA-256 key before a local TLS exchange. These are evidence layers with distinct responsibilities:

- AES-256-GCM: confidentiality and authenticated encryption of the envelope.
- HMAC-SHA-256: explicit message integrity/authentication and tamper rejection.
- TLS/mTLS: secure transport and peer authentication.

The demonstration envelope must not be described as the internal wire format of XGBoost unless the implementation proves that integration. Native XGBoost transport behavior and application-layer security evidence remain separately labeled.

## Client partitioning

The available repository has no organization identifiers or raw source files. The first experiment therefore uses deterministic, non-overlapping contiguous shards of the prepared training CSV, records each shard's class distribution, and labels the result as a prepared-partition simulation. It is not evidence of organization-level non-IID behavior.

## Resource plan

CSV input is streamed or bounded per client. At most one client process is intentionally active at a time on the 16 GB host. The monitor records process/system RAM, GPU memory/utilization, CPU, and elapsed time. At 15 GB system RAM the experiment must stop before starting another client or round and preserve the latest checkpoint.

## Scientific limits

The prepared train/test partitions contain exact feature-row overlap. Federated results from these files cannot establish clean unseen-traffic generalization. A raw source dataset and group/time/device split metadata are required for that claim.