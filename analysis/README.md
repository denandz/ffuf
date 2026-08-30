# ffuf Audit Log Analysis Toolkit

Two scripts for analysing ffuf JSON audit logs:

| Script | Purpose |
|---|---|
| `ffuf_correlation.py` | Global Pearson correlation between payload length and response length |
| `ffuf_analyse.py` | Cluster-based response profiling — groups responses into behavioural types |
| `ffuf_lof.py` | Experimental LOF outlier detector | 

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
```

## ffuf_analyse.py — Cluster-based Response Profiling

Reads an ffuf audit log and clusters responses by their measurable attributes:

- **Status code, content length, word / line count, response time**

Each cluster is annotated with **overhead** (content_length − payload_len) and the **payload length range** within the cluster. The ratio of overhead variance to payload variance reveals whether the API reflects payloads.

### How it works

1. **DBSCAN** clusters the standardised feature vectors, automatically finding groups and flagging noise.
2. Noise points (isolated points too far from any cluster) are tagged `⚠ (noise)`.
4. **Overhead analysis**: overhead_std / payload_std < 0.2 → reflective endpoint.

### Usage

```bash
python3 ffuf_analyse.py <audit-log.json> [--samples N] [--min-cluster N] [--eps FLOAT]
```

| Option | Default | Description |
|---|---|---|
| `--samples N` | 5 | Payloads shown per cluster |
| `--min-cluster N` | 3 | DBSCAN min_samples (minimum points to form a cluster) |
| `--eps FLOAT` | 0.5 | DBSCAN eps (distance in standardised feature space) |
