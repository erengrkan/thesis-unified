# Bandit-CBO Unified: A Hybrid Cost-Based Optimizer for Vector Search

This repository implements a **Hybrid Cost-Based Optimizer (CBO)** for filtered vector search, powered by **Contextual Bandits**. It dynamically decides the best execution strategy for a given filter's selectivity, adapting to data changes and distribution shifts over time.

## Overview

In vector search systems, handling metadata filters efficiently is challenging. A filter could be highly selective (matching very few documents) or loosely selective (matching most documents). 
- **Pre-filtering (Bitmap/IDSelector)** is typically faster for low selectivity.
- **Post-filtering** is typically faster and more accurate for high selectivity.

Instead of relying on hardcoded heuristics, this project uses a **Reinforcement Learning** approach (Contextual Bandits) to learn the optimal crossover point.

### Key Components

1. **Contextual Bandit Optimizer (`cbo/`)**: The core decision engine. It uses a Q-table to track the latency and recall trade-offs across different selectivity buckets. It includes:
   - SoftCliff Rewards to penalize low recall while optimizing for latency.
   - Hybrid Drift Protection (Continual Learning & Abrupt Reset) to adapt when the underlying dataset size or data distribution changes.
2. **FAISS Index (`faiss_index.py`)**: A wrapper around FAISS HNSW for high-performance vector search.
3. **Bitmap Index (`bitmap_index.py`)**: A lightweight bitmap indexing structure to quickly resolve metadata filters.
4. **MiniBitmapPredictor (`selectivity_predictor.py`)**: A sampled bitmap index that estimates the selectivity of a filter query in microseconds, providing the state context to the Bandit Optimizer.

## Getting Started

### Prerequisites

Ensure you have Python 3.9+ installed. You can install the required dependencies using:

```bash
pip install -r requirements.txt
```

### Running the Benchmarks

1. **Generate Data / Indexes (Optional)**
   The benchmarks assume you have generated embeddings and metadata. You can use the `generate_671k.py` script to generate a fake large dataset.
   
   ```bash
   python generate_671k.py
   ```

2. **Main Benchmark**
   Evaluates the Optimizer on a static dataset, showing the learning phase and the frozen (exploitation) phase.
   
   ```bash
   python main_benchmark.py
   ```

3. **Incremental Benchmark**
   Evaluates the Optimizer dynamically as the dataset size grows incrementally, testing the hybrid drift and re-learning capabilities.
   
   ```bash
   python incremental_benchmark.py
   ```

### Running Tests

We have dedicated test scripts for verifying the components:

- **Hybrid Drift Test**: Simulates content drift and size drift to verify that the CBO unfreezes correctly.
  ```bash
  python test_hybrid_drift.py --size 200000
  ```

- **Selectivity Predictor Test**: Validates the accuracy (MAE/MAPE) and speed of the sampled `MiniBitmapPredictor` against a full ground-truth bitmap.
  ```bash
  python test_selectivity_predictor.py
  ```

## Architecture Details

- **Learning Mode**: The CBO explores strategies using Epsilon-Greedy or Softmax policies. Once it has enough confidence (visits) and stable boundaries, it "freezes" the crossover point to eliminate routing overhead.
- **Continual Learning**: Even while frozen, a tiny percentage of queries (e.g., 1%) are used to explore the background. If the system detects that the suboptimal strategy has become better (due to content drift), it immediately unfreezes.
- **Abrupt Reset**: If the total corpus size changes dramatically (e.g., >30% growth), the system triggers an abrupt unfreeze to relearn.

## License

This project is part of a thesis study.
