import os
import pickle
import numpy as np
from faiss_index import FAISSIndex
import config

out_dir = config.INDEX_DIR
out_dir.mkdir(parents=True, exist_ok=True)

print("Creating fake 671k dataset...")
N = 671_000
D = 768

embeddings = np.random.randn(N, D).astype(np.float32)
np.save(out_dir / "raw_embeddings.npy", embeddings)
np.save(out_dir / "embeddings.npy", embeddings)

metadatas = [{"main_cat": "Electronics", "overall": 5.0, "brand": "Fake", "verified": "True"} for _ in range(N)]
with open(out_dir / "all_metadatas.pkl", "wb") as f:
    pickle.dump(metadatas, f)

print("Fake dataset created. Run main_benchmark or incremental_benchmark to test!")
