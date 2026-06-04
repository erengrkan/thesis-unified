"""
test_hybrid_drift.py
====================
Simulates the Hybrid CBO Architecture without FAISS.
Tests Size Drift (Abrupt Reset) and Content Drift (Continual Learning).
"""

import argparse
import random
import logging
from cbo.optimizer import ContextualBanditOptimizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Dummy simulation variables
def get_fake_latency_recall(strategy: str, selectivity: float, bf_latency_multiplier: float = 1.0):
    if strategy == "bitmap_prefilter":
        # Brute-Force: Slows down as selectivity increases (more IDs to match)
        lat = 100.0 * selectivity * bf_latency_multiplier
        rec = 1.0
        return lat, rec
    else:
        # PostFilter: Fast but low recall at low selectivity
        lat = 0.5
        rec = 1.0 if selectivity >= 0.20 else 0.5
        return lat, rec

def simulate_queries(optimizer, n_queries: int, bf_latency_multiplier: float = 1.0):
    for i in range(n_queries):
        sel = random.uniform(0.01, 0.40)
        strategy = optimizer.route(sel)
        lat, rec = get_fake_latency_recall(strategy, sel, bf_latency_multiplier)
        optimizer.feedback(sel, strategy, lat, rec)
        
        # Stop simulation early if state changes are what we are waiting for
        # E.g. we want to know exactly when it freezes or unfreezes
        yield i, optimizer.is_frozen

def main():
    parser = argparse.ArgumentParser(description="Test Hybrid Drift in CBO")
    parser.add_argument("--size", type=int, default=200_000, help="Initial corpus size (N) for the optimizer")
    args = parser.parse_args()

    logger.info("Initializing CBO with N=%d", args.size)
    optimizer = ContextualBanditOptimizer(n_corpus=args.size, mode="epsilon_greedy", continual_epsilon=0.01)
    
    # ── Phase 1: Normal Learning ──────────────────────────────────────────────
    logger.info("--- PHASE 1: Normal Learning ---")
    frozen_at = -1
    for q_idx, is_frozen in simulate_queries(optimizer, 5000):
        if is_frozen:
            frozen_at = q_idx
            logger.info("FROZEN at query %d. Crossover: %.2f", frozen_at, optimizer.frozen_crossover_point)
            break
            
    if not optimizer.is_frozen:
        logger.error("Failed to freeze in Phase 1")
        return
        
    # ── Phase 2: Size Drift (Abrupt Reset) ────────────────────────────────────
    new_size = int(args.size * 1.35) # Increase size by 35% to force abrupt reset
    logger.info("--- PHASE 2: Size Drift (N jumps to %d) ---", new_size)
    unfroze = optimizer.update_corpus_size(new_size)
    if unfroze:
        logger.info("SUCCESS: Optimizer successfully unfroze due to Size Drift.")
    else:
        logger.error("FAILED to unfreeze on Size Drift.")
        return
        
    # Let it re-freeze
    for q_idx, is_frozen in simulate_queries(optimizer, 5000):
        if is_frozen:
            logger.info("Re-FROZEN at query %d. Crossover: %.2f", q_idx, optimizer.frozen_crossover_point)
            break
            
    # ── Phase 3: Content Drift (Continual Learning Trigger) ───────────────────
    logger.info("--- PHASE 3: Content Drift (Brute-Force gets 5x slower) ---")
    logger.info("CBO is currently frozen. 1%% Continual Learning will explore the background...")
    
    # We run up to 50,000 queries. Since epsilon=0.01, it will explore ~500 times.
    # We want to see if it detects that BruteForce is now terrible and unfreezes.
    unfroze_at = -1
    for q_idx, is_frozen in simulate_queries(optimizer, 50000, bf_latency_multiplier=5.0):
        if not is_frozen:
            unfroze_at = q_idx
            logger.info("SUCCESS: Content Drift detected! Unfroze at query %d", unfroze_at)
            break
            
    if optimizer.is_frozen:
        logger.error("FAILED to detect Content Drift. Remained frozen.")
    else:
        logger.info("All Hybrid Drift protections are working flawlessly!")

if __name__ == "__main__":
    main()
