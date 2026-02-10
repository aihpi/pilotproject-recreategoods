# diffsynth/curriculum.py
import numpy as np
from torch.utils.data import Sampler

def build_curriculum_pools(df, tier_col="tier", difficulty_col="difficulty", sort_within_tier=True):
    """
    Return dict tier -> np.array of row indices. Optionally sort by difficulty inside each tier.

    Args:
        df: DataFrame with tier and difficulty columns
        tier_col: Name of tier column (default: "tier")
        difficulty_col: Name of difficulty column (default: "difficulty")
        sort_within_tier: Whether to sort samples by difficulty within each tier

    Returns:
        dict: {tier: np.array of indices} mapping tiers to sample indices
    """
    tiers = sorted(df[tier_col].astype(int).unique())
    pools = {}
    for t in tiers:
        idxs = df.index[df[tier_col] == t].to_numpy()
        if sort_within_tier and difficulty_col in df.columns:
            # stable increasing: easiest→hardest within the tier
            diffs = df.loc[idxs, difficulty_col].to_numpy()
            idxs = idxs[np.argsort(diffs)]
        pools[t] = idxs
    return pools

class CurriculumSampler(Sampler[int]):
    """
    Single-GPU curriculum sampler. Call set_epoch(epoch) each epoch.

    This sampler creates batches with configurable tier mixing ratios that evolve
    over training epochs according to a predefined schedule.
    """
    def __init__(self, pools, schedule, batch_size, steps_per_epoch, seed=42):
        """
        Args:
            pools: dict {tier: np.array of indices} from build_curriculum_pools
            schedule: list of dicts with "epochs" and "mix" keys
                     e.g., [{"epochs": 1, "mix": {0: 1.0, 1: 0.0}}, ...]
            batch_size: samples per batch
            steps_per_epoch: number of batches per epoch
            seed: random seed for reproducibility
        """
        super().__init__(None)
        self.pools = {t: p.copy() for t, p in pools.items()}
        self.batch_size = int(batch_size)
        self.steps_per_epoch = int(steps_per_epoch)
        self.rng = np.random.default_rng(seed)

        # Build epoch-to-mix mapping from schedule
        mixes = []
        for ph in schedule:
            mixes += [ph["mix"]] * ph["epochs"]
        self.mixes_by_epoch = mixes
        self.epoch = 0
        self.mix = self.mixes_by_epoch[0] if mixes else {0: 1.0}

    def set_epoch(self, epoch):
        """Update the tier mixing ratios for the given epoch."""
        self.epoch = int(epoch)
        if self.epoch < len(self.mixes_by_epoch):
            self.mix = self.mixes_by_epoch[self.epoch]
        else:
            # Use the last defined mix for epochs beyond the schedule
            self.mix = self.mixes_by_epoch[-1]

    def set_mix_dict(self, mix: dict):
        """Override the current tier mix on-the-fly, e.g. when advancing a phase."""
        self.mix = {int(k): float(v) for k, v in mix.items()}

    def __iter__(self):
        tier_list = sorted(self.pools.keys())
        probs = np.array([self.mix.get(t, 0.0) for t in tier_list], dtype=float)

        # Normalize probabilities to sum to 1
        if probs.sum() > 0:
            probs = probs / probs.sum()
        else:
            # Fallback: equal probability for all tiers
            probs = np.ones(len(tier_list)) / len(tier_list)

        # Calculate samples per tier per batch
        per_batch = np.floor(probs * self.batch_size).astype(int)

        # Distribute remaining samples to maintain exact batch size
        while per_batch.sum() < self.batch_size:
            remaining_prob = probs - per_batch / self.batch_size
            per_batch[np.argmax(remaining_prob)] += 1

        # Shuffle each tier once per epoch for variety
        for t in tier_list:
            self.rng.shuffle(self.pools[t])
        ptr = {t: 0 for t in tier_list}
        n = {t: len(self.pools[t]) for t in tier_list}

        for _ in range(self.steps_per_epoch):
            idxs = []
            for t, m in zip(tier_list, per_batch):
                if m == 0:
                    continue
                # Wrap around if we run out of samples in this tier
                if ptr[t] + m > n[t]:
                    self.rng.shuffle(self.pools[t])
                    ptr[t] = 0
                idxs.extend(self.pools[t][ptr[t]:ptr[t]+m].tolist())
                ptr[t] += m

            # Shuffle the batch to mix tiers randomly
            self.rng.shuffle(idxs)
            for i in idxs:
                yield i

    def __len__(self):
        return self.steps_per_epoch * self.batch_size

class DistributedCurriculumSampler(Sampler[int]):
    """
    DDP-friendly sampler: shards the curriculum batches by rank.

    This ensures each GPU gets a consistent portion of the curriculum
    while maintaining the overall tier mixing ratios.
    """
    def __init__(self, pools, schedule, batch_size, steps_per_epoch, world_size, rank, seed=42):
        """
        Args:
            pools: dict {tier: np.array of indices} from build_curriculum_pools
            schedule: curriculum schedule (same format as CurriculumSampler)
            batch_size: global batch size (must be divisible by world_size)
            steps_per_epoch: number of batches per epoch
            world_size: number of distributed processes
            rank: current process rank
            seed: random seed for reproducibility
        """
        super().__init__(None)
        # Note: Removed divisibility constraint - sample-level sharding works with any batch size
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.inner = CurriculumSampler(pools, schedule, batch_size, steps_per_epoch, seed)

    def set_epoch(self, epoch):
        """Update the tier mixing ratios for the given epoch."""
        self.inner.set_epoch(epoch)

    def set_mix_dict(self, mix: dict):
        """Override the current tier mix on-the-fly, e.g. when advancing a phase."""
        self.inner.set_mix_dict(mix)

    def __iter__(self):
        """Iterate through samples assigned to this rank."""
        # Take the global batch indices, then slice per rank
        shard = self.rank

        # Consume in chunks of global batch_size; then yield every stride-th element for this rank
        it = iter(self.inner)
        # The inner iterator yields sample indices one-by-one; we shard by taking every stride-th element
        for i, idx in enumerate(it):
            if (i % self.world_size) == shard:
                yield idx

    def __len__(self):
        """Return number of samples for this rank."""
        # Each rank sees 1/world_size of samples
        return len(self.inner)

def parse_curriculum_schedule(schedule_str):
    """
    Parse curriculum schedule string into list of phase dicts.

    Format: "epochs:p0,p1,p2,p3;epochs:p0,p1,p2,p3;..."
    Example: "1:1.0,0.0,0.0,0.0;1:0.7,0.3,0.0,0.0;999:0.2,0.4,0.4,0.0"

    Args:
        schedule_str: Schedule specification string

    Returns:
        list: List of phase dictionaries with "epochs" and "mix" keys
    """
    phases = []
    for chunk in schedule_str.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        epochs_str, mix_str = chunk.split(":")
        epochs = int(epochs_str)
        probs = [float(x) for x in mix_str.split(",")]
        mix = {i: p for i, p in enumerate(probs)}
        phases.append({"epochs": epochs, "mix": mix})
    return phases


class EarlyStopCurriculum:
    """
    Moves through a list of phase mixes when val metric (LPIPS; lower is better)
    stops improving. Early-stops at the final phase.

    Args:
      phases: list of dicts {tier:prob}, e.g. [{0:1.0},{0:0.7,1:0.3},...]
      patience: epochs without sufficient improvement before advancing
      min_delta_rel: relative improvement required to reset patience (e.g., 0.005 = 0.5%)
      ema_alpha: smoothing for noisy val metrics (0..1), higher = smoother
    """
    def __init__(self, phases, patience=2, min_delta_rel=0.005, ema_alpha=0.6):
        self.phases = phases
        self.patience = patience
        self.min_delta_rel = min_delta_rel
        self.alpha = ema_alpha
        self.phase_idx = 0
        self.best = None
        self.ema = None
        self.bad_epochs = 0

    def current_mix(self):
        return self.phases[self.phase_idx]

    def step(self, val_metric):
        # Smooth
        self.ema = val_metric if self.ema is None else (self.alpha*self.ema + (1-self.alpha)*val_metric)
        m = self.ema
        # Initialize best
        if self.best is None:
            self.best = m
            self.bad_epochs = 0
            return {"action": "stay", "phase": self.phase_idx, "best": self.best, "ema": self.ema}

        # Improvement? (lower is better)
        rel_impr = (self.best - m) / max(1e-8, abs(self.best))
        if rel_impr >= self.min_delta_rel:
            self.best = m
            self.bad_epochs = 0
            return {"action": "stay", "phase": self.phase_idx, "best": self.best, "ema": self.ema}

        # No sufficient improvement
        self.bad_epochs += 1
        if self.bad_epochs >= self.patience:
            if self.phase_idx < len(self.phases) - 1:
                # advance phase
                self.phase_idx += 1
                self.bad_epochs = 0
                self.best = None
                self.ema = None
                return {"action": "advance", "phase": self.phase_idx}
            else:
                # final phase: early stop
                return {"action": "stop", "phase": self.phase_idx}

        return {"action": "stay", "phase": self.phase_idx, "best": self.best, "ema": self.ema}