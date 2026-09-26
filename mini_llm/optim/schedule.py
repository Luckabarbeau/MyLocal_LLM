import math


class WarmupCosineSchedule:
    def __init__(self, peak_lr, warmup_steps, total_steps, min_lr_ratio=0.1):
        if not (0 <= warmup_steps < total_steps):
            raise ValueError("Need 0 <= warmup_steps < total_steps.")
        self.peak_lr = float(peak_lr)
        self.warmup_steps = int(warmup_steps)
        self.total_steps = int(total_steps)
        self.min_lr = self.peak_lr * float(min_lr_ratio)

    def __call__(self, step):
        step = int(step)
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return self.peak_lr * (step + 1) / self.warmup_steps

        progress = (
            (step - self.warmup_steps)
            / max(1, self.total_steps - self.warmup_steps - 1)
        )
        progress = min(max(progress, 0.0), 1.0)
        return self.min_lr + 0.5 * (self.peak_lr - self.min_lr) * (
            1.0 + math.cos(math.pi * progress)
        )
