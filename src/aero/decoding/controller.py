from collections import deque
from typing import Deque

class AdaptiveGammaController:
    def __init__(
        self,
        initial_gamma: int = 4,
        min_gamma: int = 1,
        max_gamma: int = 8,
        window_size: int = 5,
        high_threshold: float = 0.75,
        low_threshold: float = 0.40
    ):
        self.initial_gamma = initial_gamma
        self.current_gamma = initial_gamma
        self.min_gamma = min_gamma
        self.max_gamma = max_gamma
        self.window_size = window_size
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.history: Deque[float] = deque(maxlen=window_size)

    def update(self, accepted_count: int, draft_count: int) -> int:
        step_rate = accepted_count / max(1, draft_count)
        self.history.append(step_rate)
        if len(self.history) == self.window_size:
            avg_rate = sum(self.history) / self.window_size
            if avg_rate >= self.high_threshold and self.current_gamma < self.max_gamma:
                self.current_gamma += 1
                self.history.clear()
            elif avg_rate <= self.low_threshold and self.current_gamma > self.min_gamma:
                self.current_gamma -= 1
                self.history.clear()
        return self.current_gamma

    def get_gamma(self) -> int:
        return self.current_gamma

    def reset(self) -> None:
        self.current_gamma = self.initial_gamma
        self.history.clear()
