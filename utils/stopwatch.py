import time
from typing import List


class Stopwatch:
    def __init__(self):
        self._lap_times: List[float] = []
        self._names: List[str] = []
        self._total = 0.0

    def start(self):
        self._lap_times = [time.perf_counter()]
        self._total = 0
        self._names = ["Start"]

    def reset(self):
        self._lap_times = []
        self._names = []
        self._total = 0.0

    def total(self):
        return self._total

    def print_intervals_ms(self):
        print(f"Total time {self.total()} seconds")
        for idx in range(len(self._names) - 1):
            diff = self._lap_times[idx + 1] - self._lap_times[idx]
            percent = (diff / self._total) * 100
            print(
                f"{self._names[idx]} to {self._names[idx + 1]}: {diff * 1000:.2f}ms, {percent:.3f}%"
            )

    def lap(self, name, print_time=False):
        self._lap_times.append(time.perf_counter())
        diff = self._lap_times[-1] - self._lap_times[-2]
        self._total += diff
        self._names.append(name)

        if print_time:
            percent = (diff / self._total) * 100
            print(
                f"{self._names[-2]} to {self._names[-1]}: {diff * 1000:.2f}ms, {percent:.3f}%"
            )
        return
