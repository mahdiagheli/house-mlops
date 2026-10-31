from collections import defaultdict
from threading import Lock


class RequestMonitor:
    def __init__(self):
        self.total_requests = 0
        self.total_errors = 0
        self.total_latency = 0.0
        self.status_codes = defaultdict(int)
        self.lock = Lock()

    def record_request(self, status_code: int, latency: float):
        with self.lock:
            self.total_requests += 1
            self.total_latency += latency
            self.status_codes[status_code] += 1

            if status_code >= 400:
                self.total_errors += 1

    def get_metrics(self):
        with self.lock:
            avg_latency = (
                self.total_latency / self.total_requests
                if self.total_requests else 0.0
            )

            error_rate = (
                self.total_errors / self.total_requests
                if self.total_requests else 0.0
            )

            return {
                "total_requests": self.total_requests,
                "total_errors": self.total_errors,
                "error_rate": error_rate,
                "average_latency_seconds": avg_latency,
                "status_codes": dict(self.status_codes),
            }


request_monitor = RequestMonitor()
