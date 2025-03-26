import logging
import os
import signal
import socket
import sys
import time

import psutil
from prometheus_client import Counter, Gauge, start_http_server

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "logs", "metrics_generator.log"
            )
        ),
    ],
)
logger = logging.getLogger(__name__)

# Define metrics
http_requests = Counter("http_requests_total", "Total HTTP requests", ["method", "endpoint"])
request_duration = Gauge("http_request_duration_seconds", "HTTP request duration in seconds")
memory_usage = Gauge("memory_usage_bytes", "Memory usage in bytes")

# Global flag for graceful shutdown
running = True


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    global running
    logger.info(f"Received signal {signum}")
    running = False


signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("", port))
            return False
        except OSError:
            return True


def generate_metrics():
    """Generate sample metrics."""
    global running
    while running:
        try:
            # Simulate HTTP requests
            http_requests.labels(method="GET", endpoint="/api").inc()
            http_requests.labels(method="POST", endpoint="/api").inc(0.5)

            # Update request duration
            request_duration.set(0.15)

            # Update memory usage
            memory = psutil.Process(os.getpid()).memory_info().rss
            memory_usage.set(memory)

            time.sleep(5)
        except Exception as e:
            logger.error(f"Error generating metrics: {e}")
            if not running:  # If we're shutting down, exit the loop
                break
            time.sleep(1)

    logger.info("Metrics generator shutting down")


def main():
    port = 8000
    max_retries = 3
    retry_count = 0

    # Create a PID file
    pid_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "metrics_generator.pid")
    with open(pid_file, "w") as f:
        f.write(str(os.getpid()))

    try:
        while retry_count < max_retries:
            if is_port_in_use(port):
                logger.warning(f"Port {port} is in use. Attempting to kill existing process...")
                try:
                    os.system(f"lsof -ti:{port} | xargs kill -9")
                    time.sleep(2)
                except Exception as e:
                    logger.error(f"Failed to kill process on port {port}: {e}")
                    port += 1
                    retry_count += 1
                    continue

            try:
                start_http_server(port)
                logger.info(f"Starting metrics generator on port {port}")
                logger.info("Prometheus metrics server started")
                generate_metrics()
                break
            except Exception as e:
                logger.error(f"Failed to start metrics server on port {port}: {e}")
                port += 1
                retry_count += 1

        if retry_count >= max_retries:
            logger.error("Failed to start metrics server after maximum retries")
            sys.exit(1)

    finally:
        # Clean up PID file
        try:
            os.remove(pid_file)
        except:
            pass


if __name__ == "__main__":
    main()
