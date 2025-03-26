import json
import logging
import os
import signal
import socket
import sys
import threading
from datetime import datetime

from flask import Flask, jsonify, request

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs", "alert_receiver.log")
        ),
    ],
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global flag for graceful shutdown
running = True


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    global running
    logger.info(f"Received signal {signum}")
    running = False
    # Trigger a request to shut down Flask
    threading.Thread(target=shutdown_server).start()


def shutdown_server():
    """Shutdown the Flask server."""
    import requests

    try:
        requests.get("http://localhost:3456/shutdown")
    except:
        pass


signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("", port))
            s.close()
            return False
        except OSError:
            return True


@app.route("/shutdown", methods=["GET"])
def shutdown():
    """Shutdown endpoint for graceful termination."""
    func = request.environ.get("werkzeug.server.shutdown")
    if func is None:
        raise RuntimeError("Not running with the Werkzeug Server")
    func()
    return "Server shutting down..."


@app.route("/health")
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "timestamp": datetime.utcnow().isoformat()})


@app.route("/alert", methods=["POST"])
def receive_alert():
    """Receive and process alerts from Alertmanager."""
    try:
        data = request.get_json()
        logger.info(f"Received alert data: {json.dumps(data, indent=2)}")

        if not data or "alerts" not in data:
            return jsonify({"status": "error", "message": "Invalid alert format"}), 400

        for alert in data["alerts"]:
            process_alert(alert)

        return jsonify({"status": "success"})

    except Exception as e:
        logger.error(f"Error processing alert: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


def process_alert(alert):
    """Process individual alert data."""
    try:
        # Add empty fields if they don't exist
        alert.setdefault("endsAt", "")
        alert.setdefault("generatorURL", "")

        logger.info(f"Processing alert: {json.dumps(alert, indent=2)}")

        # Log critical alerts with higher severity
        if alert.get("labels", {}).get("severity") == "critical":
            logger.warning(
                f"Critical alert received: {alert['annotations'].get('description', 'No description')}"
            )

        # Here you can add more alert processing logic
        # For example: sending notifications, updating dashboards, etc.

    except Exception as e:
        logger.error(f"Error processing individual alert: {e}")
        raise


def main():
    port = 3456
    max_retries = 3
    retry_count = 0

    # Create a PID file
    pid_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "alert_receiver.pid")
    with open(pid_file, "w") as f:
        f.write(str(os.getpid()))

    try:
        while retry_count < max_retries and running:
            if is_port_in_use(port):
                logger.warning(f"Port {port} is in use. Attempting to kill existing process...")
                try:
                    os.system(f"lsof -ti:{port} | xargs kill -9")
                    import time

                    time.sleep(2)
                except Exception as e:
                    logger.error(f"Failed to kill process on port {port}: {e}")
                    port += 1
                    retry_count += 1
                    continue

            try:
                logger.info(f"Starting alert receiver on port {port}")
                app.run(host="0.0.0.0", port=port, use_reloader=False)
                break
            except Exception as e:
                logger.error(f"Failed to start alert receiver on port {port}: {e}")
                if "Address already in use" in str(e):
                    logger.error(
                        f"Port {port} is in use by another program. Either identify and stop that program, or start the server with a different port."
                    )
                port += 1
                retry_count += 1

        if retry_count >= max_retries:
            logger.error("Failed to start alert receiver after maximum retries")
            sys.exit(1)

    finally:
        # Clean up PID file
        try:
            os.remove(pid_file)
        except:
            pass


if __name__ == "__main__":
    main()
