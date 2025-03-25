import psutil
import requests
import time
import logging
import subprocess
import signal
import os
import sys
import argparse
import atexit
import json
from datetime import datetime
from typing import List, Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs', 'service_manager.log'))
    ]
)
logger = logging.getLogger(__name__)

# Store PIDs file path
PIDS_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'pids.json')

def save_pids(pids):
    """Save PIDs to file."""
    with open(PIDS_FILE, 'w') as f:
        json.dump({
            'pids': pids,
            'timestamp': datetime.now().isoformat()
        }, f)

def load_pids():
    """Load PIDs from file."""
    try:
        with open(PIDS_FILE, 'r') as f:
            data = json.load(f)
            return data.get('pids', [])
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def cleanup_on_exit():
    """Cleanup function to be called on exit."""
    logger.info("Cleaning up services...")
    stop_services()
    if os.path.exists(PIDS_FILE):
        os.remove(PIDS_FILE)

atexit.register(cleanup_on_exit)

class ServiceManager:
    def __init__(self):
        self.services = {
            'metrics_generator': {
                'port': 8000,
                'script': 'metrics_generator.py',
                'health_url': 'http://localhost:8000/metrics',
                'process': None,
                'env': dict(os.environ, PYTHONUNBUFFERED='1', 
                          PYTHONPATH=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            },
            'alert_receiver': {
                'port': 3456,
                'script': 'alert_receiver.py',
                'health_url': 'http://localhost:3456/health',
                'process': None,
                'env': dict(os.environ, PYTHONUNBUFFERED='1',
                          PYTHONPATH=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            }
        }
        self.script_dir = os.path.dirname(os.path.abspath(__file__))

    def check_port_in_use(self, port: int) -> bool:
        """Check if a port is in use."""
        for proc in psutil.process_iter(['pid', 'name', 'connections']):
            try:
                for conn in proc.connections():
                    if conn.laddr.port == port:
                        logger.warning(f"Port {port} is in use by process {proc.pid} ({proc.name()})")
                        return True
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                logger.debug(f"Error checking process: {str(e)}")
                pass
        return False

    def kill_process_on_port(self, port: int) -> None:
        """Kill process using specified port."""
        for proc in psutil.process_iter(['pid', 'name', 'connections']):
            try:
                for conn in proc.connections():
                    if conn.laddr.port == port:
                        logger.info(f"Terminating process {proc.pid} ({proc.name()}) on port {port}")
                        psutil.Process(proc.pid).terminate()
                        time.sleep(1)  # Give process time to terminate
                        if psutil.pid_exists(proc.pid):
                            logger.warning(f"Process {proc.pid} did not terminate, killing it")
                            psutil.Process(proc.pid).kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                logger.error(f"Error killing process: {str(e)}")
                pass

    def start_service(self, service_name: str) -> bool:
        """Start a service and verify it's running."""
        service = self.services[service_name]
        script_path = os.path.join(self.script_dir, service['script'])

        if not os.path.exists(script_path):
            logger.error(f"Script not found: {script_path}")
            return False

        # Kill any existing process on the port
        if self.check_port_in_use(service['port']):
            self.kill_process_on_port(service['port'])

        # Start the service
        try:
            logger.info(f"Starting {service_name} with script {script_path}")
            process = subprocess.Popen(
                [sys.executable, script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=service['env'],
                cwd=self.script_dir
            )
            service['process'] = process
            logger.info(f"Started {service_name} with PID {process.pid}")

            # Wait for service to be healthy
            retries = 5
            while retries > 0:
                try:
                    response = requests.get(service['health_url'], timeout=2)
                    if response.status_code == 200:
                        logger.info(f"{service_name} is healthy")
                        return True
                except requests.exceptions.RequestException as e:
                    logger.warning(f"Health check failed for {service_name}: {str(e)}")
                    # Check if process is still running
                    if process.poll() is not None:
                        stdout, stderr = process.communicate()
                        logger.error(f"{service_name} process exited with code {process.returncode}")
                        logger.error(f"stdout: {stdout.decode() if stdout else 'None'}")
                        logger.error(f"stderr: {stderr.decode() if stderr else 'None'}")
                        return False
                time.sleep(1)
                retries -= 1

            logger.error(f"Failed to verify {service_name} health after {5-retries} attempts")
            return False

        except Exception as e:
            logger.error(f"Error starting {service_name}: {str(e)}")
            return False

    def stop_service(self, service_name: str) -> None:
        """Stop a service."""
        service = self.services[service_name]
        if service['process']:
            try:
                service['process'].terminate()
                try:
                    service['process'].wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning(f"{service_name} did not terminate, killing it")
                    service['process'].kill()
                logger.info(f"Stopped {service_name}")
            except Exception as e:
                logger.error(f"Error stopping {service_name}: {str(e)}")
        if self.check_port_in_use(service['port']):
            self.kill_process_on_port(service['port'])

    def stop_all(self) -> None:
        """Stop all services."""
        for service_name in self.services:
            self.stop_service(service_name)

    def start_all(self) -> bool:
        """Start all services."""
        success = True
        for service_name in self.services:
            if not self.start_service(service_name):
                success = False
                break
        if not success:
            self.stop_all()
        return success

def get_script_path(script_name):
    """Get the absolute path of a script."""
    return os.path.join(os.path.dirname(__file__), script_name)

def is_port_in_use(port):
    """Check if a port is in use."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('', port))
            s.close()
            return False
        except OSError:
            return True

def kill_process_on_port(port):
    """Kill process running on specified port."""
    try:
        output = subprocess.check_output(['lsof', '-ti', f':{port}']).decode().strip()
        if output:
            for pid in output.split('\n'):
                try:
                    pid = int(pid)
                    os.kill(pid, signal.SIGKILL)
                    logger.info(f"Killed process {pid} on port {port}")
                except (ValueError, ProcessLookupError) as e:
                    logger.warning(f"Failed to kill process {pid}: {e}")
            time.sleep(1)  # Wait for ports to be released
    except subprocess.CalledProcessError:
        pass  # No process found on port
    except Exception as e:
        logger.error(f"Error killing process on port {port}: {e}")

def verify_service(name, port, health_url=None, timeout=30):
    """Verify that a service is running properly."""
    import requests
    from urllib.parse import urlparse
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        # First wait for the port to be in use
        if not is_port_in_use(port):
            logger.warning(f"{name} port {port} is not in use, waiting...")
            time.sleep(2)
            continue
            
        # If health URL is provided, check it
        if health_url:
            try:
                response = requests.get(health_url, timeout=5)
                if response.status_code == 200:
                    logger.info(f"{name} is healthy")
                    return True
            except requests.exceptions.RequestException as e:
                logger.warning(f"Health check failed for {name}: {e}")
        else:
            # If no health URL, just verify port is in use
            return True
        
        time.sleep(2)
    
    logger.error(f"{name} failed to start properly within {timeout} seconds")
    return False

def start_services():
    """Start the monitoring services."""
    # Kill any existing processes
    kill_process_on_port(8000)
    kill_process_on_port(3456)
    time.sleep(2)  # Wait for ports to be freed
    
    # Start metrics generator
    metrics_script = get_script_path('metrics_generator.py')
    if not os.path.exists(metrics_script):
        logger.error(f"Metrics generator script not found at {metrics_script}")
        return False

    # Start alert receiver
    alert_script = get_script_path('alert_receiver.py')
    if not os.path.exists(alert_script):
        logger.error(f"Alert receiver script not found at {alert_script}")
        return False

    pids = []
    try:
        # Start metrics generator with nohup
        with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs', 'metrics_generator.log'), 'a') as log:
            metrics_proc = subprocess.Popen(
                [sys.executable, metrics_script],
                stdout=log,
                stderr=log,
                preexec_fn=os.setpgrp
            )
        pids.append(metrics_proc.pid)
        logger.info(f"Started metrics generator with PID {metrics_proc.pid}")
        
        # Start alert receiver with nohup
        with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs', 'alert_receiver.log'), 'a') as log:
            alert_proc = subprocess.Popen(
                [sys.executable, alert_script],
                stdout=log,
                stderr=log,
                preexec_fn=os.setpgrp
            )
        pids.append(alert_proc.pid)
        logger.info(f"Started alert receiver with PID {alert_proc.pid}")
        
        # Save PIDs
        save_pids(pids)
        
        # Give processes time to start
        time.sleep(5)
        
        # Verify services are running
        metrics_ok = verify_service('Metrics Generator', 8000, 'http://localhost:8000/metrics', timeout=30)
        alert_ok = verify_service('Alert Receiver', 3456, 'http://localhost:3456/health', timeout=30)
        
        if metrics_ok and alert_ok:
            logger.info("All services started successfully")
            return True
        else:
            logger.error("Failed to start all services")
            stop_services()
            return False
            
    except Exception as e:
        logger.error(f"Error starting services: {e}")
        stop_services()
        return False

def stop_services():
    """Stop the monitoring services."""
    # Load saved PIDs
    pids = load_pids()
    
    # Kill processes by PID
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
            logger.info(f"Terminated process {pid}")
        except ProcessLookupError:
            logger.info(f"Process {pid} not found")
        except Exception as e:
            logger.error(f"Error stopping process {pid}: {e}")
    
    # Kill any remaining processes on ports
    kill_process_on_port(8000)
    kill_process_on_port(3456)
    
    # Remove PIDs file
    if os.path.exists(PIDS_FILE):
        os.remove(PIDS_FILE)

def main():
    parser = argparse.ArgumentParser(description='Manage monitoring services')
    parser.add_argument('action', choices=['start', 'stop'], help='Action to perform')
    args = parser.parse_args()
    
    if args.action == 'start':
        if start_services():
            # Don't clean up on successful start
            atexit.unregister(cleanup_on_exit)
            sys.exit(0)
        else:
            sys.exit(1)
    elif args.action == 'stop':
        stop_services()
        sys.exit(0)

if __name__ == '__main__':
    main() 