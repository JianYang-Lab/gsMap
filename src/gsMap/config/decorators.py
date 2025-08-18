"""
Decorators for CLI and resource tracking.
"""

import inspect
import logging
import os
import sys
import threading
import time
from dataclasses import fields
from functools import wraps

import psutil
import pyfiglet

logger = logging.getLogger("gsMap")


def process_cpu_time(proc: psutil.Process):
    """Calculate total CPU time for a process."""
    cpu_times = proc.cpu_times()
    total = cpu_times.user + cpu_times.system
    return total


def track_resource_usage(func):
    """
    Decorator to track resource usage during function execution.
    Logs memory usage, CPU time, and wall clock time at the end of the function.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get the current process
        process = psutil.Process(os.getpid())
        
        # Initialize tracking variables
        peak_memory = 0
        cpu_percent_samples = []
        stop_thread = False
        
        # Function to monitor resource usage
        def resource_monitor():
            nonlocal peak_memory, cpu_percent_samples
            while not stop_thread:
                try:
                    # Get current memory usage in MB
                    current_memory = process.memory_info().rss / (1024 * 1024)
                    peak_memory = max(peak_memory, current_memory)
                    
                    # Get CPU usage percentage
                    cpu_percent = process.cpu_percent(interval=None)
                    if cpu_percent > 0:  # Skip initial zero readings
                        cpu_percent_samples.append(cpu_percent)
                    
                    time.sleep(0.5)
                except Exception:
                    pass
        
        # Start resource monitoring in a separate thread
        monitor_thread = threading.Thread(target=resource_monitor)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        # Get start times
        start_wall_time = time.time()
        start_cpu_time = process_cpu_time(process)
        
        try:
            # Run the actual function
            result = func(*args, **kwargs)
            return result
        finally:
            # Stop the monitoring thread
            stop_thread = True
            monitor_thread.join(timeout=1.0)
            
            # Calculate elapsed times
            end_wall_time = time.time()
            end_cpu_time = process_cpu_time(process)
            
            wall_time = end_wall_time - start_wall_time
            cpu_time = end_cpu_time - start_cpu_time
            
            # Calculate average CPU percentage
            avg_cpu_percent = (
                sum(cpu_percent_samples) / len(cpu_percent_samples) if cpu_percent_samples else 0
            )
            
            # Adjust for macOS if needed
            if sys.platform == "darwin":
                from ..utils import macos_timebase_factor
                factor = macos_timebase_factor()
                cpu_time *= factor
                avg_cpu_percent *= factor
            
            # Format memory for display
            if peak_memory < 1024:
                memory_str = f"{peak_memory:.2f} MB"
            else:
                memory_str = f"{peak_memory / 1024:.2f} GB"
            
            # Format times for display
            if wall_time < 60:
                wall_time_str = f"{wall_time:.2f} seconds"
            elif wall_time < 3600:
                wall_time_str = f"{wall_time / 60:.2f} minutes"
            else:
                wall_time_str = f"{wall_time / 3600:.2f} hours"
            
            if cpu_time < 60:
                cpu_time_str = f"{cpu_time:.2f} seconds"
            elif cpu_time < 3600:
                cpu_time_str = f"{cpu_time / 60:.2f} minutes"
            else:
                cpu_time_str = f"{cpu_time / 3600:.2f} hours"
            
            # Log the resource usage
            logger.info("Resource usage summary:")
            logger.info(f"  • Wall clock time: {wall_time_str}")
            logger.info(f"  • CPU time: {cpu_time_str}")
            logger.info(f"  • Average CPU utilization: {avg_cpu_percent:.1f}%")
            logger.info(f"  • Peak memory usage: {memory_str}")
    
    return wrapper


def show_banner(command_name: str, version: str = "1.73.5"):
    """Display gsMap banner and version information."""
    command_name = command_name.replace("_", " ")
    gsMap_main_logo = pyfiglet.figlet_format(
        "gsMap",
        font="doom",
        width=80,
        justify="center",
    ).rstrip()
    print(gsMap_main_logo, flush=True)
    version_number = "Version: " + version
    print(version_number.center(80), flush=True)
    print("=" * 80, flush=True)
    logger.info(f"Running {command_name}...")
    
    # Record start time for the log message
    start_time = time.strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Started at: {start_time}")


def dataclass_typer(func):
    """
    Decorator to convert a function that takes a dataclass config
    into a Typer command with individual CLI options.
    """
    sig = inspect.signature(func)
    
    # Get the dataclass type from the function signature
    config_param = list(sig.parameters.values())[0]
    config_class = config_param.annotation
    
    @wraps(func)
    @track_resource_usage  # Add resource tracking
    def wrapper(**kwargs):
        # Show banner
        try:
            from .. import __version__
            version = __version__
        except ImportError:
            version = "development"
        show_banner(func.__name__, version)
        
        # Create the config instance
        config = config_class(**kwargs)
        result = func(config)
        
        # Record end time
        end_time = time.strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"Finished at: {end_time}")
        
        return result
    
    # Build new parameters from dataclass fields
    params = []
    for field in fields(config_class):
        # Skip fields that are part of parent class and handled internally
        if field.name in ['project_dir']:
            continue
            
        # Check if field has a default value
        if field.default is not field.default_factory:
            # Field has a default value, use it as the parameter default
            param = inspect.Parameter(
                field.name,
                inspect.Parameter.KEYWORD_ONLY,
                annotation=field.type,
                default=field.default
            )
        else:
            # No default, parameter is required
            param = inspect.Parameter(
                field.name,
                inspect.Parameter.KEYWORD_ONLY,
                annotation=field.type
            )
        params.append(param)
    
    # Update the wrapper's signature
    wrapper.__signature__ = inspect.Signature(params)
    
    # Preserve the original function's docstring
    wrapper.__doc__ = func.__doc__
    
    return wrapper