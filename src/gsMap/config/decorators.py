"""
Decorators for CLI and resource tracking.
"""

import copy
import inspect
import logging
import time
from dataclasses import fields
from functools import wraps
from importlib.metadata import PackageNotFoundError, distribution
from typing import Annotated, Any, get_args, get_origin

import pyfiglet

from gsMap.logging_config import resolve_project_dir_from_kwargs, stage_logging
from gsMap.resource_usage import track_resource_usage as track_resource_usage

logger = logging.getLogger("gsMap")


def show_banner(command_name: str, version: str = "unknown"):
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


def warn_superseded_distribution() -> None:
    """Warn when the superseded ``gsMap3D`` preview distribution is also installed.

    ``gsMap3D`` (0.1.x) and ``gsMap`` (>=2.0) install the same ``gsMap`` module, so
    removing either distribution deletes files the other one still needs.
    """
    try:
        distribution("gsMap3D")
    except PackageNotFoundError:
        return
    logger.warning(
        "The superseded 'gsMap3D' package is installed alongside 'gsMap'. Both provide the "
        "same 'gsMap' module. Run 'pip uninstall gsMap3D' and then "
        "'pip install --force-reinstall --no-deps gsMap' to repair the installation."
    )


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
    def wrapper(**kwargs):
        project_dir = resolve_project_dir_from_kwargs(kwargs)
        with stage_logging(project_dir, func.__name__) as logging_session:
            # Show banner
            try:
                from gsMap import __version__

                version = __version__
            except ImportError:
                version = "development"
            show_banner(func.__name__, version)
            warn_superseded_distribution()

            # Create the config only after the project DEBUG log is active so
            # validation and auto-resolution messages are retained.
            # CLI defaults are built once per process; copy mutable ones so a config
            # that edits its list or dict cannot change the next call's defaults.
            config = config_class(
                **{
                    key: copy.deepcopy(value) if isinstance(value, list | dict | set) else value
                    for key, value in kwargs.items()
                }
            )
            logging_session.record_config(config, stage=func.__name__)
            # stage_logging tracks the CLI and Python entry points alike.
            result = func(config)

            # Record end time
            end_time = time.strftime("%Y-%m-%d %H:%M:%S")
            logger.info(f"Finished at: {end_time}")

            return result

    # Build new parameters from dataclass fields
    from dataclasses import MISSING

    params = []

    core_only = getattr(config_class, "__core_only__", False)

    def is_core_field(field_name: str, cls: type[Any]) -> bool:
        """Check if a field originates from a 'Core' config class."""
        for base in cls.__mro__:
            if field_name in getattr(base, "__annotations__", {}):
                # If ANY class in the inheritance chain for this field explicitly disables quick mode, honor it
                if getattr(base, "__display_in_quick_mode_cli__", True) is False:
                    return False
        return True

    for field in fields(config_class):
        # Only include fields with Annotated type hints in the CLI
        # This allows internal fields to be excluded from CLI parameters

        # Check if the field type is Annotated
        if get_origin(field.type) != Annotated:
            continue

        # Get Annotated metadata
        annotated_args = get_args(field.type)

        # Check for explicit display override in Annotated metadata
        # e.g., Annotated[int, typer.Option(...), {"__display_in_quick_mode_cli__": True}]
        display_override = None
        for arg in annotated_args:
            if isinstance(arg, dict) and "__display_in_quick_mode_cli__" in arg:
                display_override = arg["__display_in_quick_mode_cli__"]
                break

        if core_only:
            # If field explicitly says True, we show it even if class says False
            if display_override is True:
                pass
            # If field explicitly says False, we hide it
            elif display_override is False:
                continue
            # Otherwise, fall back to class-level logic
            elif not is_core_field(field.name, config_class):
                continue

        # Determine the default value
        if field.default is not MISSING:
            # Field has a default value, use it as the parameter default
            default_value = field.default
        elif field.default_factory is not MISSING:
            # Field has a default factory, call it to get the default value
            default_value = field.default_factory()
        else:
            # No default, parameter is required
            default_value = inspect.Parameter.empty

        # Create the parameter
        if default_value is not inspect.Parameter.empty:
            param = inspect.Parameter(
                field.name,
                inspect.Parameter.KEYWORD_ONLY,
                annotation=field.type,  # Keep the full Annotated type
                default=default_value,
            )
        else:
            param = inspect.Parameter(
                field.name,
                inspect.Parameter.KEYWORD_ONLY,
                annotation=field.type,  # Keep the full Annotated type
            )
        params.append(param)

    # Update the wrapper's signature
    wrapper.__signature__ = inspect.Signature(params)

    # Preserve the original function's docstring
    wrapper.__doc__ = func.__doc__

    return wrapper
