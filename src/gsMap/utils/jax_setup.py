"""JAX settings that gsMap's float32 kernels rely on, applied when a run starts."""


def use_float32_jax() -> None:
    """Keep JAX in 32-bit mode for gsMap kernels.

    This is JAX's default. It is set explicitly when a gsMap step starts (not at
    import), so an environment with ``JAX_ENABLE_X64=1`` still runs gsMap's float32
    kernels unchanged, while merely importing gsMap leaves other code's JAX settings
    alone.
    """
    import jax

    jax.config.update("jax_enable_x64", False)
