_ERROR = "[Status: Error] The tool does not work."


def _check_jax() -> None:
    try:
        import jax.numpy as jnp

        a = jnp.array(3.0)
        b = jnp.array(5.0)
        (a + b).block_until_ready()
    except Exception:
        print(_ERROR)
        print(
            "    The JAX package does not seem to be installed and functioning properly."
        )
        raise


def _check_infer() -> None:
    try:
        import covvfit._cli.infer as infer
    except Exception:
        print(_ERROR)
        raise

    if not hasattr(infer, "infer"):
        raise ValueError(_ERROR)


def _check_freyja() -> None:
    try:
        import covvfit._cli.freyja as freyja
    except Exception:
        print(_ERROR)
        raise

    if not hasattr(freyja, "freyja_gather"):
        raise ValueError(_ERROR)


def check():
    """Checks if the tool has been installed properly."""
    _check_jax()
    _check_infer()
    _check_freyja()

    print("[Status: OK] The tool has been installed properly.")
