import covvfit


def test_imports() -> None:
    assert isinstance(covvfit.VERSION, str)
    assert covvfit.variant_beta_binomial is not None
