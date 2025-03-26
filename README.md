[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![build](https://github.com/cbg-ethz/covvfit/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/cbg-ethz/covvfit/actions/workflows/test.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/charliermarsh/ruff)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PyPI Latest Release](https://img.shields.io/pypi/v/covvfit.svg)](https://pypi.org/project/covvfit/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15085753.svg)](https://doi.org/10.5281/zenodo.15085753)
[![medRxiv](https://img.shields.io/badge/preprint-medRxiv-darkred)](https://doi.org/10.1101/2025.03.25.25324639)

# covvfit

![Covvfit demonstration](docs/static/infer-output-figure.jpg)

Fitness estimates of SARS-CoV-2 variants from variant abundance data.

  - **Documentation:** [https://cbg-ethz.github.io/covvfit](https://cbg-ethz.github.io/covvfit)
  - **Source code:** [https://github.com/cbg-ethz/covvfit](https://github.com/cbg-ethz/covvfit)
  - **Bug reports:** [https://github.com/cbg-ethz/covvfit/issues](https://github.com/cbg-ethz/covvfit/issues)


## Installation and usage

*Covvfit* can be installed from the Python Package Index:

```bash
$ pip install covvfit
```

For an example how to analyze the data see [this tutorial](https://cbg-ethz.github.io/covvfit/cli/).


## References

This method accompanies our manuscript:

David Dreifuss, Paweł Piotr Czyż, Niko Beerenwinkel. *Learning and forecasting selection dynamics of SARS-CoV-2 variants from wastewater sequencing data using Covvfit*. medRxiv 2025.03.25.25324639; doi: [https://doi.org/10.1101/2025.03.25.25324639](https://doi.org/10.1101/2025.03.25.25324639)


```bibtex
@article{Dreifuss2025-Covvfit,
	author = {Dreifuss, David and Czy{\.z}, Pawe{\l} Piotr and Beerenwinkel, Niko},
	title = {Learning and forecasting selection dynamics of SARS-CoV-2 variants from wastewater sequencing data using Covvfit},
	elocation-id = {2025.03.25.25324639},
	year = {2025},
	doi = {10.1101/2025.03.25.25324639},
	publisher = {Cold Spring Harbor Laboratory Press},
	eprint = {https://www.medrxiv.org/content/early/2025/03/26/2025.03.25.25324639.full.pdf},
	journal = {medRxiv}
}
```


## See Also

  - [V-pipe](https://cbg-ethz.github.io/V-pipe/): a bioinformatics pipeline for viral sequencing data.
  - [cojac](https://github.com/cbg-ethz/cojac): command-line tools for the analysis of co-occurrence of mutations on amplicons.


