# Covvfit: Variant fitness estimates from wastewater data

*Covvfit* is a framework for estimating relative growth advantages of different variants from deconvolved wastewater samples.
It consists of command line tools, which can be included in the existing workflows and a Python package, which can be used to quickly develop custom solutions and extensions.

## Getting started

*Covvfit* can be installed from the Python Package Index:

```bash
$ pip install covvfit
$ covvfit check
```

For an example **how to analyze the data** using the provided command line tool, see [this tutorial](./cli.md).

For more detailed installation instructions, including troubleshooting, see the [installation guide](./installation.md).


## FAQ

**How do I run *Covvfit* on my data?**

We recommend to start using *Covvfit* as a command line tool, with the tutorial available [here](cli.md). 

**What data does *Covvfit* use?**

*Covvfit* uses deconvolved wastewater data, accepting relative abundances of different variants measured at different locations and times.
Tools such as [LolliPop](https://github.com/cbg-ethz/LolliPop) or [Freyja](https://github.com/andersen-lab/Freyja/) can be used to deconvolve wastewater data. 

Note, however, that the deconvolution procedure should not smooth abundance results. For more information on this topic, see [here](running_deconv/lollipop.md).

**Can *Covvfit* predict emergence of new variants?**

No, *Covvfit* explicitly assumes that no new variants emerge in the considered timeframe, so its predictions are unlikely to hold on longer timescales.
The underlying model also cannot take into account changes in the transmission dynamics or immune response, so that it cannot predict the effects of vaccination programs or lockdowns.

**How can I contact the developers?**

In case you find a bug, want to ask about integrating *Covvfit* into your pipeline, or have any other feedback, we would love to hear it via our [issue tracker](https://github.com/cbg-ethz/covvfit/issues)!
In this manner, other users can also benefit from your insights.

**Is there a manuscript associated with the tool?**

The manuscript is being finalised. We hope to release the preprint describing the method in February 2025.
In case you would like to cite *Covvfit* in your work, we would recommend the following for now:

D. Dreifuss, P. Czyż, N. Beerenwinkel, *Learning and forecasting selection dynamics of SARS-CoV-2 variants from wastewater sequencing data using Covvfit* (2025; in preparation). URL: https://github.com/cbg-ethz/covvfit
