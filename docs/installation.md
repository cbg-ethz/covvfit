# Installation guide

### Conventions

We assume that the following are available:

  - A Unix-based shell. For example, the terminal emulator on Linux, BSD, MacOS or the [WSL](https://learn.microsoft.com/en-us/windows/wsl/install) terminal on Windows.
  - A Python 3.11 (or newer) installation. It can be obtained by using e.g., [Micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html).
  
We use the convention that `$` represents the command line prompt.
For example,

```bash
$ run command
```
means that one should type `run command` into the command line shell.

### Installation instructions

To install *Covvfit*, ensure that you have Python installed and run:

```bash
$ pip install covvfit
```

This installs:

  - The command line program `covvfit`, which can be run on deconvolved data to estimate fitness advantages. 
  - The Python package `covvfit`, which can be used to create more complex automated workflows or employed to build other tools.
  
### Checking if the installation works
    
Run
```bash
$ covvfit check
```

If the program outputs `[Status: OK]`, then the tool is installed properly.

However, if you see any message with `[Status: Error]`, then it means there is a problem with installation.

It is possible, that an incompatible version of the JAX package has been downloaded. In this case, consult [this installation guide](https://docs.jax.dev/en/latest/installation.html).
In any other case, please report the problem by filling an issue on [our bug tracker](https://github.com/cbg-ethz/covvfit/issues), including the output in the description of the problem.
