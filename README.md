# GooseEPM

[![CI](https://github.com/tdegeus/GooseEPM/actions/workflows/ci.yml/badge.svg)](https://github.com/tdegeus/GooseEPM/actions/workflows/ci.yml)
[![Doxygen -> gh-pages](https://github.com/tdegeus/GooseEPM/workflows/gh-pages/badge.svg)](https://tdegeus.github.io/GooseEPM)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/gooseepm.svg)](https://anaconda.org/conda-forge/gooseepm)

Implementation of an Elasto Plastic Model

# Python module

## From source

```bash
# Download GooseEPM
git checkout https://github.com/tdegeus/GooseEPM.git
cd GooseEPM

# Get prerequisites. An example is given using conda, but there are many other ways
conda activate myenv
conda env update --file environment.yaml
# (if you use hardware optimisation, below, you also want)
conda install -c conda-forge xsimd

# Compile and install the Python module
# (-v can be omitted as is controls just the verbosity)
python -m pip install . -v

# Or, compile with hardware optimisation (fastest), see scikit-build docs
SKBUILD_CONFIGURE_OPTIONS="-DUSE_SIMD=1" python -m pip install . -v

# Note that you can also compile with debug assertions (very slow)
SKBUILD_CONFIGURE_OPTIONS="-USE_DEBUG=1" python -m pip install . -v

# Or, without any assertions (slightly faster, but more dangerous)
SKBUILD_CONFIGURE_OPTIONS="-USE_ASSERT=1" python -m pip install . -v
```
