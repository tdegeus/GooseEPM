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

# Get prerequisites, e.g. using conda
conda activate myenv
conda env update --file environment.yaml

# Compile and install the Python module
python -m pip install . -v

# Or: enable hardware optimisations
SKBUILD_CONFIGURE_OPTIONS="-DUSE_DEBUG=1" python -m pip install . -v
```
