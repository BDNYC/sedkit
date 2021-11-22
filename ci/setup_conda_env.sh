#!/bin/bash
echo "Creating conda environment for Python $PYTHON_VERSION"
conda env create -f "env/environment-${PYTHON_VERSION}.yml" || exit 1
export CONDA_ENV=sedkit-$PYTHON_VERSION
source activate $CONDA_ENV
conda config --add channels conda-forge
conda install pytest-runner --no-deps
conda install healpy --no-deps
pip install pytest pytest-cov coveralls
