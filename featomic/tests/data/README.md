# Data for regression-testing featomic

This directory contains data used in featomic regression tests, as well as the
scripts used to generate this data.

Each test consist of two files: one JSON file containing all input data
(hyper-parameter and structures); and one compressed numpy (.npy) file
containing the raw data (representation value or gradients).

Since the data-generation scripts use python, you will have to use the latest
version of the python binding when generating new data or re-creating existing
data. Something like this should work:

```bash
cd featomic/tests/data/
python3 -m venv .regtests-pyvenv
source .regtests-pyvenv/bin/activate
pip install --upgrade pip
pip install ../../../
pip install -r requirements.txt

python <specific-script-here>.py
```
