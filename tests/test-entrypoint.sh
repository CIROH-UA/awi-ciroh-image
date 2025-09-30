#!/bin/bash
/srv/conda/envs/notebook/bin/python /tests/tests.py -v 2>&1 | tee /tests/pyngiab_tests.log
# No need to activate the venv since it is done automatically by PyNGIAB
#source /ngen/.venv/bin/activate && /srv/conda/envs/notebook/bin/python /tests/tests.py -v

# TEEHR(https://github.com/RTIInternational/teehr/) built-in tests
cd /tests && \
    git init teehr \
    && cd teehr \
    && git lfs install \
    && git remote add -f origin https://github.com/RTIInternational/teehr.git \
    && git config core.sparseCheckout true \
    && echo "tests/" >> .git/info/sparse-checkout \
    && git pull origin main
# Tests need to be executed from a specific directory (Thanks to Matt from RTI for the help)
cd /tests/teehr/ && /srv/conda/envs/notebook/bin/pytest tests/ 2>&1 | tee /tests/teehr_tests.log
