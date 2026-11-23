#!/bin/bash
# Runs after every search worker has ended: retrains each fold's best settings and tests each fold once.
source "${SLURM_SUBMIT_DIR:?Submit with: bash submit.sh}/common.sh"
exec $PY -m prismt hpo --config config.json --run-dir results --finalize
