#!/bin/bash
# One search worker for one fold (array task = fold x workers + worker); workers of a fold
# share results/hpo/fold_<k>/study.journal.
source "${SLURM_SUBMIT_DIR:?Submit with: bash submit.sh}/common.sh"
exec $PY -m prismt hpo --config config.json --run-dir results --task-index "${SLURM_ARRAY_TASK_ID:-0}"
