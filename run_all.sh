#!/usr/bin/env bash
# Esegue l'intera pipeline robusta in ordine.
# Il tuning (05) e' opzionale: di default si usano i best params gia' in config.py.
set -e
cd "$(dirname "$0")/src"

echo "===== 01 target ====="              && python3 01_target.py
echo "===== 02 profilo ====="             && python3 02_profilo.py
echo "===== 00 eda target ====="          && python3 00_eda.py
echo "===== 03 features ====="            && python3 03_features.py
echo "===== 04 validazione walk-forward =====" && python3 04_validation.py
# echo "===== 05 tuning ====="            && python3 05_tune.py     # opzionale (lento)
echo "===== 06 train + submission =====" && python3 06_train.py
echo "FATTO -> output/submission_onesta.txt (report) + output/submission_leaderboard.txt (punteggio)"
