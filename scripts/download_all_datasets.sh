#!/usr/bin/env bash
set -u
python -m pip install -q -e '.[paper]'
# Each dataset is independent so one repository guestbook/API change does not
# prevent the other open datasets from downloading/preparing.
for d in lira kit kuleuven deep_dynamics comma2k19 extreme_road bicycle_tire mendeley_friction; do
  echo "==== $d ===="
  if safegrip download --datasets "$d"; then
    safegrip prepare --dataset "$d" || echo "[warning] $d downloaded but its optional canonicalizer needs schema adjustment. Raw data remain intact."
  else
    echo "[warning] $d download unavailable under current upstream access/API; continuing."
  fi
done
