#!/bin/bash
# Regenerates every figure in the paper.
#
# Each line below is one figure; add, remove or retune them freely — the slice
# indices (x/y/z) and the RSM direction (k) are the knobs you are most likely
# to want to change once you have seen the output.  Run
#
#   python make_figures.py --list-directions dataset=zenodo
#
# to see which k values sit inside the missing wedge (large missing arc) and
# which are fully measured (arc 0).  With the standard 45-degree tilt limit,
# k=0 is the worst-case direction and k=2 is fully covered.
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"
PY=/opt/conda/bin/python

$PY make_figures.py \
  --fig "dataset=zenodo:dc_type=main:k=0:diff=1:name=fig_trabecular_bone_wedge" \
  --fig "dataset=zenodo:dc_type=main:k=2:diff=1:name=fig_trabecular_bone_covered" \
  --fig "dataset=nielsen-mammoth:k=0:diff=1:name=fig_mammoth_wedge" \
  --fig "dataset=nielsen-m:k=0:diff=1:name=fig_nielsenm_wedge" \
  --fig "dataset=nielsen-t:k=0:diff=1:name=fig_nielsent_wedge" \
  --fig "dataset=fiber-synthetic-full:k=0:diff=1:name=fig_fiber_full" \
  --fig "dataset=steel-wire-waxs:k=0:diff=0:name=fig_steelwire_wedge" \
  "$@"

echo "done."
