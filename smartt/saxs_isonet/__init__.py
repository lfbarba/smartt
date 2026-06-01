"""SAXS-TT missing-wedge correction pipeline.

Self-contained isonet-style pipeline for filling the missing wedge in each
of the K FBP sub-CT volumes produced by saxs_fbp_reconstruction.

Public sub-modules
------------------
wedge   — missing wedge mask (Fourier-space bool array) and conditioning scalar.
dataset — MissingWedgeSAXS training dataset.
train   — training script entry point (called as subprocess by pipeline).
infer   — inference script entry point (called as subprocess by pipeline).
pipeline — top-level iterative orchestrator.
"""
