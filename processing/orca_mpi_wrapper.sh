#!/bin/bash
# Multi-node ORCA launcher. ASE calls: orca_mpi_wrapper.sh orca.inp
#
# Uses srun (SLURM-native) with --mpi=pmi2.
# PMI2 is chosen because:
#   - SLURM 23.11 on this cluster supports pmix_v4 and pmi2
#   - The OpenMPI 4.1.8 build has pmix3x (not v4), so pmix would version-mismatch
#   - pmi2 is compatible with both SLURM 23.11 and OpenMPI 4.1.8
#
# srun inherits SLURM_NTASKS automatically — no -np flag needed.
# Environment variables (PATH, LD_LIBRARY_PATH) are propagated by SLURM.
#
# NOTE: scratch directory must be on a shared filesystem, not /dev/shm.

ORCA_BIN="/home/cme290/joaop/SLAC-atoms/orca_6_1_1_linux_x86-64_shared_openmpi418_nodmrg/orca"
exec srun --mpi=pmi2 "$ORCA_BIN" "$@"
