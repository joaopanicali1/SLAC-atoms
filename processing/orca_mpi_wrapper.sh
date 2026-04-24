#!/bin/bash
# Multi-node ORCA launcher. ASE calls: orca_mpi_wrapper.sh mol.inp
# Translates to: mpirun -np $SLURM_NTASKS /path/to/orca mol.inp
ORCA_BIN="/home/cme290/joaop/SLAC-atoms/orca_6_1_1_linux_x86-64_shared_openmpi418_nodmrg/orca"
exec mpirun -np "${SLURM_NTASKS:-32}" "$ORCA_BIN" "$@"