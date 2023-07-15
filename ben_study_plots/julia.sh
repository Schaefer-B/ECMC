#!/bin/sh

export JULIA_PREFIX="/net/e4-nfs-home.e4.physik.tu-dortmund.de/home/bschaefer/julia-1.9.2"
export PATH="$JULIA_PREFIX/bin:$PATH"

export JULIA_NUM_THREADS=4

julia /net/e4-nfs-home.e4.physik.tu-dortmund.de/home/bschaefer/performance_tests/performance_tests_run.jl

