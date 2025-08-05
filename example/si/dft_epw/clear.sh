#!/usr/bin/env bash

shopt -s extglob

rm -rf _ph0/
rm -rf  save/
rm -rf  si.save/
rm -rf  si_dft.save/
rm -rf  Fepmatkqcb1/
rm -rf  Fsparsecb/

rm -v !(si.jl|Si.upf|clear.sh|run_all.sh|scf.in|nscf.in|ph.in|q2r.in|matdyn.in|bands.in|bands_plot.in|epw0.in|epw1.in|epw2.in|epw3.in|epw4.in|path_k.kpt|path_q.kpt|si_path.kpt|fake2nscf.py|parse_epb.py|plt_bands1.in|plt_bands2.in|plt_freq1.in|plt_freq2.in|quadrupole.fmt)

# rm *.out *.dyn*
# rm *.dyn*
# rm *.dat
