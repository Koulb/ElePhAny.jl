## Index

```@index
```

## Electrons

```@docs
ElectronPhonon.read_qe_xml
ElectronPhonon.create_scf_calc
ElectronPhonon.generate_kpoints
ElectronPhonon.include_kpoins
ElectronPhonon.create_disp_calc
ElectronPhonon.create_disp_calc!
ElectronPhonon.create_perturbed_kcw
ElectronPhonon.run_scf
ElectronPhonon.run_scf_cluster
ElectronPhonon.run_nscf_calc
ElectronPhonon.run_disp_calc
ElectronPhonon.run_disp_nscf_calc
ElectronPhonon.prepare_kcw_data
ElectronPhonon.read_potential
ElectronPhonon.save_potential
ElectronPhonon.get_kpoint_list
ElectronPhonon.fold_kpoint
ElectronPhonon.prepare_eigenvalues
ElectronPhonon.create_electrons
ElectronPhonon.load_electrons
```

## Phonons

```@docs
ElectronPhonon.determine_q_point
ElectronPhonon.determine_q_point_cart
ElectronPhonon.read_forces_xml
ElectronPhonon.dislpaced_unitecells
ElectronPhonon.collect_forces
ElectronPhonon.save_dyn_matirx
ElectronPhonon.prepare_phonons_data
ElectronPhonon.prepare_phonons
ElectronPhonon.create_phonons
ElectronPhonon.load_phonons
```

## Electron-phonon coupling

```@docs
ElectronPhonon.run_calculations
ElectronPhonon.prepare_model
ElectronPhonon.electron_phonon_qe
ElectronPhonon.find_degenerate
ElectronPhonon.parse_ph
ElectronPhonon.electron_phonon
ElectronPhonon.plot_ep_coupling
```

## Symmetries

```@docs
ElectronPhonon.check_symmetries
ElectronPhonon.check_symmetries!
ElectronPhonon.fold_component
ElectronPhonon.rotate_grid
ElectronPhonon.rotate_deriv
```

## Wave functions

```@docs
ElectronPhonon.parse_wf
ElectronPhonon.parse_hdf
ElectronPhonon.parse_fortran_bin
ElectronPhonon.wf_from_G
ElectronPhonon.wf_from_G_slow
ElectronPhonon.wf_from_G_list
ElectronPhonon.wf_to_G
ElectronPhonon.wf_to_G_list
ElectronPhonon.wf_pc_to_sc
ElectronPhonon.determine_fft_grid
ElectronPhonon.determine_phase
ElectronPhonon.prepare_unfold_to_sc
ElectronPhonon.wf_phase!
ElectronPhonon.prepare_wave_functions_to_R
ElectronPhonon.prepare_wave_functions_to_G
ElectronPhonon.prepare_wave_functions_undisp
ElectronPhonon.prepare_wave_functions_disp
ElectronPhonon.calculate_braket_real
ElectronPhonon.calculate_braket
```

## Model

```@docs
ElectronPhonon.Symmetries
ElectronPhonon.ModelQE
ElectronPhonon.ModelKCW
ElectronPhonon.create_model
ElectronPhonon.create_model_kcw
ElectronPhonon.Electrons
ElectronPhonon.Phonons
```

## IO

```@docs
ElectronPhonon.parse_frozen_params
ElectronPhonon.parse_qe_in
ElectronPhonon.getfirst
ElectronPhonon.parse_file
```

