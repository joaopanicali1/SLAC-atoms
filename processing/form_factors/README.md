# Form Factors

This folder stores the x-ray atomic form factor data used during trajectory and
scattering generation.

`processing/analyse_mol_v2.py` imports `atomic_form_factors.py` and evaluates
the Waasmaier-Kirfel coefficients on the q grid for each molecule. Those
per-element factors are used to weight the synthetic scattering curves `S0(q,t)`
and the downstream finite-q pair-density quantities.

`form_factors.py` is retained as the raw/helper source for the table. Most
workflows should import `ATOMIC_FORM_FACTORS` from `atomic_form_factors.py`
rather than running this folder directly.
