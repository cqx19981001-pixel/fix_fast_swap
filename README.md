
# fix_fast_swap

`fix_fast_swap` is a custom LAMMPS fix implementing an optimized and strictly localized Monte Carlo (MC) atom-swap algorithm for semi-grand canonical (SGC) simulations. The primary goal of this fix is to significantly reduce the computational cost of MC swap attempts by evaluating energy differences locally, rather than recomputing the full system energy for every trial move.

This repository contains only the new fix implementation and does not modify any original LAMMPS source files.

---

## Key Features

- Localized MC energy-difference evaluation  
  Energy changes are computed only for the affected atom and its local neighborhood.
- Pairwise-only formulation  
  The algorithm is rigorously valid only for pairwise-additive interactions.
- Explicit allow-list of supported pair styles  
  Unsupported potentials are rejected at runtime.
- Non-intrusive integration  
  Implemented entirely as a standalone fix.
- Fail-fast safety design  
  Incorrect usage is detected early and terminates with a clear error message.

---

## Supported Pair Styles

To guarantee physical correctness and reproducibility, `fix_fast_swap` explicitly restricts the allowed pair styles via an internal allow-list implemented locally inside the fix.

Currently supported pair styles:

- `pair_style lj/cut`
- `pair_style lj/table`

All other pair styles are intentionally disabled.

In particular:

- Many-body potentials (e.g., EAM, MEAM, Tersoff, SNAP, DeepMD, etc.) are not supported and will trigger a hard runtime error.
- This restriction does not affect other LAMMPS fixes or simulations and applies only when `fix fast/swap` is explicitly invoked.

Future extensions can be enabled by appending new entries to the internal supported-pair-style list in `fix_fast_swap.cpp`.

---

## Design Rationale

The localized MC swap formulation relies on the following assumptions:

1. Pairwise additivity of interaction energies  
2. Finite and well-defined interaction cutoff  
3. Strict locality of the energy perturbation induced by a single swap  

Standard Lennard–Jones pair potentials (`lj/cut`, `lj/table`) satisfy these conditions. Most many-body potentials do not, making a localized energy-difference formulation generally invalid.

Rather than silently producing incorrect results, this fix adopts a conservative fail-fast design and aborts execution if an unsupported pair style is detected.

This design prioritizes physical correctness, reproducibility, clear user feedback, and long-term maintainability.

---

## Installation

1. Copy the following files into your LAMMPS `src/MC` directory:

   fix_fast_swap.cpp
   fix_fast_swap.h


2. Re-compile and Rebuild LAMMPS using your standard build procedure, for example:


   make -j 8


No changes to any original LAMMPS source files are required.

---

## Usage Example


pair_style lj/cut 2.5
pair_coeff * * 1.0 1.0 2.5

fix mcswap all fast/swap 1 10 12345 0.7 types 2 1 2 mu 0 2.0
which is exactly the same as (but much faster than) the fix atom/swap command:
fix mcswap all atom/swap 1 10 12345 0.7 semi-grand yes types 1 2 mu 0.0 2.0


If an unsupported pair style is used, LAMMPS will terminate with an error message similar to:


Fix fast/swap supports only explicitly listed pair styles (currently: lj/cut, lj/table)


---

## Parallel Safety

* The fix includes runtime checks to determine whether parallel MC site selection is safe under the current MPI domain decomposition.
* If the safety criteria are not satisfied, the fix automatically and permanently falls back to a serial MC selection path, ensuring correctness.

---

## Scope and Limitations

* Intended for pairwise Lennard–Jones–type systems.
* Not a drop-in replacement for `fix atom/swap` in general many-body simulations.
* The current implementation favors correctness and clarity over generality.

---

## Citation

If you use this fix in your research, please cite:

> Q. Chen, L. Wang, J. Wang, J. J. Hoyt,
> *Efficient Localized Monte Carlo Sampling for the Semi-Grand Canonical Ensemble in LAMMPS*,
> Manuscript in preparation.

---

## License

This code is provided for research use. Please refer to the LICENSE file for details.

---

## Contact

For questions, bug reports, or discussions regarding algorithmic or implementation details, please open an issue or contact the author directly.


