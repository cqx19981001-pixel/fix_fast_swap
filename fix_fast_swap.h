#ifdef FIX_CLASS
// clang-format off
FixStyle(fast/swap,FixFastSwap);
// clang-format on
#else

#ifndef FIX_FAST_SWAP_H
#define FIX_FAST_SWAP_H

#include "fix.h"

namespace LAMMPS_NS {

class Compute;
class NeighRequest;
class NeighList;
class RanPark;
class PairEAM;

class FixFastSwap : public Fix {
 public:
  FixFastSwap(class LAMMPS *, int, char **);
  ~FixFastSwap() override;

  // ---- LAMMPS fix interface ----
  int setmask() override;
  void init() override;
  void init_list(int, NeighList *) override;

  // execution hooks
  void post_force(int vflag) override;
  void post_force_respa(int vflag, int ilevel, int iloop) override;

  void pre_force(int vflag) override;
  void pre_force_respa(int vflag, int ilevel, int iloop) override;

  // communication (atom types)
  // owner -> ghost
  int pack_forward_comm(int n, int *list, double *buf, int pbc_flag, int *pbc) override;
  void unpack_forward_comm(int n, int first, double *buf) override;

  // ghost -> owner (EAM-style accumulation)
  // int pack_reverse_comm(int n, int first, double *buf) override;
  // void unpack_reverse_comm(int n, int *list, double *buf) override;

  // thermo output
  double compute_vector(int i) override;

 protected:
#include <vector>
  inline int owner_index(int idx) const;

  // ---------- many-body (EAM) local energy support ----------

  // Whether rho needs to be maintained (for many-body potentials)
  bool need_rho_init;

  // Rho cache (size at least atom->nmax)
  std::vector<double> rho_cache;

  // Timestep associated with the current rho_cache
  // (used to avoid redundant recomputation)
  bigint rho_cache_step = -1;

 private:
  bigint mc_cycle_;    // MC scheduling cycle counter
  std::vector<int> candidate_list_;
  int ncolors_;         // Number of colors actually used (1, 2, 4, or 8)
  int bx_, by_, bz_;    // Enable parity coloring in x/y/z directions
  bool mc_parallel_enabled;
  bool mc_parallel_checked;

  int active_color(bigint mc_cycle) const;
  bool is_active_rank(int color) const;

  inline double influence_radius() const;

  // ---------- new: internal function to build this mapping ----------
  void build_tag2ids_cache();

  Compute *c_pe;    // Full-system energy compute (diagnostic only)

  // ---- neighbor list ----
  NeighRequest *request;
  NeighList *list_full;

  // ---- RNG ----
  RanPark *random_equal;
  RanPark *random_unequal;

  // ---- MC parameters ----
  int nevery;
  int ncycle;    // Number of MC attempts per invocation
                 // (compatible with atom/swap)
  double beta;
  int seed;

  int nswaptypes;
  int *type_list;
  double *mu;

  // ---- eligible atom bookkeeping ----
  int niswap_local, niswap_global, niswap_before;
  int *local_iatom_list;
  int nmax_local_list;

  int unequal_cutoffs;

  // ---- thermo counters ----
  bigint n_attempts;
  bigint n_accepts;

  // Output cache (global accumulated values)
  bigint n_attempts_global;
  bigint n_accepts_global;

  // Avoid redundant global reductions
  bigint last_global_reduce_step;

  // ---- internal helpers ----
  void build_local_swap_list();
  int pick_i_swap_atom();           // Unified entry point
  int pick_i_swap_atom_serial();    // Serial fallback
  bool check_parallel_mc_feasible();
  int pick_i_swap_atom_parallel();
  int pick_new_type(int itype, int &jtype);
  double energy_local_atom(int i);
  int attempt_one_flip();

  // ---- diagnostic only ----
  double energy_full();    // Full-system energy (for validation/debug only)
};

}    // namespace LAMMPS_NS

#endif
#endif
