#ifndef LBM_HH
#define LBM_HH

#include <mpi.h>

#include <cstddef>
#include <cstdint>
#include <vector>

/**
 * Timers used by the MPI performance driver.
 *
 * total   : measured outside the timestep loop in main.cc.
 * compute : local non-MPI work, such as collide / bounce-back / stream / BCs.
 * comm    : exposed halo-exchange overhead, including packing/unpacking and
 *           MPI calls such as MPI_Irecv, MPI_Isend, and MPI_Waitall.
 */
struct MPITimers
{
  double total   = 0.0;
  double compute = 0.0;
  double comm    = 0.0;
};

/**
 * Print MPI timing statistics on rank 0.
 *
 * This function performs reductions after the timed solver loop. Therefore,
 * the MPI_Reduce overhead is not included in timers.total.
 */
void report_mpi_timers(const MPITimers & timers,
                       std::size_t nx,
                       std::size_t ny,
                       std::size_t steps,
                       MPI_Comm comm);

/**
 * @brief MPI 1D row-decomposed D2Q9 BGK LBM solver.
 *
 * This class keeps the original serial solver structure as much as possible:
 * the same D2Q9 constants, BGK collision, bounce-back, pull streaming, inlet,
 * and outlet routines are retained. The main difference is that each MPI rank
 * owns only a contiguous block of global rows, plus two ghost rows.
 *
 * Local row layout:
 *   local y = 0              top ghost row
 *   local y = 1..local_ny    owned rows
 *   local y = local_ny + 1   bottom ghost row
 *
 * Distributions are still stored in structure-of-arrays layout:
 *   f_[ i * (nx * local_rows) + local_y * nx + x ]
 */
class LBM
{
public:
  static constexpr int Q = 9;

  static const int    cx[Q];   ///< Discrete velocity x-components.
  static const int    cy[Q];   ///< Discrete velocity y-components.
  static const double w[Q];    ///< Equilibrium weights.
  static const int    opp[Q];  ///< Index of the opposite direction.

  LBM(std::size_t nx, std::size_t ny,
          double u_in, double Re,
          double cyl_x, double cyl_y, double cyl_r,
          MPI_Comm comm = MPI_COMM_WORLD);

  /// Add a second circular obstacle. No-op if r2 <= 0.
  void add_second_cylinder(double cyl2_x, double cyl2_y, double cyl2_r);

  /// Set f to the equilibrium distribution with rho = 1, u = (u_in, 0).
  void initialize();

  /// Advance one step using non-blocking halo exchange.
  void step_nonblocking(MPITimers & timers);

  /// Serial-compatible wrapper. No timing breakdown is accumulated.
  void step();

  // Macroscopic accessors using global coordinates. They return 0 if the
  // requested global row is not owned by this rank.
  double rho      (std::size_t x, std::size_t y) const;
  double ux       (std::size_t x, std::size_t y) const;
  double uy       (std::size_t x, std::size_t y) const;
  double vorticity(std::size_t x, std::size_t y) const;
  bool   is_solid (std::size_t x, std::size_t y) const;


  /// Gather the global solid mask onto the root rank.
  /// On non-root ranks, global_mask is cleared.
  void gather_mask(std::vector<std::uint8_t> & global_mask, int root = 0) const;

  /// Gather rho/ux/uy onto the root rank and compute vorticity on root.
  /// On non-root ranks, output vectors are cleared.
  void gather_fields(std::vector<double> & rho_v,
                     std::vector<double> & ux_v,
                     std::vector<double> & uy_v,
                     std::vector<double> & vort_v,
                     int root = 0) const;

  /// Reduce the probe velocity to root. Only the rank owning y contributes.
  void probe_velocity(std::size_t x, std::size_t y,
                      double & ux_out, double & uy_out,
                      int root = 0) const;

  int rank() const { return rank_; }
  int size() const { return size_; }

  std::size_t nx()         const { return nx_; }
  std::size_t ny()         const { return ny_global_; }
  std::size_t local_ny()   const { return local_ny_; }
  std::size_t y_begin()    const { return y0_; }
  double      tau()        const { return tau_; }
  double      u_in()       const { return u_in_; }

private:
  std::size_t local_rows() const { return local_ny_ + 2; }
  std::size_t local_N()    const { return nx_ * local_rows(); }

  std::size_t idx_local(std::size_t x, std::size_t ly) const
  {
    return ly * nx_ + x;
  }

  std::size_t fidx(int i, std::size_t x, std::size_t ly) const
  {
    return std::size_t(i) * local_N() + idx_local(x, ly);
  }

  bool owns_global_y(std::size_t gy) const
  {
    return gy >= y0_ && gy < y0_ + local_ny_;
  }

  std::size_t local_y_from_global(std::size_t gy) const
  {
    return (gy - y0_) + 1;
  }

  long global_y_from_local(std::size_t ly) const
  {
    return long(y0_) + long(ly) - 1;
  }

  void mark_obstacle(double c_x, double c_y, double r);

  // Original serial kernels, adjusted to loop only over owned rows.
  void collide();
  void bounce_back();
  void apply_inlet();
  void apply_outlet();

  // Streaming split for communication/computation overlap.
  void stream_rows(std::size_t ly_begin, std::size_t ly_end);
  void stream_interior_rows();
  void stream_boundary_rows();

  // Halo exchange helpers.
  //
  // Optimization note:
  // For row-wise MPI decomposition, streaming across a horizontal rank boundary
  // only needs the three D2Q9 populations whose cy points across that boundary.
  // Therefore each message sends 3 * nx values instead of the full 9 * nx row.
  // Across both neighbours, this exchanges the 6 directions with cy != 0.
  void pack_row_dirs(std::size_t ly,
                   const int dirs[3],
                   std::vector<double> & buffer) const;

  void unpack_row_dirs(std::size_t ly,
                     const int dirs[3],
                     const std::vector<double> & buffer);

  void post_halo_exchange(MPI_Request requests[4]);
  void finish_halo_exchange(MPI_Request requests[4]);
  std::size_t nx_;
  std::size_t ny_global_;
  std::size_t local_ny_;
  std::size_t y0_;

  double u_in_;
  double tau_;

  MPI_Comm comm_;
  int rank_;
  int size_;
  int up_rank_;
  int down_rank_;

  std::vector<double>  f_;      ///< Current distributions, size 9*nx*(local_ny+2).
  std::vector<double>  ftmp_;   ///< Scratch buffer for streaming.
  std::vector<uint8_t> solid_;  ///< Local solid mask including ghost rows.

  std::vector<double> send_top_;
  std::vector<double> send_bottom_;
  std::vector<double> recv_top_;
  std::vector<double> recv_bottom_;
};

#endif  // LBM_HH
