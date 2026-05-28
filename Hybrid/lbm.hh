#ifndef LBM_HH
#define LBM_HH

#include <mpi.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct HybridTimers
{
  double total   = 0.0;
  double compute = 0.0;
  double comm    = 0.0;  // Halo-exchange overhead, including synchronization/copies.
};

void report_hybrid_timers(const HybridTimers & timers,
                          std::size_t nx,
                          std::size_t ny,
                          std::size_t steps,
                          MPI_Comm comm);

/**
 * Intra-node hybrid MPI + CUDA D2Q9 BGK LBM solver.
 *
 * One MPI rank owns one row-block subdomain and one GPU. The local device
 * domain stores two ghost rows. Since local y increases with global y, the rows
 * are:
 *   local y = 0              low-y ghost row, from rank-1 when present
 *   local y = 1..local_ny    owned rows
 *   local y = local_ny + 1   high-y ghost row, from rank+1 when present
 *
 * The main CUDA kernel uses the same optimization as the single-GPU version:
 * fused pull-streaming + BGK collision with double buffering. Halo exchange is
 * CUDA-aware by default: MPI_Isend/Irecv receive GPU device pointers directly.
 * A host-staged mode is also provided for comparison.
 */
class LBM
{
public:
  static constexpr int Q = 9;
  static const int    cx[Q];
  static const int    cy[Q];
  static const double w[Q];
  static const int    opp[Q];

  LBM(std::size_t nx, std::size_t ny,
      double u_in, double Re,
      double cyl_x, double cyl_y, double cyl_r,
      MPI_Comm comm = MPI_COMM_WORLD,
      std::size_t pitch_align = 32,
      const std::string & halo_mode = "cuda");

  ~LBM();

  LBM(const LBM &) = delete;
  LBM & operator=(const LBM &) = delete;

  void set_cuda_block(unsigned int block_x, unsigned int block_y);
  void add_second_cylinder(double cyl2_x, double cyl2_y, double cyl2_r);
  void initialize();
  void step(HybridTimers & timers);
  void synchronize() const;

  void gather_mask(std::vector<std::uint8_t> & global_mask, int root = 0) const;
  void gather_fields(std::vector<double> & rho_v,
                     std::vector<double> & ux_v,
                     std::vector<double> & uy_v,
                     std::vector<double> & vort_v,
                     int root = 0) const;
  void probe_velocity(std::size_t x, std::size_t y,
                      double & ux_out, double & uy_out,
                      int root = 0) const;

  // Serial-style accessors. Only meaningful on the rank that owns the row.
  // vorticity() uses neighboring rows and is therefore only safe for single-rank
  // diagnostics; multi-rank output should use gather_fields().
  double rho      (std::size_t x, std::size_t y) const;
  double ux       (std::size_t x, std::size_t y) const;
  double uy       (std::size_t x, std::size_t y) const;
  double vorticity(std::size_t x, std::size_t y) const;
  bool   is_solid (std::size_t x, std::size_t y) const;

  int rank() const { return rank_; }
  int size() const { return size_; }
  int device() const { return device_; }

  std::size_t nx() const { return nx_; }
  std::size_t ny() const { return ny_global_; }
  std::size_t local_ny() const { return local_ny_; }
  std::size_t y_begin() const { return y0_; }
  std::size_t pitch() const { return pitch_; }
  std::size_t pitch_align() const { return pitch_align_; }
  double tau() const { return tau_; }
  double u_in() const { return u_in_; }
  const std::string & halo_mode() const { return halo_mode_; }

private:
  static std::size_t round_up(std::size_t n, std::size_t a)
  {
    return (a == 0) ? n : ((n + a - 1) / a) * a;
  }

  std::size_t local_rows() const { return local_ny_ + 2; }
  std::size_t idx_local(std::size_t x, std::size_t ly) const { return ly * nx_ + x; }
  std::size_t pidx(std::size_t x, std::size_t ly) const { return ly * pitch_ + x; }
  std::size_t fidx(int q, std::size_t x, std::size_t ly) const { return std::size_t(q) * plane_ + pidx(x, ly); }

  bool owns_global_y(std::size_t gy) const { return gy >= y0_ && gy < y0_ + local_ny_; }
  std::size_t local_y_from_global(std::size_t gy) const { return gy - y0_ + 1; }
  long global_y_from_local(std::size_t ly) const { return long(y0_) + long(ly) - 1; }

  void select_device();
  void mark_obstacle(double c_x, double c_y, double r);
  void allocate_device();
  void copy_host_to_device();
  void sync_host() const;
  void free_device();

  double * row_ptr(double * base, int q, std::size_t ly) const
  {
    return base + std::size_t(q) * plane_ + ly * pitch_;
  }
  const double * row_ptr(const double * base, int q, std::size_t ly) const
  {
    return base + std::size_t(q) * plane_ + ly * pitch_;
  }

  void exchange_halos_cuda_aware(HybridTimers & timers);
  void exchange_halos_host_staged(HybridTimers & timers);
  void compute_rows(std::size_t ly_begin, std::size_t ly_end, HybridTimers & timers);
  void apply_boundaries(HybridTimers & timers);
  void compute_local_fields(std::vector<double> & rho_local,
                            std::vector<double> & ux_local,
                            std::vector<double> & uy_local) const;
  void gather_layout(std::vector<int> & counts, std::vector<int> & displs) const;

  std::size_t nx_;
  std::size_t ny_global_;
  std::size_t local_ny_;
  std::size_t y0_;
  std::size_t pitch_align_;
  std::size_t pitch_;
  std::size_t plane_;

  double u_in_;
  double tau_;

  MPI_Comm comm_;
  MPI_Comm local_comm_;
  int rank_;
  int size_;
  int local_rank_;
  int local_size_;
  // up_rank_ is the low-y neighbor rank (rank-1); down_rank_ is the high-y
  // neighbor rank (rank+1). The names are kept to match the original MPI code.
  int up_rank_;
  int down_rank_;
  int device_;
  int device_count_;

  std::string halo_mode_; // "cuda" or "staged"

  unsigned int block_x_;
  unsigned int block_y_;

  mutable std::vector<double>       f_;
  std::vector<std::uint8_t>         solid_;

  double       * d_f_;
  double       * d_next_;
  std::uint8_t * d_solid_;

  // Contiguous halo buffers. For halo=cuda these are device buffers passed
  // directly to MPI as one packed message per neighbor direction. For
  // halo=staged the host buffers are pinned and are used for explicit staging.
  double * d_send_top_;
  double * d_send_bottom_;
  double * d_recv_top_;
  double * d_recv_bottom_;

  double * h_send_top_;
  double * h_send_bottom_;
  double * h_recv_top_;
  double * h_recv_bottom_;

  double * d_probe_;
  double * h_probe_;

  bool device_allocated_;
  mutable bool host_current_;
  mutable bool solid_probe_warning_printed_;
};

#endif
