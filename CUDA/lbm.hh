#ifndef LBM_HH
#define LBM_HH

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

/**
 * @brief Optimized CUDA D2Q9 LBM solver.
 *
 * Optimizations compared with the direct CUDA baseline:
 *   1. fused pull-streaming + collision kernel,
 *   2. double buffering, one thread writes one output cell,
 *   3. optional device-side probe buffering, copied to host once at the end,
 *   4. padded row pitch for better row alignment.
 *
 * Distribution layout is Structure of Arrays with a padded row pitch:
 *   f[q * (pitch * ny) + y * pitch + x], q in [0, 9), x < nx.
 * Padding cells x >= nx are unused.
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
      std::size_t pitch_align = 32);
  ~LBM();

  LBM(const LBM &) = delete;
  LBM & operator=(const LBM &) = delete;

  /// Configure the CUDA block shape used by the main 2D kernel.
  /// Good values to sweep include (16,16), (32,4), (32,8), (32,16), (64,4).
  void set_cuda_block(unsigned int block_x, unsigned int block_y);
  unsigned int block_x() const { return block_x_; }
  unsigned int block_y() const { return block_y_; }

  /// Add a second circular obstacle. No-op if r2 <= 0. For a clean initial
  /// equilibrium, call this before initialize(); main.cc does this explicitly.
  void add_second_cylinder(double cyl2_x, double cyl2_y, double cyl2_r);

  /// Initialize host distributions, then copy them to the GPU.
  void initialize();

  /// Allocate device-side probe buffers. If probe_every == 0, probe recording is disabled.
  void configure_probe(std::size_t probe_x, std::size_t probe_y,
                       std::size_t steps, std::size_t probe_every);

  /// Advance one time step on the GPU. step_number is 1-based and used for probe recording.
  void step(std::size_t step_number = 0);

  /// Write the device-side probe buffer to CSV. Copies probe arrays back once.
  void write_probe_csv(const std::string & path) const;

  /// Explicitly synchronize the device. Useful before stopping a timer.
  void synchronize() const;

  /// Copy the current device distributions back to the host if needed.
  void sync_host() const;

  // Macroscopic accessors. These synchronize the full distribution array on
  // first use after a GPU step, then reuse the host copy.
  /// Fill rho, ux, uy, and vorticity in one host-side pass after one device-host copy.
  /// This is used by the HDF5 writer to avoid redundant per-field recomputation.
  void compute_macroscopic_fields(std::vector<double> & rho_v,
                                  std::vector<double> & ux_v,
                                  std::vector<double> & uy_v,
                                  std::vector<double> & vorticity_v) const;

  double rho      (std::size_t x, std::size_t y) const;
  double ux       (std::size_t x, std::size_t y) const;
  double uy       (std::size_t x, std::size_t y) const;
  double vorticity(std::size_t x, std::size_t y) const;
  bool   is_solid (std::size_t x, std::size_t y) const;

  std::size_t nx()          const { return nx_; }
  std::size_t ny()          const { return ny_; }
  std::size_t pitch()       const { return pitch_; }
  std::size_t pitch_align() const { return pitch_align_; }
  double      tau()         const { return tau_; }
  double      u_in()        const { return u_in_; }

private:
  static std::size_t round_up(std::size_t n, std::size_t a)
  {
    return (a == 0) ? n : ((n + a - 1) / a) * a;
  }

  std::size_t idx       (std::size_t x, std::size_t y) const { return y * nx_ + x; }
  std::size_t pidx      (std::size_t x, std::size_t y) const { return y * pitch_ + x; }
  std::size_t fidx      (int q, std::size_t x, std::size_t y) const { return q * plane_ + pidx(x, y); }

  void mark_obstacle (double c_x, double c_y, double r);
  void allocate_device();
  void copy_host_to_device();
  void free_device();

  std::size_t nx_, ny_;
  std::size_t pitch_align_;
  std::size_t pitch_;   ///< Padded row length, in cells.
  std::size_t plane_;   ///< pitch_ * ny_.
  double u_in_;
  double tau_;

  unsigned int block_x_;
  unsigned int block_y_;

  mutable std::vector<double>       f_;       ///< Host mirror, 9 * pitch_ * ny_.
  std::vector<std::uint8_t>         solid_;   ///< Host solid mask, nx_ * ny_ only.

  double       * d_f_;
  double       * d_next_;
  std::uint8_t * d_solid_;

  // Device-side probe buffering.
  std::size_t probe_x_;
  std::size_t probe_y_;
  std::size_t probe_every_;
  std::size_t probe_samples_;
  unsigned long long * d_probe_step_;
  double             * d_probe_ux_;
  double             * d_probe_uy_;

  bool device_allocated_;
  mutable bool host_current_;
};

#endif  // LBM_HH
