#include "lbm.hh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err__ = (call);                                                \
    if (err__ != cudaSuccess) {                                                \
      std::cerr << "CUDA error at " << __FILE__ << ':' << __LINE__             \
                << ": " << cudaGetErrorString(err__) << std::endl;            \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

#ifdef LBM_CUDA_DEBUG
#define CUDA_KERNEL_CHECK()                                                    \
  do {                                                                         \
    CUDA_CHECK(cudaGetLastError());                                             \
    CUDA_CHECK(cudaDeviceSynchronize());                                        \
  } while (0)
#else
#define CUDA_KERNEL_CHECK() CUDA_CHECK(cudaGetLastError())
#endif

// D2Q9 lattice constants. Indexing convention:
//   0: rest         5: NE
//   1: E            6: NW
//   2: N            7: SW
//   3: W            8: SE
//   4: S
const int LBM::cx[9] = { 0,  1,  0, -1,  0,  1, -1, -1,  1};
const int LBM::cy[9] = { 0,  0,  1,  0, -1,  1,  1, -1, -1};

const double LBM::w[9] = {
  4.0 / 9.0,
  1.0 / 9.0,  1.0 / 9.0,  1.0 / 9.0,  1.0 / 9.0,
  1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0
};

const int LBM::opp[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6};

namespace {

__constant__ int    d_cx[9]  = { 0,  1,  0, -1,  0,  1, -1, -1,  1};
__constant__ int    d_cy[9]  = { 0,  0,  1,  0, -1,  1,  1, -1, -1};
__constant__ int    d_opp[9] = { 0,  3,  4,  1,  2,  7,  8,  5,  6};
__constant__ double d_w [9]  = {
  4.0 / 9.0,
  1.0 / 9.0,  1.0 / 9.0,  1.0 / 9.0,  1.0 / 9.0,
  1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0
};

__device__ __forceinline__ std::size_t
pidx2d(std::size_t x, std::size_t y, std::size_t pitch)
{
  return y * pitch + x;
}

__device__ __forceinline__ std::size_t
sidx2d(std::size_t x, std::size_t y, std::size_t nx)
{
  return y * nx + x;
}

/**
 * Fused pull-streaming + collision kernel.
 *
 * Each thread owns one destination cell (x,y). It first pulls the nine incoming
 * populations from neighboring cells in f_old into registers, then computes the
 * BGK collision locally and writes the post-collision populations to f_new at
 * the same destination cell. This avoids the separate collide and stream global
 * memory passes used by the direct baseline.
 *
 * Boundary handling:
 *   - solid cells store a simple reflected state and are ignored in macroscopic
 *     post-processing through the solid mask;
 *   - if a fluid cell pulls from a solid or outside-domain source, it uses
 *     on-link bounce-back from its own opposite population;
 *   - inlet and outlet are applied by a small follow-up boundary kernel;
 *   - optional probe recording is fused into the boundary kernel to avoid a
 *     one-thread kernel launch at every sampled step.
 */
__global__ void
pull_collide_kernel(const double * __restrict__ f_old,
                    double       * __restrict__ f_new,
                    const std::uint8_t * __restrict__ solid,
                    std::size_t nx, std::size_t ny,
                    std::size_t pitch, double inv_tau)
{
  const std::size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= nx || y >= ny) return;

  const std::size_t plane = pitch * ny;
  const std::size_t k     = pidx2d(x, y, pitch);
  const std::size_t sk    = sidx2d(x, y, nx);

  // Keep solid cells defined. Fluid cells next to solids use on-link bounce-back
  // below, so solid distributions are not part of the fluid update.
  if (solid[sk]) {
#pragma unroll
    for (int q = 0; q < 9; ++q) {
      f_new[q * plane + k] = f_old[d_opp[q] * plane + k];
    }
    return;
  }

  double fin[9];

#pragma unroll
  for (int q = 0; q < 9; ++q) {
    const std::int64_t sx = static_cast<std::int64_t>(x) - static_cast<std::int64_t>(d_cx[q]);
    const std::int64_t sy = static_cast<std::int64_t>(y) - static_cast<std::int64_t>(d_cy[q]);

    if (sx >= 0 && sx < static_cast<std::int64_t>(nx) && sy >= 0 && sy < static_cast<std::int64_t>(ny)) {
      const std::size_t src_s = sidx2d(std::size_t(sx), std::size_t(sy), nx);
      if (!solid[src_s]) {
        const std::size_t src = pidx2d(std::size_t(sx), std::size_t(sy), pitch);
        fin[q] = f_old[q * plane + src];
      } else {
        // Pulling from a solid neighbor: bounce the opposite population from
        // the current fluid cell. This is the common on-link no-slip treatment.
        fin[q] = f_old[d_opp[q] * plane + k];
      }
    } else {
      // Outside the domain. Inlet/outlet are overwritten after this kernel;
      // top/bottom are solid walls, so use the same reflected value.
      fin[q] = f_old[d_opp[q] * plane + k];
    }
  }

  double rho = 0.0;
  double mx  = 0.0;
  double my  = 0.0;

#pragma unroll
  for (int q = 0; q < 9; ++q) {
    rho += fin[q];
    mx  += double(d_cx[q]) * fin[q];
    my  += double(d_cy[q]) * fin[q];
  }

  const double ux = (rho > 0.0) ? mx / rho : 0.0;
  const double uy = (rho > 0.0) ? my / rho : 0.0;
  const double u2 = ux * ux + uy * uy;

#pragma unroll
  for (int q = 0; q < 9; ++q) {
    const double cu  = double(d_cx[q]) * ux + double(d_cy[q]) * uy;
    const double feq = d_w[q] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2);
    f_new[q * plane + k] = fin[q] - inv_tau * (fin[q] - feq);
  }
}

__global__ void
apply_boundary_kernel(double * f, const std::uint8_t * solid,
                      std::size_t nx, std::size_t ny,
                      std::size_t pitch, double u_in,
                      bool record_probe,
                      std::size_t px, std::size_t py,
                      unsigned long long step_number,
                      std::size_t sample_index,
                      unsigned long long * probe_step,
                      double * probe_ux,
                      double * probe_uy)
{
  const std::size_t y = blockIdx.x * blockDim.x + threadIdx.x;
  if (y >= ny) return;

  const std::size_t plane = pitch * ny;

  // Inlet: set x=0 to equilibrium with rho=1, u=(u_in,0), except solid wall cells.
  {
    const std::size_t x = 0;
    const std::size_t sk = sidx2d(x, y, nx);
    if (!solid[sk]) {
      const std::size_t k = pidx2d(x, y, pitch);
      const double rho = 1.0;
      const double ux  = u_in;
      const double uy  = 0.0;
      const double u2  = ux * ux + uy * uy;
#pragma unroll
      for (int q = 0; q < 9; ++q) {
        const double cu = double(d_cx[q]) * ux + double(d_cy[q]) * uy;
        f[q * plane + k] = d_w[q] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2);
      }
    }
  }

  // Outlet: zero-gradient copy of the second-to-last column into the last.
  // Keep solid wall cells untouched so the distribution state stays consistent
  // with the solid mask at the top and bottom boundaries.
  if (nx >= 2) {
    const std::size_t x  = nx - 1;
    const std::size_t sk = sidx2d(x, y, nx);
    if (!solid[sk]) {
      const std::size_t xs = nx - 2;
      const std::size_t k  = pidx2d(x,  y, pitch);
      const std::size_t ks = pidx2d(xs, y, pitch);
#pragma unroll
      for (int q = 0; q < 9; ++q) {
        f[q * plane + k] = f[q * plane + ks];
      }
    }
  }

  // Optional device-side probe recording. This is fused into the boundary
  // kernel, which is already launched every time step, avoiding a separate
  // one-thread kernel launch for each probe sample.
  if (record_probe && y == py) {
    const std::size_t k = pidx2d(px, py, pitch);
    double rho = 0.0;
    double mx  = 0.0;
    double my  = 0.0;

#pragma unroll
    for (int q = 0; q < 9; ++q) {
      const double fq = f[q * plane + k];
      rho += fq;
      mx  += double(d_cx[q]) * fq;
      my  += double(d_cy[q]) * fq;
    }

    probe_step[sample_index] = step_number;
    probe_ux  [sample_index] = (rho > 0.0) ? mx / rho : 0.0;
    probe_uy  [sample_index] = (rho > 0.0) ? my / rho : 0.0;
  }
}

template <typename T>
void
cuda_free_noexcept(T *& ptr, const char * name) noexcept
{
  if (!ptr) return;
  const cudaError_t err = cudaFree(ptr);
  if (err != cudaSuccess) {
    std::cerr << "CUDA warning while freeing " << name << " at "
              << __FILE__ << ':' << __LINE__ << ": "
              << cudaGetErrorString(err) << std::endl;
  }
  ptr = nullptr;
}

}  // namespace

LBM::LBM(std::size_t nx, std::size_t ny,
         double u_in, double Re,
         double cyl_x, double cyl_y, double cyl_r,
         std::size_t pitch_align)
  : nx_(nx), ny_(ny),
    pitch_align_(pitch_align == 0 ? 1 : pitch_align),
    pitch_(round_up(nx, pitch_align_)),
    plane_(pitch_ * ny),
    u_in_(u_in), tau_(0.0),
    block_x_(32), block_y_(8),
    f_(9 * plane_, 0.0),
    solid_(nx * ny, 0),
    d_f_(nullptr), d_next_(nullptr), d_solid_(nullptr),
    probe_x_(0), probe_y_(0), probe_every_(0), probe_samples_(0),
    d_probe_step_(nullptr), d_probe_ux_(nullptr), d_probe_uy_(nullptr),
    device_allocated_(false), host_current_(true)
{
  if (nx_ < 3 || ny_ < 3) {
    throw std::runtime_error("Invalid grid size: nx and ny must both be at least 3");
  }
  if (!std::isfinite(u_in_) || u_in_ <= 0.0) {
    throw std::runtime_error("Invalid inlet velocity: u_in must be finite and > 0");
  }
  if (!std::isfinite(Re) || Re <= 0.0) {
    throw std::runtime_error("Invalid Reynolds number: re must be finite and > 0");
  }
  if (!std::isfinite(cyl_r) || cyl_r <= 0.0) {
    throw std::runtime_error("Invalid cylinder radius: cyl_r must be finite and > 0");
  }

  const double nu = u_in_ * (2.0 * cyl_r) / Re;
  tau_ = 3.0 * nu + 0.5;
  if (!std::isfinite(tau_) || tau_ <= 0.5) {
    throw std::runtime_error("Unstable LBM parameters: tau must be finite and > 0.5");
  }

  // No-slip top and bottom walls.
  for (std::size_t x = 0; x < nx_; ++x) {
    solid_[idx(x, 0)]        = 1;
    solid_[idx(x, ny_ - 1)]  = 1;
  }
  mark_obstacle(cyl_x, cyl_y, cyl_r);
}

LBM::~LBM()
{
  free_device();
}

void
LBM::set_cuda_block(unsigned int block_x, unsigned int block_y)
{
  if (block_x == 0 || block_y == 0 || block_x * block_y > 1024) {
    throw std::runtime_error("Invalid CUDA block shape: block_x and block_y must be positive and block_x*block_y <= 1024");
  }
  block_x_ = block_x;
  block_y_ = block_y;
}

void
LBM::add_second_cylinder(double cyl2_x, double cyl2_y, double cyl2_r)
{
  if (cyl2_r > 0.0) {
    mark_obstacle(cyl2_x, cyl2_y, cyl2_r);
    if (device_allocated_) {
      CUDA_CHECK(cudaMemcpy(d_solid_, solid_.data(), solid_.size() * sizeof(std::uint8_t),
                            cudaMemcpyHostToDevice));
    }
  }
}

void
LBM::mark_obstacle(double c_x, double c_y, double r)
{
  const double r2 = r * r;
  for (std::size_t y = 0; y < ny_; ++y) {
    for (std::size_t x = 0; x < nx_; ++x) {
      const double dx = double(x) - c_x;
      const double dy = double(y) - c_y;
      if (dx * dx + dy * dy <= r2) solid_[idx(x, y)] = 1;
    }
  }
}

void
LBM::allocate_device()
{
  if (device_allocated_) return;

  CUDA_CHECK(cudaMalloc(&d_f_,     9 * plane_ * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_next_,  9 * plane_ * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_solid_, solid_.size() * sizeof(std::uint8_t)));
  device_allocated_ = true;
}

void
LBM::copy_host_to_device()
{
  allocate_device();
  CUDA_CHECK(cudaMemcpy(d_f_,     f_.data(),     9 * plane_ * sizeof(double),       cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_next_,  0,             9 * plane_ * sizeof(double)));
  CUDA_CHECK(cudaMemcpy(d_solid_, solid_.data(), solid_.size() * sizeof(std::uint8_t), cudaMemcpyHostToDevice));
  host_current_ = true;
}

void
LBM::free_device()
{
  // free_device() is called from the destructor, so it must not terminate the
  // program through CUDA_CHECK/std::exit. Report cleanup errors, clear the
  // pointers, and let normal shutdown continue.
  cuda_free_noexcept(d_f_,          "d_f_");
  cuda_free_noexcept(d_next_,       "d_next_");
  cuda_free_noexcept(d_solid_,      "d_solid_");
  cuda_free_noexcept(d_probe_step_, "d_probe_step_");
  cuda_free_noexcept(d_probe_ux_,   "d_probe_ux_");
  cuda_free_noexcept(d_probe_uy_,   "d_probe_uy_");

  device_allocated_ = false;
}

void
LBM::initialize()
{
  std::fill(f_.begin(), f_.end(), 0.0);

  for (std::size_t y = 0; y < ny_; ++y) {
    for (std::size_t x = 0; x < nx_; ++x) {
      const double rho = 1.0;
      const double ux  = solid_[idx(x, y)] ? 0.0 : u_in_;
      const double uy  = 0.0;
      const double u2  = ux * ux + uy * uy;
      for (int q = 0; q < Q; ++q) {
        const double cu  = cx[q] * ux + cy[q] * uy;
        const double feq = w[q] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2);
        f_[fidx(q, x, y)] = feq;
      }
    }
  }

  host_current_ = true;
  copy_host_to_device();
}

void
LBM::configure_probe(std::size_t probe_x, std::size_t probe_y,
                     std::size_t steps, std::size_t probe_every)
{
  if (probe_every > 0 && (probe_x >= nx_ || probe_y >= ny_)) {
    throw std::runtime_error("Probe location is outside the simulation domain");
  }
  if (probe_every > 0 && solid_[idx(probe_x, probe_y)]) {
    throw std::runtime_error("Probe location is inside a solid cell");
  }

  probe_x_ = probe_x;
  probe_y_ = probe_y;
  probe_every_ = probe_every;
  probe_samples_ = (probe_every_ == 0) ? 0 : (steps / probe_every_);

  if (d_probe_step_) { CUDA_CHECK(cudaFree(d_probe_step_)); d_probe_step_ = nullptr; }
  if (d_probe_ux_)   { CUDA_CHECK(cudaFree(d_probe_ux_));   d_probe_ux_   = nullptr; }
  if (d_probe_uy_)   { CUDA_CHECK(cudaFree(d_probe_uy_));   d_probe_uy_   = nullptr; }

  if (probe_samples_ > 0) {
    CUDA_CHECK(cudaMalloc(&d_probe_step_, probe_samples_ * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_probe_ux_,   probe_samples_ * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_probe_uy_,   probe_samples_ * sizeof(double)));
  }
}

void
LBM::step(std::size_t step_number)
{
  if (!device_allocated_) copy_host_to_device();

  const dim3 block(block_x_, block_y_);
  const dim3 grid((unsigned int)((nx_ + block.x - 1) / block.x),
                  (unsigned int)((ny_ + block.y - 1) / block.y));

  const unsigned int boundary_threads = 256;
  const unsigned int boundary_blocks  = (unsigned int)((ny_ + boundary_threads - 1) / boundary_threads);

  const double inv_tau = 1.0 / tau_;

  const bool record_probe =
    (probe_every_ > 0 && step_number > 0 && step_number % probe_every_ == 0);
  const std::size_t sample_index = record_probe ? (step_number / probe_every_ - 1) : 0;
  const bool valid_probe_sample = record_probe && sample_index < probe_samples_;

  pull_collide_kernel<<<grid, block>>>(d_f_, d_next_, d_solid_, nx_, ny_, pitch_, inv_tau);
  CUDA_KERNEL_CHECK();

  apply_boundary_kernel<<<boundary_blocks, boundary_threads>>>(
    d_next_, d_solid_, nx_, ny_, pitch_, u_in_,
    valid_probe_sample, probe_x_, probe_y_, static_cast<unsigned long long>(step_number), sample_index,
    d_probe_step_, d_probe_ux_, d_probe_uy_);
  CUDA_KERNEL_CHECK();

  std::swap(d_f_, d_next_);

  host_current_ = false;
}

void
LBM::write_probe_csv(const std::string & path) const
{
  std::ofstream probe(path.c_str());
  probe << "step,ux,uy\n";

  if (probe_samples_ == 0) return;

  std::vector<unsigned long long> h_step(probe_samples_);
  std::vector<double> h_ux(probe_samples_);
  std::vector<double> h_uy(probe_samples_);

  CUDA_CHECK(cudaMemcpy(h_step.data(), d_probe_step_, probe_samples_ * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_ux.data(),   d_probe_ux_,   probe_samples_ * sizeof(double),             cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_uy.data(),   d_probe_uy_,   probe_samples_ * sizeof(double),             cudaMemcpyDeviceToHost));

  for (std::size_t i = 0; i < probe_samples_; ++i) {
    probe << h_step[i] << ',' << h_ux[i] << ',' << h_uy[i] << '\n';
  }
}

void
LBM::synchronize() const
{
  CUDA_CHECK(cudaDeviceSynchronize());
}

void
LBM::sync_host() const
{
  if (host_current_) return;
  CUDA_CHECK(cudaMemcpy(f_.data(), d_f_, 9 * plane_ * sizeof(double), cudaMemcpyDeviceToHost));
  host_current_ = true;
}


void
LBM::compute_macroscopic_fields(std::vector<double> & rho_v,
                                std::vector<double> & ux_v,
                                std::vector<double> & uy_v,
                                std::vector<double> & vorticity_v) const
{
  sync_host();

  const std::size_t n = nx_ * ny_;
  rho_v.assign(n, 0.0);
  ux_v.assign(n, 0.0);
  uy_v.assign(n, 0.0);
  vorticity_v.assign(n, 0.0);

  for (std::size_t y = 0; y < ny_; ++y) {
    for (std::size_t x = 0; x < nx_; ++x) {
      const std::size_t out = idx(x, y);
      if (solid_[out]) {
        // Keep obstacle/wall visualization clean and avoid plotting artificial
        // velocities inside no-slip cells. The solid mask is written separately.
        rho_v[out] = 1.0;
        ux_v[out]  = 0.0;
        uy_v[out]  = 0.0;
        continue;
      }

      double rho = 0.0;
      double mx  = 0.0;
      double my  = 0.0;

      for (int q = 0; q < Q; ++q) {
        const double fq = f_[fidx(q, x, y)];
        rho += fq;
        mx  += cx[q] * fq;
        my  += cy[q] * fq;
      }

      rho_v[out] = rho;
      if (rho > 0.0) {
        ux_v[out] = mx / rho;
        uy_v[out] = my / rho;
      }
    }
  }

  if (nx_ < 3 || ny_ < 3) return;

  for (std::size_t y = 1; y + 1 < ny_; ++y) {
    for (std::size_t x = 1; x + 1 < nx_; ++x) {
      const std::size_t out = idx(x, y);
      if (solid_[out]) continue;
      vorticity_v[out] = 0.5 * (
        (uy_v[idx(x + 1, y)] - uy_v[idx(x - 1, y)]) -
        (ux_v[idx(x, y + 1)] - ux_v[idx(x, y - 1)]));
    }
  }
}

double
LBM::rho(std::size_t x, std::size_t y) const
{
  if (solid_[idx(x, y)]) return 1.0;
  sync_host();
  double r = 0.0;
  for (int q = 0; q < Q; ++q) r += f_[q * plane_ + pidx(x, y)];
  return r;
}

double
LBM::ux(std::size_t x, std::size_t y) const
{
  if (solid_[idx(x, y)]) return 0.0;
  sync_host();
  double r = 0.0, m = 0.0;
  for (int q = 0; q < Q; ++q) {
    const double fq = f_[q * plane_ + pidx(x, y)];
    r += fq;
    m += cx[q] * fq;
  }
  return (r > 0.0) ? m / r : 0.0;
}

double
LBM::uy(std::size_t x, std::size_t y) const
{
  if (solid_[idx(x, y)]) return 0.0;
  sync_host();
  double r = 0.0, m = 0.0;
  for (int q = 0; q < Q; ++q) {
    const double fq = f_[q * plane_ + pidx(x, y)];
    r += fq;
    m += cy[q] * fq;
  }
  return (r > 0.0) ? m / r : 0.0;
}

double
LBM::vorticity(std::size_t x, std::size_t y) const
{
  if (x == 0 || x == nx_ - 1 || y == 0 || y == ny_ - 1) return 0.0;
  if (solid_[idx(x, y)]) return 0.0;
  return 0.5 * ((uy(x + 1, y) - uy(x - 1, y)) - (ux(x, y + 1) - ux(x, y - 1)));
}

bool
LBM::is_solid(std::size_t x, std::size_t y) const
{
  return solid_[idx(x, y)] != 0;
}
