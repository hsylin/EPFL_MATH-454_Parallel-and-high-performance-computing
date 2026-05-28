#include "lbm.hh"

#include <cuda_runtime.h>

#if defined(__has_include)
#  if __has_include(<mpi-ext.h>)
#    include <mpi-ext.h>
#  endif
#endif

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err__ = (call);                                                \
    if (err__ != cudaSuccess) {                                                \
      std::cerr << "CUDA error at " << __FILE__ << ':' << __LINE__             \
                << ": " << cudaGetErrorString(err__) << std::endl;            \
      MPI_Abort(MPI_COMM_WORLD, 1);                                            \
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

const int LBM::cx[9] = { 0,  1,  0, -1,  0,  1, -1, -1,  1};
const int LBM::cy[9] = { 0,  0,  1,  0, -1,  1,  1, -1, -1};
const double LBM::w[9] = {
  4.0 / 9.0,
  1.0 / 9.0,  1.0 / 9.0,  1.0 / 9.0,  1.0 / 9.0,
  1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0
};
const int LBM::opp[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6};

namespace {

constexpr int HALO_Q = 3;
constexpr int CY_POS[HALO_Q] = {2, 5, 6}; // cy = +1 populations crossing a +y row boundary
constexpr int CY_NEG[HALO_Q] = {4, 7, 8}; // cy = -1 populations crossing a -y row boundary
constexpr int TAG_BASE = 7100;
constexpr int TAG_TO_LOW_Y  = TAG_BASE + 1; // message sent toward rank-1 / low-y neighbor
constexpr int TAG_TO_HIGH_Y = TAG_BASE + 2; // message sent toward rank+1 / high-y neighbor

__constant__ int    d_cx[9]  = { 0,  1,  0, -1,  0,  1, -1, -1,  1};
__constant__ int    d_cy[9]  = { 0,  0,  1,  0, -1,  1,  1, -1, -1};
__constant__ int    d_opp[9] = { 0,  3,  4,  1,  2,  7,  8,  5,  6};
__constant__ double d_w [9]  = {
  4.0 / 9.0,
  1.0 / 9.0,  1.0 / 9.0,  1.0 / 9.0,  1.0 / 9.0,
  1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0
};

__device__ __forceinline__ std::size_t pidx2d(std::size_t x, std::size_t y, std::size_t pitch)
{
  return y * pitch + x;
}

__device__ __forceinline__ std::size_t sidx2d(std::size_t x, std::size_t y, std::size_t nx)
{
  return y * nx + x;
}

__global__ void
pull_collide_range_kernel(const double * __restrict__ f_old,
                          double       * __restrict__ f_new,
                          const std::uint8_t * __restrict__ solid,
                          std::size_t nx,
                          std::size_t local_rows,
                          std::size_t pitch,
                          std::size_t ly_begin,
                          std::size_t ly_end,
                          double inv_tau)
{
  if (ly_begin > ly_end) return;

  const std::size_t rows = ly_end - ly_begin + 1;
  const std::size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t r = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= nx || r >= rows) return;

  const std::size_t ly = ly_begin + r;
  const std::size_t plane = pitch * local_rows;
  const std::size_t k = pidx2d(x, ly, pitch);
  const std::size_t sk = sidx2d(x, ly, nx);

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
    const std::int64_t sx  = static_cast<std::int64_t>(x)  - static_cast<std::int64_t>(d_cx[q]);
    const std::int64_t sly = static_cast<std::int64_t>(ly) - static_cast<std::int64_t>(d_cy[q]);

    if (sx >= 0 && sx < static_cast<std::int64_t>(nx) &&
        sly >= 0 && sly < static_cast<std::int64_t>(local_rows)) {
      const std::size_t src_s = sidx2d(std::size_t(sx), std::size_t(sly), nx);
      if (!solid[src_s]) {
        const std::size_t src = pidx2d(std::size_t(sx), std::size_t(sly), pitch);
        fin[q] = f_old[q * plane + src];
      } else {
        fin[q] = f_old[d_opp[q] * plane + k];
      }
    } else {
      fin[q] = f_old[d_opp[q] * plane + k];
    }
  }

  double rho = 0.0;
  double mx = 0.0;
  double my = 0.0;
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
    const double cu = double(d_cx[q]) * ux + double(d_cy[q]) * uy;
    const double feq = d_w[q] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2);
    f_new[q * plane + k] = fin[q] - inv_tau * (fin[q] - feq);
  }
}

__global__ void
apply_boundary_kernel(double * f,
                      const std::uint8_t * __restrict__ solid,
                      std::size_t nx,
                      std::size_t local_rows,
                      std::size_t pitch,
                      std::size_t local_ny,
                      double u_in)
{
  const std::size_t owned_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (owned_idx >= local_ny) return;

  const std::size_t ly = owned_idx + 1;
  const std::size_t plane = pitch * local_rows;

  // Inlet, x = 0.
  {
    const std::size_t x = 0;
    const std::size_t sk = sidx2d(x, ly, nx);
    if (!solid[sk]) {
      const std::size_t k = pidx2d(x, ly, pitch);
      const double rho = 1.0;
      const double ux = u_in;
      const double uy = 0.0;
      const double u2 = ux * ux + uy * uy;
#pragma unroll
      for (int q = 0; q < 9; ++q) {
        const double cu = double(d_cx[q]) * ux + double(d_cy[q]) * uy;
        f[q * plane + k] = d_w[q] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2);
      }
    }
  }

  // Outlet, x = nx - 1: zero-gradient copy from x = nx - 2.
  if (nx >= 2) {
    const std::size_t x = nx - 1;
    const std::size_t sk = sidx2d(x, ly, nx);
    if (!solid[sk]) {
      const std::size_t xs = nx - 2;
      const std::size_t k  = pidx2d(x,  ly, pitch);
      const std::size_t ks = pidx2d(xs, ly, pitch);
#pragma unroll
      for (int q = 0; q < 9; ++q) {
        f[q * plane + k] = f[q * plane + ks];
      }
    }
  }
}

__global__ void
probe_values_kernel(const double * __restrict__ f,
                    double       * __restrict__ out,
                    std::size_t x,
                    std::size_t ly,
                    std::size_t pitch,
                    std::size_t plane)
{
  const int q = int(threadIdx.x);
  if (q < 9) {
    out[q] = f[std::size_t(q) * plane + ly * pitch + x];
  }
}

void cuda_free_noexcept(void * ptr, const char * name) noexcept
{
  if (!ptr) return;
  cudaError_t err = cudaFree(ptr);
  if (err != cudaSuccess) {
    std::cerr << "CUDA warning while freeing " << name << ": "
              << cudaGetErrorString(err) << std::endl;
  }
}

void cuda_free_host_noexcept(void * ptr, const char * name) noexcept
{
  if (!ptr) return;
  cudaError_t err = cudaFreeHost(ptr);
  if (err != cudaSuccess) {
    std::cerr << "CUDA warning while freeing pinned " << name << ": "
              << cudaGetErrorString(err) << std::endl;
  }
}

} // namespace

LBM::LBM(std::size_t nx, std::size_t ny,
         double u_in, double Re,
         double cyl_x, double cyl_y, double cyl_r,
         MPI_Comm comm,
         std::size_t pitch_align,
         const std::string & halo_mode)
  : nx_(nx),
    ny_global_(ny),
    local_ny_(0),
    y0_(0),
    pitch_align_(pitch_align == 0 ? 1 : pitch_align),
    pitch_(0),
    plane_(0),
    u_in_(u_in),
    tau_(0.0),
    comm_(comm),
    local_comm_(MPI_COMM_NULL),
    rank_(0),
    size_(1),
    local_rank_(0),
    local_size_(1),
    up_rank_(MPI_PROC_NULL),
    down_rank_(MPI_PROC_NULL),
    device_(0),
    device_count_(0),
    halo_mode_(halo_mode),
    block_x_(64),
    block_y_(4),
    d_f_(nullptr),
    d_next_(nullptr),
    d_solid_(nullptr),
    d_send_top_(nullptr),
    d_send_bottom_(nullptr),
    d_recv_top_(nullptr),
    d_recv_bottom_(nullptr),
    h_send_top_(nullptr),
    h_send_bottom_(nullptr),
    h_recv_top_(nullptr),
    h_recv_bottom_(nullptr),
    d_probe_(nullptr),
    h_probe_(nullptr),
    device_allocated_(false),
    host_current_(true),
    solid_probe_warning_printed_(false)
{
  MPI_Comm_rank(comm_, &rank_);
  MPI_Comm_size(comm_, &size_);

  MPI_Comm_split_type(comm_, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_comm_);
  MPI_Comm_rank(local_comm_, &local_rank_);
  MPI_Comm_size(local_comm_, &local_size_);

  if (local_size_ != size_) {
    if (rank_ == 0) {
      std::cerr << "This code is intended for intra-node runs. Use --nodes=1.\n";
    }
    MPI_Abort(comm_, 1);
  }

  if (halo_mode_ != "cuda" && halo_mode_ != "staged") {
    if (rank_ == 0) {
      std::cerr << "Invalid halo mode '" << halo_mode_ << "'. Use halo=cuda or halo=staged.\n";
    }
    MPI_Abort(comm_, 1);
  }

  if (nx_ < 3 || ny_global_ < 3 || size_ > int(ny_global_)) {
    if (rank_ == 0) std::cerr << "Invalid grid or rank count. Need nx,ny >= 3 and ranks <= ny.\n";
    MPI_Abort(comm_, 1);
  }
  if (nx_ > std::size_t(std::numeric_limits<int>::max()) ||
      HALO_Q * nx_ > std::size_t(std::numeric_limits<int>::max())) {
    if (rank_ == 0) std::cerr << "Grid too wide for MPI int counts in halo exchange.\n";
    MPI_Abort(comm_, 1);
  }
  if (!std::isfinite(u_in_) || u_in_ <= 0.0 || !std::isfinite(Re) || Re <= 0.0 ||
      !std::isfinite(cyl_r) || cyl_r <= 0.0) {
    if (rank_ == 0) std::cerr << "Invalid physical parameters.\n";
    MPI_Abort(comm_, 1);
  }

  select_device();

#if defined(MPIX_CUDA_AWARE_SUPPORT)
  if (halo_mode_ == "cuda" && !MPIX_Query_cuda_support()) {
    if (rank_ == 0) {
      std::cerr << "halo=cuda requested, but this MPI library reports that CUDA-aware MPI "
                << "is unavailable. Reload a CUDA-aware MPI module or use halo=staged.\n";
    }
    MPI_Abort(comm_, 1);
  }
#endif

  const std::size_t p = std::size_t(size_);
  const std::size_t r = std::size_t(rank_);
  const std::size_t base = ny_global_ / p;
  const std::size_t rem  = ny_global_ % p;
  local_ny_ = base + (r < rem ? 1u : 0u);
  y0_       = r * base + std::min(r, rem);

  if (local_ny_ < 2) {
    if (rank_ == 0) {
      std::cerr << "Invalid decomposition: each rank must own at least two rows "
                << "for this row-decomposed validation/performance code. "
                << "Use fewer MPI ranks or a larger ny.\n";
    }
    MPI_Abort(comm_, 1);
  }

  up_rank_   = (rank_ == 0)         ? MPI_PROC_NULL : rank_ - 1;
  down_rank_ = (rank_ == size_ - 1) ? MPI_PROC_NULL : rank_ + 1;

  pitch_ = round_up(nx_, pitch_align_);
  plane_ = pitch_ * local_rows();

  const double nu = u_in_ * (2.0 * cyl_r) / Re;
  tau_ = 3.0 * nu + 0.5;
  if (!std::isfinite(tau_) || tau_ <= 0.5) {
    if (rank_ == 0) std::cerr << "Unstable LBM parameters: tau <= 0.5.\n";
    MPI_Abort(comm_, 1);
  }

  f_.assign(9 * plane_, 0.0);
  solid_.assign(nx_ * local_rows(), 0);

  // Mark deterministic solid cells on owned rows and valid ghost rows.
  // The pull-streaming kernel tests whether the source cell is solid; therefore
  // ghost-row solid masks must be valid at rank boundaries, especially when the
  // cylinder intersects an MPI split line.
  for (std::size_t ly = 0; ly < local_rows(); ++ly) {
    const long gy = global_y_from_local(ly);
    if (gy < 0 || gy >= long(ny_global_)) continue;
    if (gy == 0 || gy == long(ny_global_) - 1) {
      for (std::size_t x = 0; x < nx_; ++x) solid_[idx_local(x, ly)] = 1;
    }
  }
  mark_obstacle(cyl_x, cyl_y, cyl_r);
}

LBM::~LBM()
{
  free_device();
  int finalized = 0;
  MPI_Finalized(&finalized);
  if (!finalized && local_comm_ != MPI_COMM_NULL) MPI_Comm_free(&local_comm_);
}

void LBM::select_device()
{
  CUDA_CHECK(cudaGetDeviceCount(&device_count_));
  if (device_count_ <= 0) {
    if (rank_ == 0) std::cerr << "No CUDA device visible.\n";
    MPI_Abort(comm_, 1);
  }

  // Slurm often gives each MPI rank a private CUDA_VISIBLE_DEVICES list.
  // In that case, every rank may see only one logical CUDA device, even
  // though the job owns multiple physical GPUs. Then each rank should use
  // visible device 0.
  if (device_count_ == 1) {
    device_ = 0;
  } else {
    if (local_size_ > device_count_) {
      if (rank_ == 0) {
        std::cerr << "Need at least one visible GPU per local MPI rank: local ranks = "
                  << local_size_ << ", visible CUDA devices = " << device_count_
                  << ". Use one node with enough GPUs or adjust Slurm GPU visibility.\n";
      }
      MPI_Abort(comm_, 1);
    }

    // If all GPUs are visible to all ranks, map local rank -> CUDA device.
    device_ = local_rank_;
  }

  CUDA_CHECK(cudaSetDevice(device_));
}

void LBM::set_cuda_block(unsigned int block_x, unsigned int block_y)
{
  if (block_x == 0 || block_y == 0 || block_x * block_y > 1024) {
    if (rank_ == 0) std::cerr << "Invalid CUDA block shape.\n";
    MPI_Abort(comm_, 1);
  }
  block_x_ = block_x;
  block_y_ = block_y;
}

void LBM::mark_obstacle(double c_x, double c_y, double r)
{
  const double r2 = r * r;
  // Include valid ghost rows so that a fluid cell near an MPI rank boundary can
  // correctly detect a solid source cell during pull-streaming bounce-back.
  for (std::size_t ly = 0; ly < local_rows(); ++ly) {
    const long gy_l = global_y_from_local(ly);
    if (gy_l < 0 || gy_l >= long(ny_global_)) continue;
    const double gy = double(gy_l);
    for (std::size_t x = 0; x < nx_; ++x) {
      const double dx = double(x) - c_x;
      const double dy = gy - c_y;
      if (dx * dx + dy * dy <= r2) solid_[idx_local(x, ly)] = 1;
    }
  }
}

void LBM::add_second_cylinder(double cyl2_x, double cyl2_y, double cyl2_r)
{
  if (cyl2_r <= 0.0) return;
  mark_obstacle(cyl2_x, cyl2_y, cyl2_r);
  if (device_allocated_) {
    CUDA_CHECK(cudaSetDevice(device_));
    CUDA_CHECK(cudaMemcpy(d_solid_, solid_.data(), solid_.size() * sizeof(std::uint8_t), cudaMemcpyHostToDevice));
  }
}

void LBM::allocate_device()
{
  if (device_allocated_) return;
  CUDA_CHECK(cudaSetDevice(device_));
  CUDA_CHECK(cudaMalloc(&d_f_,     9 * plane_ * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_next_,  9 * plane_ * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_solid_, solid_.size() * sizeof(std::uint8_t)));
  CUDA_CHECK(cudaMalloc(&d_probe_, Q * sizeof(double)));
  CUDA_CHECK(cudaMallocHost(&h_probe_, Q * sizeof(double)));

  const std::size_t halo_bytes = HALO_Q * nx_ * sizeof(double);
  if (halo_mode_ == "cuda") {
    // Packed device-side halo buffers: CUDA-aware MPI sends one contiguous
    // device buffer per neighbor direction. This matches the staged path's
    // MPI message count and message size, so the comparison differs mainly in
    // whether explicit D<->H staging is used.
    CUDA_CHECK(cudaMalloc(&d_send_top_,    halo_bytes));
    CUDA_CHECK(cudaMalloc(&d_send_bottom_, halo_bytes));
    CUDA_CHECK(cudaMalloc(&d_recv_top_,    halo_bytes));
    CUDA_CHECK(cudaMalloc(&d_recv_bottom_, halo_bytes));
  } else if (halo_mode_ == "staged") {
    CUDA_CHECK(cudaMallocHost(&h_send_top_,    halo_bytes));
    CUDA_CHECK(cudaMallocHost(&h_send_bottom_, halo_bytes));
    CUDA_CHECK(cudaMallocHost(&h_recv_top_,    halo_bytes));
    CUDA_CHECK(cudaMallocHost(&h_recv_bottom_, halo_bytes));
  }
  device_allocated_ = true;
}

void LBM::copy_host_to_device()
{
  allocate_device();
  CUDA_CHECK(cudaSetDevice(device_));
  CUDA_CHECK(cudaMemcpy(d_f_,     f_.data(),     9 * plane_ * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_next_,  0,             9 * plane_ * sizeof(double)));
  CUDA_CHECK(cudaMemcpy(d_solid_, solid_.data(), solid_.size() * sizeof(std::uint8_t), cudaMemcpyHostToDevice));
  host_current_ = true;
}

void LBM::free_device()
{
  if (device_count_ > 0) cudaSetDevice(device_);
  cuda_free_noexcept(d_f_,     "d_f_");       d_f_ = nullptr;
  cuda_free_noexcept(d_next_,  "d_next_");    d_next_ = nullptr;
  cuda_free_noexcept(d_solid_, "d_solid_");   d_solid_ = nullptr;
  cuda_free_noexcept(d_send_top_,    "d_send_top_");    d_send_top_ = nullptr;
  cuda_free_noexcept(d_send_bottom_, "d_send_bottom_"); d_send_bottom_ = nullptr;
  cuda_free_noexcept(d_recv_top_,    "d_recv_top_");    d_recv_top_ = nullptr;
  cuda_free_noexcept(d_recv_bottom_, "d_recv_bottom_"); d_recv_bottom_ = nullptr;
  cuda_free_noexcept(d_probe_, "d_probe_");   d_probe_ = nullptr;
  cuda_free_host_noexcept(h_probe_, "h_probe_"); h_probe_ = nullptr;
  cuda_free_host_noexcept(h_send_top_,    "h_send_top_");    h_send_top_ = nullptr;
  cuda_free_host_noexcept(h_send_bottom_, "h_send_bottom_"); h_send_bottom_ = nullptr;
  cuda_free_host_noexcept(h_recv_top_,    "h_recv_top_");    h_recv_top_ = nullptr;
  cuda_free_host_noexcept(h_recv_bottom_, "h_recv_bottom_"); h_recv_bottom_ = nullptr;
  device_allocated_ = false;
}

void LBM::initialize()
{
  std::fill(f_.begin(), f_.end(), 0.0);
  for (std::size_t ly = 0; ly < local_rows(); ++ly) {
    for (std::size_t x = 0; x < nx_; ++x) {
      const bool solid = solid_[idx_local(x, ly)] != 0;
      const double rho = 1.0;
      const double ux  = solid ? 0.0 : u_in_;
      const double uy  = 0.0;
      const double u2  = ux * ux + uy * uy;
      for (int q = 0; q < Q; ++q) {
        const double cu  = cx[q] * ux + cy[q] * uy;
        const double feq = w[q] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2);
        f_[fidx(q, x, ly)] = feq;
      }
    }
  }
  host_current_ = true;
  copy_host_to_device();
}

void LBM::exchange_halos_cuda_aware(HybridTimers & timers)
{
  if (size_ == 1) return;
  CUDA_CHECK(cudaSetDevice(device_));

  // Ensure all previous default-stream work has completed before MPI reads from
  // or writes to device buffers. The synchronization is intentionally inside
  // the halo timer so cuda-aware and staged modes charge the same pre-exchange stall.
  const double t0 = MPI_Wtime();
  CUDA_CHECK(cudaDeviceSynchronize());

  const std::size_t row_bytes = nx_ * sizeof(double);
  const std::size_t msg_count = HALO_Q * nx_;

  // Pack the same three crossing populations into one contiguous device buffer
  // per neighbor direction. MPI therefore sees the same number of messages and
  // the same number of bytes as the host-staged mode; the intended difference is
  // that halo=cuda passes GPU buffers directly to MPI instead of explicit D<->H
  // staging through pinned host memory.
  if (up_rank_ != MPI_PROC_NULL) {
    for (int j = 0; j < HALO_Q; ++j) {
      CUDA_CHECK(cudaMemcpy(d_send_top_ + std::size_t(j) * nx_,
                            row_ptr(d_f_, CY_NEG[j], 1),
                            row_bytes, cudaMemcpyDeviceToDevice));
    }
  }
  if (down_rank_ != MPI_PROC_NULL) {
    for (int j = 0; j < HALO_Q; ++j) {
      CUDA_CHECK(cudaMemcpy(d_send_bottom_ + std::size_t(j) * nx_,
                            row_ptr(d_f_, CY_POS[j], local_ny_),
                            row_bytes, cudaMemcpyDeviceToDevice));
    }
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  MPI_Request req[4];
  int nreq = 0;

  if (up_rank_ != MPI_PROC_NULL) {
    // Low-y ghost row receives cy=+1 populations from rank-1's high-y owned row.
    MPI_Irecv(d_recv_top_, int(msg_count), MPI_DOUBLE,
              up_rank_, TAG_TO_HIGH_Y, comm_, &req[nreq++]);
    // Send this rank's low-y owned row cy=-1 populations toward rank-1.
    MPI_Isend(d_send_top_, int(msg_count), MPI_DOUBLE,
              up_rank_, TAG_TO_LOW_Y, comm_, &req[nreq++]);
  }

  if (down_rank_ != MPI_PROC_NULL) {
    // High-y ghost row receives cy=-1 populations from rank+1's low-y owned row.
    MPI_Irecv(d_recv_bottom_, int(msg_count), MPI_DOUBLE,
              down_rank_, TAG_TO_LOW_Y, comm_, &req[nreq++]);
    // Send this rank's high-y owned row cy=+1 populations toward rank+1.
    MPI_Isend(d_send_bottom_, int(msg_count), MPI_DOUBLE,
              down_rank_, TAG_TO_HIGH_Y, comm_, &req[nreq++]);
  }

  if (nreq > 0) MPI_Waitall(nreq, req, MPI_STATUSES_IGNORE);

  if (up_rank_ != MPI_PROC_NULL) {
    for (int j = 0; j < HALO_Q; ++j) {
      CUDA_CHECK(cudaMemcpy(row_ptr(d_f_, CY_POS[j], 0),
                            d_recv_top_ + std::size_t(j) * nx_,
                            row_bytes, cudaMemcpyDeviceToDevice));
    }
  }
  if (down_rank_ != MPI_PROC_NULL) {
    for (int j = 0; j < HALO_Q; ++j) {
      CUDA_CHECK(cudaMemcpy(row_ptr(d_f_, CY_NEG[j], local_ny_ + 1),
                            d_recv_bottom_ + std::size_t(j) * nx_,
                            row_bytes, cudaMemcpyDeviceToDevice));
    }
  }

  // Some CUDA-aware MPI stacks complete GPU-buffer operations on internal
  // streams. Synchronize before the next default-stream kernel reads ghosts.
  CUDA_CHECK(cudaDeviceSynchronize());
  timers.comm += MPI_Wtime() - t0;
}

void LBM::exchange_halos_host_staged(HybridTimers & timers)
{
  if (size_ == 1) return;
  CUDA_CHECK(cudaSetDevice(device_));
  const double t0 = MPI_Wtime();

  const std::size_t row_bytes = nx_ * sizeof(double);
  const std::size_t msg_count = HALO_Q * nx_;

  // Fairness with the CUDA-aware mode: this staged path uses the same packed
  // MPI message structure, i.e. one contiguous 3-row message per neighbor
  // direction with the same tags and byte count. The only intended difference
  // is explicit device<->host staging through pinned host buffers.
  if (up_rank_ != MPI_PROC_NULL) {
    for (int j = 0; j < HALO_Q; ++j) {
      CUDA_CHECK(cudaMemcpy(h_send_top_ + std::size_t(j) * nx_,
                            row_ptr(d_f_, CY_NEG[j], 1),
                            row_bytes, cudaMemcpyDeviceToHost));
    }
  }
  if (down_rank_ != MPI_PROC_NULL) {
    for (int j = 0; j < HALO_Q; ++j) {
      CUDA_CHECK(cudaMemcpy(h_send_bottom_ + std::size_t(j) * nx_,
                            row_ptr(d_f_, CY_POS[j], local_ny_),
                            row_bytes, cudaMemcpyDeviceToHost));
    }
  }


  MPI_Request req[4];
  int nreq = 0;

  if (up_rank_ != MPI_PROC_NULL) {
    // Low-y ghost row receives cy=+1 populations from rank-1.
    MPI_Irecv(h_recv_top_, int(msg_count), MPI_DOUBLE,
              up_rank_, TAG_TO_HIGH_Y, comm_, &req[nreq++]);
    // Send this rank's low-y owned row cy=-1 populations to rank-1.
    MPI_Isend(h_send_top_, int(msg_count), MPI_DOUBLE,
              up_rank_, TAG_TO_LOW_Y, comm_, &req[nreq++]);
  }

  if (down_rank_ != MPI_PROC_NULL) {
    // High-y ghost row receives cy=-1 populations from rank+1.
    MPI_Irecv(h_recv_bottom_, int(msg_count), MPI_DOUBLE,
              down_rank_, TAG_TO_LOW_Y, comm_, &req[nreq++]);
    // Send this rank's high-y owned row cy=+1 populations to rank+1.
    MPI_Isend(h_send_bottom_, int(msg_count), MPI_DOUBLE,
              down_rank_, TAG_TO_HIGH_Y, comm_, &req[nreq++]);
  }

  if (nreq > 0) MPI_Waitall(nreq, req, MPI_STATUSES_IGNORE);

  if (up_rank_ != MPI_PROC_NULL) {
    for (int j = 0; j < HALO_Q; ++j) {
      CUDA_CHECK(cudaMemcpy(row_ptr(d_f_, CY_POS[j], 0),
                            h_recv_top_ + std::size_t(j) * nx_,
                            row_bytes, cudaMemcpyHostToDevice));
    }
  }
  if (down_rank_ != MPI_PROC_NULL) {
    for (int j = 0; j < HALO_Q; ++j) {
      CUDA_CHECK(cudaMemcpy(row_ptr(d_f_, CY_NEG[j], local_ny_ + 1),
                            h_recv_bottom_ + std::size_t(j) * nx_,
                            row_bytes, cudaMemcpyHostToDevice));
    }
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  timers.comm += MPI_Wtime() - t0;
}

void LBM::compute_rows(std::size_t ly_begin, std::size_t ly_end, HybridTimers & timers)
{
  if (ly_begin > ly_end) return;
  CUDA_CHECK(cudaSetDevice(device_));
  const std::size_t rows = ly_end - ly_begin + 1;
  const dim3 block(block_x_, block_y_);
  const dim3 grid((unsigned int)((nx_ + block.x - 1) / block.x),
                  (unsigned int)((rows + block.y - 1) / block.y));
  const double inv_tau = 1.0 / tau_;

  const double t0 = MPI_Wtime();
  pull_collide_range_kernel<<<grid, block>>>(d_f_, d_next_, d_solid_,
                                             nx_, local_rows(), pitch_,
                                             ly_begin, ly_end, inv_tau);
  CUDA_KERNEL_CHECK();
  CUDA_CHECK(cudaDeviceSynchronize());
  timers.compute += MPI_Wtime() - t0;
}

void LBM::apply_boundaries(HybridTimers & timers)
{
  CUDA_CHECK(cudaSetDevice(device_));
  const unsigned int threads = std::min<unsigned int>(256u, std::max<unsigned int>(1u, (unsigned int)local_ny_));
  const unsigned int blocks = (unsigned int)((local_ny_ + threads - 1) / threads);
  const double t0 = MPI_Wtime();
  apply_boundary_kernel<<<blocks, threads>>>(d_next_, d_solid_, nx_, local_rows(), pitch_, local_ny_, u_in_);
  CUDA_KERNEL_CHECK();
  CUDA_CHECK(cudaDeviceSynchronize());
  timers.compute += MPI_Wtime() - t0;
}

void LBM::step(HybridTimers & timers)
{
  if (!device_allocated_) copy_host_to_device();

  if (halo_mode_ == "cuda") exchange_halos_cuda_aware(timers);
  else exchange_halos_host_staged(timers);

  compute_rows(1, local_ny_, timers);
  apply_boundaries(timers);

  std::swap(d_f_, d_next_);
  host_current_ = false;
}

void LBM::synchronize() const
{
  CUDA_CHECK(cudaSetDevice(device_));
  CUDA_CHECK(cudaDeviceSynchronize());
}

void LBM::sync_host() const
{
  if (host_current_) return;
  CUDA_CHECK(cudaSetDevice(device_));
  CUDA_CHECK(cudaMemcpy(f_.data(), d_f_, 9 * plane_ * sizeof(double), cudaMemcpyDeviceToHost));
  host_current_ = true;
}

void LBM::compute_local_fields(std::vector<double> & rho_local,
                               std::vector<double> & ux_local,
                               std::vector<double> & uy_local) const
{
  sync_host();
  const std::size_t n = nx_ * local_ny_;
  rho_local.assign(n, 0.0);
  ux_local.assign(n, 0.0);
  uy_local.assign(n, 0.0);

  for (std::size_t ly = 1; ly <= local_ny_; ++ly) {
    const std::size_t out_y = ly - 1;
    for (std::size_t x = 0; x < nx_; ++x) {
      const std::size_t out = out_y * nx_ + x;
      if (solid_[idx_local(x, ly)]) {
        rho_local[out] = 1.0;
        continue;
      }
      double r = 0.0;
      double mx = 0.0;
      double my = 0.0;
      for (int q = 0; q < Q; ++q) {
        const double fq = f_[fidx(q, x, ly)];
        r += fq;
        mx += cx[q] * fq;
        my += cy[q] * fq;
      }
      rho_local[out] = r;
      if (r > 0.0) {
        ux_local[out] = mx / r;
        uy_local[out] = my / r;
      }
    }
  }
}

void LBM::gather_layout(std::vector<int> & counts, std::vector<int> & displs) const
{
  counts.resize(size_);
  displs.resize(size_);
  const std::size_t p = std::size_t(size_);
  const std::size_t base = ny_global_ / p;
  const std::size_t rem = ny_global_ % p;
  for (int r = 0; r < size_; ++r) {
    const std::size_t lr = base + (std::size_t(r) < rem ? 1u : 0u);
    const std::size_t y0 = std::size_t(r) * base + std::min(std::size_t(r), rem);
    const std::size_t c = lr * nx_;
    const std::size_t d = y0 * nx_;
    if (c > std::size_t(std::numeric_limits<int>::max()) ||
        d > std::size_t(std::numeric_limits<int>::max())) {
      if (rank_ == 0) std::cerr << "Global grid too large for MPI_Gatherv int counts.\n";
      MPI_Abort(comm_, 1);
    }
    counts[r] = int(c);
    displs[r] = int(d);
  }
}

void LBM::gather_mask(std::vector<std::uint8_t> & global_mask, int root) const
{
  std::vector<std::uint8_t> local_mask(nx_ * local_ny_);
  for (std::size_t ly = 1; ly <= local_ny_; ++ly) {
    std::copy(solid_.begin() + std::ptrdiff_t(ly * nx_),
              solid_.begin() + std::ptrdiff_t((ly + 1) * nx_),
              local_mask.begin() + std::ptrdiff_t((ly - 1) * nx_));
  }

  std::vector<int> counts, displs;
  gather_layout(counts, displs);
  if (rank_ == root) global_mask.assign(nx_ * ny_global_, 0);
  else global_mask.clear();

  MPI_Gatherv(local_mask.data(), counts[rank_], MPI_UNSIGNED_CHAR,
              rank_ == root ? global_mask.data() : nullptr,
              counts.data(), displs.data(), MPI_UNSIGNED_CHAR,
              root, comm_);
}

void LBM::gather_fields(std::vector<double> & rho_v,
                        std::vector<double> & ux_v,
                        std::vector<double> & uy_v,
                        std::vector<double> & vort_v,
                        int root) const
{
  std::vector<double> rho_local, ux_local, uy_local;
  compute_local_fields(rho_local, ux_local, uy_local);

  std::vector<int> counts, displs;
  gather_layout(counts, displs);
  const std::size_t global_n = nx_ * ny_global_;
  if (rank_ == root) {
    rho_v.assign(global_n, 0.0);
    ux_v.assign(global_n, 0.0);
    uy_v.assign(global_n, 0.0);
    vort_v.assign(global_n, 0.0);
  } else {
    rho_v.clear(); ux_v.clear(); uy_v.clear(); vort_v.clear();
  }

  MPI_Gatherv(rho_local.data(), counts[rank_], MPI_DOUBLE,
              rank_ == root ? rho_v.data() : nullptr, counts.data(), displs.data(), MPI_DOUBLE,
              root, comm_);
  MPI_Gatherv(ux_local.data(), counts[rank_], MPI_DOUBLE,
              rank_ == root ? ux_v.data() : nullptr, counts.data(), displs.data(), MPI_DOUBLE,
              root, comm_);
  MPI_Gatherv(uy_local.data(), counts[rank_], MPI_DOUBLE,
              rank_ == root ? uy_v.data() : nullptr, counts.data(), displs.data(), MPI_DOUBLE,
              root, comm_);

  if (rank_ == root && nx_ >= 3 && ny_global_ >= 3) {
    for (std::size_t y = 1; y + 1 < ny_global_; ++y) {
      for (std::size_t x = 1; x + 1 < nx_; ++x) {
        const std::size_t k = y * nx_ + x;
        vort_v[k] = 0.5 * ((uy_v[y * nx_ + (x + 1)] - uy_v[y * nx_ + (x - 1)]) -
                           (ux_v[(y + 1) * nx_ + x] - ux_v[(y - 1) * nx_ + x]));
      }
    }
  }
}

void LBM::probe_velocity(std::size_t x, std::size_t y,
                         double & ux_out, double & uy_out,
                         int root) const
{
  double local[2] = {0.0, 0.0};
  if (x < nx_ && y < ny_global_ && owns_global_y(y)) {
    const std::size_t ly = local_y_from_global(y);
    if (solid_[idx_local(x, ly)] && !solid_probe_warning_printed_) {
      std::cerr << "Warning: probe location (" << x << ", " << y
                << ") is inside a solid cell on rank " << rank_
                << "; Strouhal estimation from this probe may be meaningless.\n";
      solid_probe_warning_printed_ = true;
    }
    CUDA_CHECK(cudaSetDevice(device_));
    probe_values_kernel<<<1, Q>>>(d_f_, d_probe_, x, ly, pitch_, plane_);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_probe_, d_probe_, Q * sizeof(double), cudaMemcpyDeviceToHost));

    double r = 0.0, mx = 0.0, my = 0.0;
    for (int q = 0; q < Q; ++q) {
      r += h_probe_[q];
      mx += cx[q] * h_probe_[q];
      my += cy[q] * h_probe_[q];
    }
    if (r > 0.0) {
      local[0] = mx / r;
      local[1] = my / r;
    }
  }

  double global[2] = {0.0, 0.0};
  MPI_Reduce(local, global, 2, MPI_DOUBLE, MPI_SUM, root, comm_);
  if (rank_ == root) {
    ux_out = global[0];
    uy_out = global[1];
  } else {
    ux_out = uy_out = 0.0;
  }
}

double LBM::rho(std::size_t x, std::size_t y) const
{
  if (!owns_global_y(y)) return 0.0;
  sync_host();
  const std::size_t ly = local_y_from_global(y);
  double r = 0.0;
  for (int q = 0; q < Q; ++q) r += f_[fidx(q, x, ly)];
  return r;
}

double LBM::ux(std::size_t x, std::size_t y) const
{
  if (!owns_global_y(y)) return 0.0;
  if (is_solid(x, y)) return 0.0;
  sync_host();
  const std::size_t ly = local_y_from_global(y);
  double r = 0.0, mx = 0.0;
  for (int q = 0; q < Q; ++q) { const double fq = f_[fidx(q, x, ly)]; r += fq; mx += cx[q] * fq; }
  return r > 0.0 ? mx / r : 0.0;
}

double LBM::uy(std::size_t x, std::size_t y) const
{
  if (!owns_global_y(y)) return 0.0;
  if (is_solid(x, y)) return 0.0;
  sync_host();
  const std::size_t ly = local_y_from_global(y);
  double r = 0.0, my = 0.0;
  for (int q = 0; q < Q; ++q) { const double fq = f_[fidx(q, x, ly)]; r += fq; my += cy[q] * fq; }
  return r > 0.0 ? my / r : 0.0;
}

double LBM::vorticity(std::size_t x, std::size_t y) const
{
  if (x == 0 || x + 1 >= nx_ || y == 0 || y + 1 >= ny_global_) return 0.0;
  return 0.5 * ((uy(x + 1, y) - uy(x - 1, y)) - (ux(x, y + 1) - ux(x, y - 1)));
}

bool LBM::is_solid(std::size_t x, std::size_t y) const
{
  if (!owns_global_y(y)) return false;
  return solid_[idx_local(x, local_y_from_global(y))] != 0;
}

void report_hybrid_timers(const HybridTimers & timers,
                          std::size_t nx,
                          std::size_t ny,
                          std::size_t steps,
                          MPI_Comm comm)
{
  int rank = 0, size = 1;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  double max_total = 0.0, max_compute = 0.0, max_comm = 0.0, max_other = 0.0;
  double avg_compute = 0.0, avg_comm = 0.0, avg_other = 0.0;
  const double local_other = std::max(0.0, timers.total - timers.compute - timers.comm);
  const double local_halo_percent =
      (timers.total > 0.0) ? 100.0 * timers.comm / timers.total : 0.0;
  double max_halo_percent = 0.0, avg_halo_percent = 0.0;

  MPI_Reduce(&timers.total,      &max_total,      1, MPI_DOUBLE, MPI_MAX, 0, comm);
  MPI_Reduce(&timers.compute,    &max_compute,    1, MPI_DOUBLE, MPI_MAX, 0, comm);
  MPI_Reduce(&timers.comm,       &max_comm,       1, MPI_DOUBLE, MPI_MAX, 0, comm);
  MPI_Reduce(&local_other,       &max_other,      1, MPI_DOUBLE, MPI_MAX, 0, comm);
  MPI_Reduce(&timers.compute,    &avg_compute,    1, MPI_DOUBLE, MPI_SUM, 0, comm);
  MPI_Reduce(&timers.comm,       &avg_comm,       1, MPI_DOUBLE, MPI_SUM, 0, comm);
  MPI_Reduce(&local_other,       &avg_other,      1, MPI_DOUBLE, MPI_SUM, 0, comm);
  MPI_Reduce(&local_halo_percent, &max_halo_percent, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
  MPI_Reduce(&local_halo_percent, &avg_halo_percent, 1, MPI_DOUBLE, MPI_SUM, 0, comm);

  if (rank == 0) {
    avg_compute /= double(size);
    avg_comm /= double(size);
    avg_other /= double(size);
    avg_halo_percent /= double(size);
    const double updates = double(nx) * double(ny) * double(steps);
    const double mlups = (max_total > 0.0) ? updates / max_total / 1.0e6 : 0.0;

    std::cout << "Wall time: " << max_total << " s\n";
    std::cout << "MLUPS: " << mlups << "\n";
    std::cout << "Max compute time: " << max_compute << " s\n";
    std::cout << "Max halo time: " << max_comm << " s\n";
    std::cout << "Avg compute time: " << avg_compute << " s\n";
    std::cout << "Avg halo time: " << avg_comm << " s\n";
    std::cout << "Max other time: " << max_other << " s\n";
    std::cout << "Avg other time: " << avg_other << " s\n";
    std::cout << "Max halo overhead percent: " << max_halo_percent << " %\n";
    std::cout << "Avg halo overhead percent: " << avg_halo_percent << " %\n";
  }
}
