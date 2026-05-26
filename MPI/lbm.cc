#include "lbm.hh"

#include <algorithm>
#include <cmath>
#include <iostream>

// D2Q9 lattice constants. Same indexing convention as the serial code:
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
constexpr int TAG_UP   = 1001;  // message sent to the upper rank
constexpr int TAG_DOWN = 1002;  // message sent to the lower rank

// Only three directions are needed per neighbour in a row-decomposed domain.
//
// For pull streaming:
// - The top owned row needs data from the top ghost row only for cy = +1.
// - The bottom owned row needs data from the bottom ghost row only for cy = -1.
//
// D2Q9 indexing:
//   cy = +1 : 2, 5, 6
//   cy = -1 : 4, 7, 8
constexpr int HALO_Q = 3;

constexpr int CY_POS[HALO_Q] = {2, 5, 6};
constexpr int CY_NEG[HALO_Q] = {4, 7, 8};
}  // namespace

LBM::LBM(std::size_t nx, std::size_t ny,
                 double u_in, double Re,
                 double cyl_x, double cyl_y, double cyl_r,
                 MPI_Comm comm)
  : nx_(nx),
    ny_global_(ny),
    local_ny_(0),
    y0_(0),
    u_in_(u_in),
    tau_(0.0),
    comm_(comm),
    rank_(0),
    size_(1),
    up_rank_(MPI_PROC_NULL),
    down_rank_(MPI_PROC_NULL)
{
  MPI_Comm_rank(comm_, &rank_);
  MPI_Comm_size(comm_, &size_);

  if (size_ > int(ny_global_)) {
    if (rank_ == 0) {
    std::cerr << "Error: number of MPI ranks must be <= ny\n";
    }
    MPI_Abort(comm_, 1);
  }


  // Balanced contiguous row-block decomposition.
  const std::size_t p = std::size_t(size_);
  const std::size_t r = std::size_t(rank_);
  const std::size_t base = ny_global_ / p;
  const std::size_t rem  = ny_global_ % p;

  local_ny_ = base + (r < rem ? 1u : 0u);
  y0_       = r * base + std::min(r, rem);


  // MPI_PROC_NULL is a special MPI value that represents no neighbor.
  // Sending to or receiving from it automatically becomes a no-op.
  up_rank_   = (rank_ == 0)         ? MPI_PROC_NULL : rank_ - 1;
  down_rank_ = (rank_ == size_ - 1) ? MPI_PROC_NULL : rank_ + 1;

  f_.assign(9 * nx_ * local_rows(), 0.0);
  ftmp_.assign(9 * nx_ * local_rows(), 0.0);
  solid_.assign(nx_ * local_rows(), 0);

  // Halo optimization:
  // We do not send the full 9-population row. For a horizontal MPI boundary,
  // only the three populations crossing that boundary are needed.
  // Total per step: 3 * nx to the upper rank + 3 * nx to the lower rank.
  send_top_.assign(HALO_Q * nx_, 0.0);
  send_bottom_.assign(HALO_Q * nx_, 0.0);
  recv_top_.assign(HALO_Q * nx_, 0.0);
  recv_bottom_.assign(HALO_Q * nx_, 0.0);

  // ν = c_s^2 (τ - 1/2) with c_s^2 = 1/3, and Re = u_in * D / ν.
  const double nu = u_in_ * (2.0 * cyl_r) / Re;
  tau_ = 3.0 * nu + 0.5;

  // No-slip top and bottom walls. Only the ranks that own the global wall
  // rows mark them as solid.
  for (std::size_t ly = 1; ly <= local_ny_; ++ly) {
    const long gy = global_y_from_local(ly);
    if (gy == 0 || gy == long(ny_global_) - 1) {
      for (std::size_t x = 0; x < nx_; ++x) {
        solid_[idx_local(x, ly)] = 1;
      }
    }
  }

  mark_obstacle(cyl_x, cyl_y, cyl_r);
}

void
LBM::add_second_cylinder(double cyl2_x, double cyl2_y, double cyl2_r)
{
  if (cyl2_r > 0.0) mark_obstacle(cyl2_x, cyl2_y, cyl2_r);
}

void
LBM::mark_obstacle(double c_x, double c_y, double r)
{
  const double r2 = r * r;
  for (std::size_t ly = 1; ly <= local_ny_; ++ly) {
    const double gy = double(global_y_from_local(ly));
    for (std::size_t x = 0; x < nx_; ++x) {
      const double dx = double(x) - c_x;
      const double dy = gy - c_y;
      if (dx * dx + dy * dy <= r2) {
        solid_[idx_local(x, ly)] = 1;
      }
    }
  }
}

void
LBM::initialize()
{
  // Initialize owned rows. Ghost rows are overwritten by halo exchange.
  for (std::size_t ly = 1; ly <= local_ny_; ++ly) {
    for (std::size_t x = 0; x < nx_; ++x) {
      const double rho = 1.0;
      const double ux  = solid_[idx_local(x, ly)] ? 0.0 : u_in_;
      const double uy  = 0.0;
      const double u2  = ux * ux + uy * uy;

      for (int i = 0; i < Q; ++i) {
        const double cu  = cx[i] * ux + cy[i] * uy;
        const double feq = w[i] * rho *
          (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2);
        f_[fidx(i, x, ly)] = feq;
      }
    }
  }
}

void
LBM::step()
{
  MPITimers unused;
  step_nonblocking(unused);
}

void
LBM::step_nonblocking(MPITimers & timers)
{
  double t0 = MPI_Wtime();

  collide();
  bounce_back();

  double t1 = MPI_Wtime();
  timers.compute += t1 - t0;

  MPI_Request reqs[4];

  t0 = MPI_Wtime();
  post_halo_exchange(reqs);
  t1 = MPI_Wtime();
  timers.comm += t1 - t0;

  // Useful work that does not need incoming ghost rows.
  t0 = MPI_Wtime();
  stream_interior_rows();
  t1 = MPI_Wtime();
  timers.compute += t1 - t0;

  // Exposed communication wait.
  t0 = MPI_Wtime();
  finish_halo_exchange(reqs);
  t1 = MPI_Wtime();
  timers.comm += t1 - t0;

  // Boundary rows need the received ghost rows. Streaming writes the next
  // state into ftmp_.
  t0 = MPI_Wtime();
  stream_boundary_rows();
  t1 = MPI_Wtime();
  timers.compute += t1 - t0;

  // Match the serial order: stream() swaps first, then inlet/outlet are applied
  // to the new current state. Applying BCs before the swap would write them
  // into the old f_ buffer and then discard them.
  f_.swap(ftmp_);   // swap is necessarily because streaming has read/write hazard

  t0 = MPI_Wtime();
  apply_inlet();
  apply_outlet();
  t1 = MPI_Wtime();
  timers.compute += t1 - t0;
}

void
LBM::collide()
{
  const double inv_tau = 1.0 / tau_;
  const std::size_t LN = local_N();


  // ly = 0              top ghost row
  // ly = 1~local_ny_    owned rows
  // ly = local_ny_ + 1  bottom ghost row
  for (std::size_t ly = 1; ly <= local_ny_; ++ly) 
  {
    for (std::size_t x = 0; x < nx_; ++x) 
    {
      const std::size_t k = idx_local(x, ly);

      double rho = 0.0, mx = 0.0, my = 0.0;
      for (int i = 0; i < Q; ++i) 
      {
        const double fi = f_[std::size_t(i) * LN + k];
        rho += fi;
        mx  += cx[i] * fi;
        my  += cy[i] * fi;
      }

      const double ux = (rho > 0.0) ? mx / rho : 0.0;
      const double uy = (rho > 0.0) ? my / rho : 0.0;
      const double u2 = ux * ux + uy * uy;

      for (int i = 0; i < Q; ++i) 
      {
        const double cu  = cx[i] * ux + cy[i] * uy;
        const double feq = w[i] * rho *
          (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2);
        f_[std::size_t(i) * LN + k] +=
          -inv_tau * (f_[std::size_t(i) * LN + k] - feq);
      }
    }
  }
}

void
LBM::bounce_back()
{
  // Same fullway bounce-back as the serial code, but only on owned rows.
  const std::size_t LN = local_N();

  for (std::size_t ly = 1; ly <= local_ny_; ++ly) {
    for (std::size_t x = 0; x < nx_; ++x) {
      const std::size_t k = idx_local(x, ly);
      if (!solid_[k]) continue;

      std::swap(f_[1 * LN + k], f_[3 * LN + k]);
      std::swap(f_[2 * LN + k], f_[4 * LN + k]);
      std::swap(f_[5 * LN + k], f_[7 * LN + k]);
      std::swap(f_[6 * LN + k], f_[8 * LN + k]);
    }
  }
}



// f_ is SoA layout： [f0 all cells][f1 all cells]...[f8 all cells]

void
LBM::pack_row_dirs(std::size_t ly,
                   const int dirs[3],
                   std::vector<double> & buffer) const
{
  const std::size_t LN = local_N();

  for (int d = 0; d < HALO_Q; ++d) {
    const int i = dirs[d];
    for (std::size_t x = 0; x < nx_; ++x) {
      buffer[std::size_t(d) * nx_ + x] =
        f_[std::size_t(i) * LN + idx_local(x, ly)];
    }
  }
}

void
LBM::unpack_row_dirs(std::size_t ly,
                     const int dirs[3],
                     const std::vector<double> & buffer)
{
  const std::size_t LN = local_N();

  for (int d = 0; d < HALO_Q; ++d) {
    const int i = dirs[d];
    for (std::size_t x = 0; x < nx_; ++x) {
      f_[std::size_t(i) * LN + idx_local(x, ly)] =
        buffer[std::size_t(d) * nx_ + x];
    }
  }
}


void
LBM::post_halo_exchange(MPI_Request requests[4])
{
  // Optimized halo exchange.
  //
  // The previous simple implementation sent all 9 D2Q9 distributions for each
  // boundary row. That is correct but wasteful.
  //
  // With pull streaming and 1D row decomposition:
  //
  // - Data sent to the upper rank will be used as its bottom ghost row.
  //   Its bottom boundary needs only cy = -1 directions: {4, 7, 8}.
  //
  // - Data sent to the lower rank will be used as its top ghost row.
  //   Its top boundary needs only cy = +1 directions: {2, 5, 6}.
  //
  // Therefore each MPI message contains only 3 * nx doubles instead of
  // 9 * nx doubles.
  if (local_ny_ > 0) {
    pack_row_dirs(1,         CY_NEG, send_top_);
    pack_row_dirs(local_ny_, CY_POS, send_bottom_);
  }

  const int count = int(HALO_Q * nx_);

  MPI_Irecv(recv_top_.data(), count, MPI_DOUBLE,
            up_rank_, TAG_DOWN, comm_, &requests[0]);

  MPI_Irecv(recv_bottom_.data(), count, MPI_DOUBLE,
            down_rank_, TAG_UP, comm_, &requests[1]);

  MPI_Isend(send_top_.data(), count, MPI_DOUBLE,
            up_rank_, TAG_UP, comm_, &requests[2]);

  MPI_Isend(send_bottom_.data(), count, MPI_DOUBLE,
            down_rank_, TAG_DOWN, comm_, &requests[3]);
}


void
LBM::finish_halo_exchange(MPI_Request requests[4])
{
  MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);


  if (up_rank_ != MPI_PROC_NULL) {
    unpack_row_dirs(0, CY_POS, recv_top_);
  }


  if (down_rank_ != MPI_PROC_NULL) {
    unpack_row_dirs(local_ny_ + 1, CY_NEG, recv_bottom_);
  }
}

void
LBM::stream_rows(std::size_t ly_begin, std::size_t ly_end)
{
  if (local_ny_ == 0 || ly_begin > ly_end) return;

  const std::size_t LN = local_N();

  for (int i = 0; i < Q; ++i) {
    for (std::size_t ly = ly_begin; ly <= ly_end; ++ly) {
      for (std::size_t x = 0; x < nx_; ++x) {
        const long sx  = long(x) - cx[i];
        const long sly = long(ly) - cy[i];
        const long sgy = long(y0_) + sly - 1;

        const bool source_inside_x =
          (sx >= 0 && sx < long(nx_));

        const bool source_inside_global_y =
          (sgy >= 0 && sgy < long(ny_global_));

        const bool source_inside_local_storage =
          (sly >= 0 && sly < long(local_rows()));

        if (source_inside_x &&
            source_inside_global_y &&
            source_inside_local_storage) {
          ftmp_[std::size_t(i) * LN + idx_local(x, ly)] =
            f_[std::size_t(i) * LN + idx_local(std::size_t(sx), std::size_t(sly))];
        } else {
          // Same fallback as the serial code: keep current value at physical
          // boundaries. Inlet/outlet routines overwrite the relevant columns.
          ftmp_[std::size_t(i) * LN + idx_local(x, ly)] =
            f_[std::size_t(i) * LN + idx_local(x, ly)];
        }
      }
    }
  }
}

void
LBM::stream_interior_rows()
{
  // First and last owned rows may need ghost rows. Rows 2..local_ny-1 do not.
  if (local_ny_ > 2) {
    stream_rows(2, local_ny_ - 1);
  }
}

void
LBM::stream_boundary_rows()
{
  if (local_ny_ == 0) return;

  stream_rows(1, 1);

  if (local_ny_ > 1) {
    stream_rows(local_ny_, local_ny_);
  }
}

void
LBM::apply_inlet()
{
  // Reset the inlet column to equilibrium. Applied only to owned rows.
  const std::size_t LN = local_N();
  const std::size_t x = 0;

  for (std::size_t ly = 1; ly <= local_ny_; ++ly) {
    if (solid_[idx_local(x, ly)]) continue;

    const double rho = 1.0;
    const double ux  = u_in_;
    const double uy  = 0.0;
    const double u2  = ux * ux + uy * uy;

    for (int i = 0; i < Q; ++i) {
      const double cu = cx[i] * ux + cy[i] * uy;
      f_[std::size_t(i) * LN + idx_local(x, ly)] =
        w[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2);
    }
  }
}

void
LBM::apply_outlet()
{
  // Zero-gradient outlet: copy the second-to-last column into the last.
  if (nx_ < 2) return;

  const std::size_t LN = local_N();
  const std::size_t x  = nx_ - 1;
  const std::size_t xs = nx_ - 2;

  for (std::size_t ly = 1; ly <= local_ny_; ++ly) {
    if (solid_[idx_local(x, ly)]) continue;

    for (int i = 0; i < Q; ++i) {
      f_[std::size_t(i) * LN + idx_local(x, ly)] =
        f_[std::size_t(i) * LN + idx_local(xs, ly)];
    }
  }
}

double
LBM::rho(std::size_t x, std::size_t y) const
{
  if (x >= nx_ || !owns_global_y(y)) return 0.0;

  const std::size_t ly = local_y_from_global(y);
  const std::size_t LN = local_N();

  double r = 0.0;
  for (int i = 0; i < Q; ++i) {
    r += f_[std::size_t(i) * LN + idx_local(x, ly)];
  }
  return r;
}

double
LBM::ux(std::size_t x, std::size_t y) const
{
  if (x >= nx_ || !owns_global_y(y)) return 0.0;

  const std::size_t ly = local_y_from_global(y);
  const std::size_t LN = local_N();

  double r = 0.0, m = 0.0;
  for (int i = 0; i < Q; ++i) {
    const double fi = f_[std::size_t(i) * LN + idx_local(x, ly)];
    r += fi;
    m += cx[i] * fi;
  }
  return (r > 0.0) ? m / r : 0.0;
}

double
LBM::uy(std::size_t x, std::size_t y) const
{
  if (x >= nx_ || !owns_global_y(y)) return 0.0;

  const std::size_t ly = local_y_from_global(y);
  const std::size_t LN = local_N();

  double r = 0.0, m = 0.0;
  for (int i = 0; i < Q; ++i) {
    const double fi = f_[std::size_t(i) * LN + idx_local(x, ly)];
    r += fi;
    m += cy[i] * fi;
  }
  return (r > 0.0) ? m / r : 0.0;
}

double
LBM::vorticity(std::size_t x, std::size_t y) const
{
  if (x == 0 || x >= nx_ - 1 || y == 0 || y >= ny_global_ - 1) return 0.0;
  if (!owns_global_y(y)) return 0.0;

  return 0.5 * ((uy(x + 1, y) - uy(x - 1, y)) -
                (ux(x, y + 1) - ux(x, y - 1)));
}

bool
LBM::is_solid(std::size_t x, std::size_t y) const
{
  if (x >= nx_ || !owns_global_y(y)) return false;
  const std::size_t ly = local_y_from_global(y);
  return solid_[idx_local(x, ly)] != 0;
}


void
LBM::gather_mask(std::vector<std::uint8_t> & global_mask, int root) const
{
  std::vector<std::uint8_t> local_mask(nx_ * local_ny_);
  for (std::size_t ly = 1; ly <= local_ny_; ++ly) {
    for (std::size_t x = 0; x < nx_; ++x) {
      local_mask[(ly - 1) * nx_ + x] = solid_[idx_local(x, ly)];
    }
  }

  std::vector<int> recvcounts;
  std::vector<int> displs;
  if (rank_ == root) {
    recvcounts.resize(size_);
    displs.resize(size_);
    int disp = 0;
    for (int r = 0; r < size_; ++r) {
      const std::size_t base = ny_global_ / std::size_t(size_);
      const std::size_t rem  = ny_global_ % std::size_t(size_);
      const std::size_t lny  = base + (std::size_t(r) < rem ? 1u : 0u);
      recvcounts[r] = int(lny * nx_);
      displs[r] = disp;
      disp += recvcounts[r];
    }
    global_mask.assign(nx_ * ny_global_, 0);
  } else {
    global_mask.clear();
  }

  MPI_Gatherv(local_mask.empty() ? nullptr : local_mask.data(),
              int(local_mask.size()), MPI_BYTE,
              rank_ == root ? global_mask.data() : nullptr,
              rank_ == root ? recvcounts.data() : nullptr,
              rank_ == root ? displs.data() : nullptr,
              MPI_BYTE, root, comm_);
}

void
LBM::gather_fields(std::vector<double> & rho_v,
                   std::vector<double> & ux_v,
                   std::vector<double> & uy_v,
                   std::vector<double> & vort_v,
                   int root) const
{
  const std::size_t local_count = nx_ * local_ny_;
  std::vector<double> local_rho(local_count);
  std::vector<double> local_ux(local_count);
  std::vector<double> local_uy(local_count);

  const std::size_t LN = local_N();
  for (std::size_t ly = 1; ly <= local_ny_; ++ly) {
    for (std::size_t x = 0; x < nx_; ++x) {
      const std::size_t out_k = (ly - 1) * nx_ + x;
      const std::size_t k = idx_local(x, ly);

      double r = 0.0;
      double mx = 0.0;
      double my = 0.0;
      for (int i = 0; i < Q; ++i) {
        const double fi = f_[std::size_t(i) * LN + k];
        r  += fi;
        mx += cx[i] * fi;
        my += cy[i] * fi;
      }

      local_rho[out_k] = r;
      local_ux[out_k]  = (r > 0.0) ? mx / r : 0.0;
      local_uy[out_k]  = (r > 0.0) ? my / r : 0.0;
    }
  }

  std::vector<int> recvcounts;
  std::vector<int> displs;
  if (rank_ == root) {
    recvcounts.resize(size_);
    displs.resize(size_);
    int disp = 0;
    for (int r = 0; r < size_; ++r) {
      const std::size_t base = ny_global_ / std::size_t(size_);
      const std::size_t rem  = ny_global_ % std::size_t(size_);
      const std::size_t lny  = base + (std::size_t(r) < rem ? 1u : 0u);
      recvcounts[r] = int(lny * nx_);
      displs[r] = disp;
      disp += recvcounts[r];
    }

    rho_v.assign(nx_ * ny_global_, 0.0);
    ux_v.assign(nx_ * ny_global_, 0.0);
    uy_v.assign(nx_ * ny_global_, 0.0);
    vort_v.assign(nx_ * ny_global_, 0.0);
  } else {
    rho_v.clear();
    ux_v.clear();
    uy_v.clear();
    vort_v.clear();
  }

  MPI_Gatherv(local_rho.empty() ? nullptr : local_rho.data(),
              int(local_rho.size()), MPI_DOUBLE,
              rank_ == root ? rho_v.data() : nullptr,
              rank_ == root ? recvcounts.data() : nullptr,
              rank_ == root ? displs.data() : nullptr,
              MPI_DOUBLE, root, comm_);

  MPI_Gatherv(local_ux.empty() ? nullptr : local_ux.data(),
              int(local_ux.size()), MPI_DOUBLE,
              rank_ == root ? ux_v.data() : nullptr,
              rank_ == root ? recvcounts.data() : nullptr,
              rank_ == root ? displs.data() : nullptr,
              MPI_DOUBLE, root, comm_);

  MPI_Gatherv(local_uy.empty() ? nullptr : local_uy.data(),
              int(local_uy.size()), MPI_DOUBLE,
              rank_ == root ? uy_v.data() : nullptr,
              rank_ == root ? recvcounts.data() : nullptr,
              rank_ == root ? displs.data() : nullptr,
              MPI_DOUBLE, root, comm_);

  if (rank_ == root) {
    for (std::size_t y = 1; y + 1 < ny_global_; ++y) {
      for (std::size_t x = 1; x + 1 < nx_; ++x) {
        const std::size_t k = y * nx_ + x;
        vort_v[k] = 0.5 * ((uy_v[y * nx_ + (x + 1)] - uy_v[y * nx_ + (x - 1)]) -
                           (ux_v[(y + 1) * nx_ + x] - ux_v[(y - 1) * nx_ + x]));
      }
    }
  }
}

void
LBM::probe_velocity(std::size_t x, std::size_t y,
                    double & ux_out, double & uy_out,
                    int root) const
{
  double local[2] = {0.0, 0.0};
  if (x < nx_ && owns_global_y(y)) {
    local[0] = ux(x, y);
    local[1] = uy(x, y);
  }

  double global[2] = {0.0, 0.0};
  MPI_Reduce(local, global, 2, MPI_DOUBLE, MPI_SUM, root, comm_);

  if (rank_ == root) {
    ux_out = global[0];
    uy_out = global[1];
  } else {
    ux_out = 0.0;
    uy_out = 0.0;
  }
}

void
report_mpi_timers(const MPITimers & timers,
                  std::size_t nx,
                  std::size_t ny,
                  std::size_t steps,
                  MPI_Comm comm)
{
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);



  const double local_comm_fraction =
    timers.total > 0.0 ? timers.comm / timers.total : 0.0;

  double max_total = 0.0;
  double max_compute = 0.0;
  double max_comm = 0.0;
  double sum_total = 0.0;
  double sum_compute = 0.0;
  double sum_comm = 0.0;
  double max_comm_fraction = 0.0;
  double sum_comm_fraction = 0.0;

  // These reductions happen after the timed loop, so their cost is not
  // included in timers.total.
  MPI_Reduce(&timers.total, &max_total, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
  MPI_Reduce(&timers.compute, &max_compute, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
  MPI_Reduce(&timers.comm, &max_comm, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

  MPI_Reduce(&timers.total, &sum_total, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
  MPI_Reduce(&timers.compute, &sum_compute, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
  MPI_Reduce(&timers.comm, &sum_comm, 1, MPI_DOUBLE, MPI_SUM, 0, comm);

  MPI_Reduce(&local_comm_fraction, &max_comm_fraction, 1, MPI_DOUBLE,
             MPI_MAX, 0, comm);
  MPI_Reduce(&local_comm_fraction, &sum_comm_fraction, 1, MPI_DOUBLE,
             MPI_SUM, 0, comm);

  if (rank == 0) {
    const double avg_total = sum_total / double(size);
    const double avg_compute = sum_compute / double(size);
    const double avg_comm = sum_comm / double(size);
    const double avg_comm_fraction = sum_comm_fraction / double(size);

    const double mlups =
      double(nx) * double(ny) * double(steps) / max_total / 1.0e6;

    std::cout << "Wall time max       : " << max_total << " s\n"
              << "Wall time avg       : " << avg_total << " s\n"
              << "Compute time max    : " << max_compute << " s\n"
              << "Compute time avg    : " << avg_compute << " s\n"
              << "Comm time max       : " << max_comm << " s\n"
              << "Comm time avg       : " << avg_comm << " s\n"
              << "Comm fraction max   : " << 100.0 * max_comm_fraction << " %\n"
              << "Comm fraction avg   : " << 100.0 * avg_comm_fraction << " %\n"
              << "Comm/max_total      : " << 100.0 * max_comm / max_total << " %\n"
              << "Comm/max_compute    : " << 100.0 * max_comm / max_compute << " %\n"
              << "MLUPS               : " << mlups << "\n";
  }
}
