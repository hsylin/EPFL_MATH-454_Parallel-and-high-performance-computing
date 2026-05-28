#include "lbm.hh"
#include "output.hh"

#include <mpi.h>
#if defined(__has_include)
#  if __has_include(<mpi-ext.h>)
#    include <mpi-ext.h>
#  endif
#endif

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace {
using Args = std::unordered_map<std::string, std::string>;

enum class CudaAwareStatus {
  Available,
  Unavailable,
  Unknown
};

Args parse_args(int argc, char ** argv)
{
  Args kv;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    const auto eq = a.find('=');
    if (eq == std::string::npos || eq == 0) {
      std::cerr << "Bad argument '" << a << "' (expected key=value)\n";
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    kv[a.substr(0, eq)] = a.substr(eq + 1);
  }
  return kv;
}

template <typename T>
T get(const Args & kv, const std::string & key, T def)
{
  auto it = kv.find(key);
  if (it == kv.end()) return def;

  if (std::numeric_limits<T>::is_integer && !std::numeric_limits<T>::is_signed &&
      it->second.find('-') != std::string::npos) {
    std::cerr << "Invalid negative value for unsigned argument '" << key << "': '" << it->second << "'\n";
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  std::istringstream iss(it->second);
  T v{};
  iss >> v;
  iss >> std::ws;
  if (!iss || !iss.eof()) {
    std::cerr << "Invalid value for argument '" << key << "': '" << it->second << "'\n";
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  return v;
}

std::string get_string(const Args & kv, const std::string & key, const std::string & def)
{
  auto it = kv.find(key);
  return (it == kv.end()) ? def : it->second;
}

CudaAwareStatus query_cuda_aware_mpi()
{
#if defined(MPIX_CUDA_AWARE_SUPPORT)
  const int supported = MPIX_Query_cuda_support();
  return supported ? CudaAwareStatus::Available : CudaAwareStatus::Unavailable;
#else
  return CudaAwareStatus::Unknown;
#endif
}

std::string cuda_aware_mpi_status_string(CudaAwareStatus status)
{
  switch (status) {
    case CudaAwareStatus::Available:
      return "available (MPIX_Query_cuda_support=1)";
    case CudaAwareStatus::Unavailable:
      return "not available / not reported by MPIX_Query_cuda_support";
    case CudaAwareStatus::Unknown:
    default:
      return "unknown (MPIX_CUDA_AWARE_SUPPORT macro unavailable)";
  }
}

void abort_if(bool cond, const std::string & msg, int rank)
{
  if (!cond) return;
  if (rank == 0) std::cerr << msg << '\n';
  MPI_Abort(MPI_COMM_WORLD, 1);
}
} // namespace

int main(int argc, char ** argv)
{
  MPI_Init(&argc, &argv);

  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const Args kv = parse_args(argc, argv);

  const std::size_t nx    = get<std::size_t>(kv, "nx",    800);
  const std::size_t ny    = get<std::size_t>(kv, "ny",    400);
  const double      Re    = get<double>     (kv, "re",    100.0);
  const double      u_in  = get<double>     (kv, "u_in",  0.05);
  const std::size_t steps = get<std::size_t>(kv, "steps", 60000);

  const double cx0 = get<double>(kv, "cyl_x", double(nx) * 0.25);
  const double cy0 = get<double>(kv, "cyl_y", double(ny) * 0.50);
  const double cr0 = get<double>(kv, "cyl_r", double(ny) * 0.025);

  const double cx1 = get<double>(kv, "cyl2_x", -1.0);
  const double cy1 = get<double>(kv, "cyl2_y", -1.0);
  const double cr1 = get<double>(kv, "cyl2_r", -1.0);

  const std::size_t every       = get<std::size_t>(kv, "every", 500);
  const std::size_t probe_every = get<std::size_t>(kv, "probe_every", 1);
  const std::string out_pref    = get_string(kv, "out", "out/hybrid_lbm");
  const std::string probe_csv   = get_string(kv, "probe", "probe.csv");

  const std::size_t px = get<std::size_t>(kv, "probe_x", std::size_t(cx0 + 8.0 * cr0));
  const std::size_t py = get<std::size_t>(kv, "probe_y", std::size_t(cy0));

  const unsigned int block_x = get<unsigned int>(kv, "block_x", 64);
  const unsigned int block_y = get<unsigned int>(kv, "block_y", 4);
  const std::size_t pitch_align = get<std::size_t>(kv, "pitch_align", 32);
  const std::string halo_mode = get_string(kv, "halo", "cuda");
  const bool strict_cuda_aware = get<int>(kv, "strict_cuda_aware", 0) != 0;

  abort_if(nx < 3 || ny < 3, "Invalid grid: nx and ny must both be >= 3.", rank);
  abort_if(block_x == 0 || block_y == 0 || block_x * block_y > 1024,
           "Invalid CUDA block shape: block_x*block_y must be in [1, 1024].", rank);
  abort_if(halo_mode != "cuda" && halo_mode != "staged",
           "Invalid halo mode. Use halo=cuda or halo=staged.", rank);
  abort_if(px >= nx || py >= ny, "Invalid probe location: probe_x/probe_y outside the global grid.", rank);

  const CudaAwareStatus cuda_status = query_cuda_aware_mpi();
  if (halo_mode == "cuda") {
    if (cuda_status == CudaAwareStatus::Unavailable) {
      if (rank == 0) {
        std::cerr << "halo=cuda requested, but this MPI library reports that CUDA-aware MPI is unavailable.\n"
                  << "Use halo=staged or load a CUDA-aware MPI module.\n";
      }
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (cuda_status == CudaAwareStatus::Unknown && strict_cuda_aware) {
      if (rank == 0) {
        std::cerr << "halo=cuda requested with strict_cuda_aware=1, but CUDA-aware MPI support is unknown.\n"
                  << "Either load an MPI module exposing MPIX_Query_cuda_support, use halo=staged,\n"
                  << "or rerun with strict_cuda_aware=0 after checking ompi_info manually.\n";
      }
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  }

  const bool write_snapshots = (every > 0);
  const bool write_probe = (!probe_csv.empty() && probe_csv != "/dev/null" && probe_every > 0);

  LBM solver(nx, ny, u_in, Re, cx0, cy0, cr0, MPI_COMM_WORLD, pitch_align, halo_mode);
  solver.set_cuda_block(block_x, block_y);
  if (cr1 > 0.0) solver.add_second_cylinder(cx1, cy1, cr1);
  solver.initialize();

  if (rank == 0) {
    std::cout << "LBM 2D D2Q9 BGK Hybrid MPI+CUDA\n"
              << "  ranks         : " << size << "\n"
              << "  node mode     : intra-node only\n"
              << "  halo mode     : " << halo_mode << "\n"
              << "  CUDA-aware MPI: " << cuda_aware_mpi_status_string(cuda_status) << "\n"
              << "  CUDA block    : " << block_x << " x " << block_y << "\n"
              << "  pitch_align   : " << pitch_align << "\n"
              << "  grid          : " << nx << " x " << ny << "\n"
              << "  Re            : " << Re << "\n"
              << "  u_in          : " << u_in << "\n"
              << "  tau           : " << solver.tau() << "\n"
              << "  nu            : " << (u_in * (2.0 * cr0) / Re) << "\n"
              << "  Mach          : " << (u_in * std::sqrt(3.0)) << "\n"
              << "  cylinder      : (" << cx0 << ", " << cy0 << "), r = " << cr0 << "\n"
              << "  steps         : " << steps << "\n"
              << "  output every  : " << every << " (0 = off)\n"
              << "  output prefix : " << out_pref << "\n"
              << "  probe at      : (" << px << ", " << py << ")\n"
              << "  probe every   : " << probe_every << "\n"
              << "  probe csv     : " << probe_csv << "\n";
    if (halo_mode == "cuda") {
      std::cout << "  CUDA-aware note: device pointers are passed directly to MPI; "
                << "this does not by itself imply NVLink, GPUDirect RDMA, or zero-copy transfer.\n";
      if (cuda_status == CudaAwareStatus::Unknown) {
        std::cout << "  CUDA-aware warning: MPI support could not be queried at compile time; "
                  << "check ompi_info/job logs before making performance claims.\n";
      }
    }
  }

  {
    const unsigned long long local_info[3] = {
      static_cast<unsigned long long>(solver.y_begin()),
      static_cast<unsigned long long>(solver.local_ny()),
      static_cast<unsigned long long>(solver.device())
    };
    std::vector<unsigned long long> all_info;
    if (rank == 0) all_info.resize(std::size_t(size) * 3);
    MPI_Gather(local_info, 3, MPI_UNSIGNED_LONG_LONG,
               rank == 0 ? all_info.data() : nullptr, 3, MPI_UNSIGNED_LONG_LONG,
               0, MPI_COMM_WORLD);
    if (rank == 0) {
      std::cout << "  rank layout    :";
      for (int r = 0; r < size; ++r) {
        const auto y0 = all_info[std::size_t(r) * 3 + 0];
        const auto ln = all_info[std::size_t(r) * 3 + 1];
        const auto dev = all_info[std::size_t(r) * 3 + 2];
        std::cout << " rank " << r << " -> rows [" << y0 << ", " << (y0 + ln - 1)
                  << "], GPU " << dev << ";";
      }
      std::cout << "\n";
    }
  }

  std::unique_ptr<XDMFWriter> writer;
  std::ofstream probe;

  if (rank == 0 && write_snapshots) {
    writer.reset(new XDMFWriter(out_pref, nx, ny));
  }

  if (rank == 0 && write_probe) {
    probe.open(probe_csv.c_str());
    if (!probe) {
      std::cerr << "Could not open probe CSV for writing: " << probe_csv << "\n";
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    probe << "step,ux,uy\n";
  }

  if (write_snapshots) {
    std::vector<std::uint8_t> global_mask;
    solver.gather_mask(global_mask, 0);
    if (rank == 0) writer->write_mask_data(global_mask);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  HybridTimers timers;
  std::size_t probe_writes = 0;
  const double t0 = MPI_Wtime();

  for (std::size_t step = 1; step <= steps; ++step) {
    solver.step(timers);

    if (write_probe && step % probe_every == 0) {
      double vx = 0.0, vy = 0.0;
      solver.probe_velocity(px, py, vx, vy, 0);
      if (rank == 0) {
        probe << step << ',' << vx << ',' << vy << '\n';
        if (++probe_writes % 1000 == 0) probe.flush();
      }
    }

    if (write_snapshots && step % every == 0) {
      std::vector<double> rho_v, ux_v, uy_v, vor_v;
      solver.gather_fields(rho_v, ux_v, uy_v, vor_v, 0);
      if (rank == 0) {
        writer->write_snapshot_data(rho_v, ux_v, uy_v, vor_v, double(step));
        std::cout << "\r  step " << step << " / " << steps << std::flush;
      }
    }
  }

  solver.synchronize();
  MPI_Barrier(MPI_COMM_WORLD);
  timers.total = MPI_Wtime() - t0;

  if (rank == 0 && write_snapshots) std::cout << "\n";
  report_hybrid_timers(timers, nx, ny, steps, MPI_COMM_WORLD);

  MPI_Finalize();
  return 0;
}
