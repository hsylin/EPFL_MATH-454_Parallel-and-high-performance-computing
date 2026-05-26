#include "lbm.hh"
#include "output.hh"

#include <mpi.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

using Args = std::unordered_map<std::string, std::string>;

Args
parse_args(int argc, char ** argv)
{
  Args kv;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    const auto eq = a.find('=');
    if (eq == std::string::npos) {
      std::cerr << "Bad argument '" << a << "' (expected key=value)\n";
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    kv[a.substr(0, eq)] = a.substr(eq + 1);
  }
  return kv;
}

template <typename T>
T
get(const Args & kv, const std::string & key, T def)
{
  auto it = kv.find(key);
  if (it == kv.end()) return def;
  std::istringstream iss(it->second);
  T v;
  iss >> v;
  return v;
}

std::string
get_string(const Args & kv, const std::string & key, const std::string & def)
{
  auto it = kv.find(key);
  return (it == kv.end()) ? def : it->second;
}

}  // namespace

int
main(int argc, char ** argv)
{
  MPI_Init(&argc, &argv);

  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const Args kv = parse_args(argc, argv);

  // Grid + physics.
  const std::size_t nx    = get<std::size_t>(kv, "nx",    800);
  const std::size_t ny    = get<std::size_t>(kv, "ny",    400);
  const double      Re    = get<double>     (kv, "re",    100.0);
  const double      u_in  = get<double>     (kv, "u_in",  0.05);
  const std::size_t steps = get<std::size_t>(kv, "steps", 60000);

  // Cylinder geometry. Defaults: at (nx/4, ny/2) with radius ny/40.
  const double cx0 = get<double>(kv, "cyl_x", double(nx) * 0.25);
  const double cy0 = get<double>(kv, "cyl_y", double(ny) * 0.50);
  const double cr0 = get<double>(kv, "cyl_r", double(ny) * 0.025);

  // Optional second cylinder. Disabled if cyl2_r <= 0.
  const double cx1 = get<double>(kv, "cyl2_x", -1.0);
  const double cy1 = get<double>(kv, "cyl2_y", -1.0);
  const double cr1 = get<double>(kv, "cyl2_r", -1.0);

  // Output. Same interface as the serial driver.
  const std::size_t every     = get<std::size_t>(kv, "every", 500);
  const std::string out_pref  = get_string(kv, "out", "out/lbm");
  const std::string probe_csv = get_string(kv, "probe", "probe.csv");

  const bool write_snapshots = (every > 0);
  const bool write_probe = (!probe_csv.empty() && probe_csv != "/dev/null");

  // Probe location: about 4 diameters downstream, on the cylinder centerline.
  const std::size_t px = get<std::size_t>(
    kv, "probe_x", std::size_t(cx0 + 8.0 * cr0));
  const std::size_t py = get<std::size_t>(
    kv, "probe_y", std::size_t(cy0));

  LBM solver(nx, ny, u_in, Re, cx0, cy0, cr0, MPI_COMM_WORLD);
  if (cr1 > 0.0) solver.add_second_cylinder(cx1, cy1, cr1);
  solver.initialize();

  if (rank == 0) {
    std::cout << "LBM 2D D2Q9 BGK MPI\n"
              << "  ranks         : " << size << "\n"
              << "  decomposition : 1D row-block\n"
              << "  communication : non-blocking halo exchange\n"
              << "  grid          : " << nx << " x " << ny << "\n"
              << "  Re            : " << Re << "\n"
              << "  u_in          : " << u_in << "\n"
              << "  tau           : " << solver.tau() << "\n"
              << "  cylinder      : (" << cx0 << ", " << cy0
              << "), r = " << cr0 << "\n";
    if (cr1 > 0.0) {
      std::cout << "  cylinder #2   : (" << cx1 << ", " << cy1
                << "), r = " << cr1 << "\n";
    }
    std::cout << "  steps         : " << steps << "\n"
              << "  output every  : " << every << " (0 = off)\n"
              << "  output prefix : " << out_pref << "\n"
              << "  probe at      : (" << px << ", " << py << ")\n"
              << "  probe csv     : " << probe_csv << "\n";
  }

  std::unique_ptr<XDMFWriter> writer;
  std::ofstream probe;

  if (rank == 0 && write_snapshots) {
    writer.reset(new XDMFWriter(out_pref, nx, ny));
  }

  if (rank == 0 && write_probe) {
    probe.open(probe_csv.c_str());
    probe << "step,ux,uy\n";
  }

  // Write global mask once before the timed loop, matching the serial driver.
  if (write_snapshots) {
    std::vector<std::uint8_t> global_mask;
    solver.gather_mask(global_mask, 0);
    if (rank == 0) writer->write_mask_data(global_mask);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  MPITimers timers;
  const double t0 = MPI_Wtime();

  for (std::size_t step = 1; step <= steps; ++step) {
    solver.step_nonblocking(timers);

    if (write_probe) {
      double probe_ux = 0.0;
      double probe_uy = 0.0;
      solver.probe_velocity(px, py, probe_ux, probe_uy, 0);
      if (rank == 0) {
        probe << step << ',' << probe_ux << ',' << probe_uy << '\n';
      }
    }

    if (write_snapshots && step % every == 0) {
      std::vector<double> rho_v;
      std::vector<double> ux_v;
      std::vector<double> uy_v;
      std::vector<double> vor_v;
      solver.gather_fields(rho_v, ux_v, uy_v, vor_v, 0);

      if (rank == 0) {
        writer->write_snapshot_data(rho_v, ux_v, uy_v, vor_v, double(step));
        std::cout << "\r  step " << step << " / " << steps << std::flush;
      }
    }
  }

  if (rank == 0 && write_snapshots) {
    std::cout << "\n";
  }

  // This total includes probe reductions and snapshot gather/write if enabled,
  // just as the serial driver includes probe/snapshot output inside its loop.
  // For pure solver performance, run with every=0 probe=/dev/null.
  timers.total = MPI_Wtime() - t0;

  report_mpi_timers(timers, nx, ny, steps, MPI_COMM_WORLD);

  MPI_Finalize();
  return 0;
}
