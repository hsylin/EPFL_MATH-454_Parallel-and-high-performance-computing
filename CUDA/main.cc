#include "lbm.hh"
#include "output.hh"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

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
      std::exit(1);
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
  T v = def;
  if (!(iss >> v) || !(iss >> std::ws).eof()) {
    throw std::runtime_error("Invalid value for " + key + ": " + it->second);
  }
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
  try {
  const Args kv = parse_args(argc, argv);

  const std::size_t nx    = get<std::size_t>(kv, "nx",    800);
  const std::size_t ny    = get<std::size_t>(kv, "ny",    400);
  const double      Re    = get<double>     (kv, "re",    100.0);
  const double      u_in  = get<double>     (kv, "u_in",  0.05);
  const std::size_t steps = get<std::size_t>(kv, "steps", 60000);

  // CUDA launch configuration for the main 2D fused kernel.
  const unsigned int block_x = get<unsigned int>(kv, "block_x", 32);
  const unsigned int block_y = get<unsigned int>(kv, "block_y", 8);

  // Padded row pitch. pitch = round_up(nx, pitch_align). Use pitch_align=1 to disable padding.
  const std::size_t pitch_align = get<std::size_t>(kv, "pitch_align", 32);

  const double cx0 = get<double>(kv, "cyl_x", double(nx) * 0.25);
  const double cy0 = get<double>(kv, "cyl_y", double(ny) * 0.50);
  const double cr0 = get<double>(kv, "cyl_r", double(ny) * 0.025);

  const double cx1 = get<double>(kv, "cyl2_x", -1.0);
  const double cy1 = get<double>(kv, "cyl2_y", -1.0);
  const double cr1 = get<double>(kv, "cyl2_r", -1.0);

  const std::size_t every       = get<std::size_t>(kv, "every", 500);
  const std::string out_pref    = get_string(kv, "out", "out/lbm");
  const std::string probe_csv   = get_string(kv, "probe", "probe.csv");
  const std::size_t probe_every = get<std::size_t>(kv, "probe_every", 1); // 0 disables sampling

  const std::size_t px = get<std::size_t>(kv, "probe_x", std::size_t(cx0 + 8.0 * cr0));
  const std::size_t py = get<std::size_t>(kv, "probe_y", std::size_t(cy0));

  LBM solver(nx, ny, u_in, Re, cx0, cy0, cr0, pitch_align);
  solver.set_cuda_block(block_x, block_y);

  // Add all optional obstacles before initialize(), so every solid cell starts
  // from the intended zero-velocity equilibrium instead of being converted from
  // an already-initialized fluid cell.
  if (cr1 > 0.0) solver.add_second_cylinder(cx1, cy1, cr1);
  solver.initialize();
  solver.configure_probe(px, py, steps, probe_every);

  std::cout << "LBM 2D D2Q9 BGK (CUDA optimized: fused pull-collide)\n"
            << "  grid          : " << nx << " x " << ny << "\n"
            << "  CUDA block    : " << solver.block_x() << " x " << solver.block_y()
            << " (" << solver.block_x() * solver.block_y() << " threads)\n"
            << "  pitch         : " << solver.pitch() << " (align " << solver.pitch_align() << ")\n"
            << "  Re            : " << Re << "\n"
            << "  u_in          : " << u_in << "\n"
            << "  tau           : " << solver.tau() << "\n"
            << "  cylinder      : (" << cx0 << ", " << cy0 << "), r = " << cr0 << "\n";
  if (cr1 > 0.0)
    std::cout << "  cylinder #2   : (" << cx1 << ", " << cy1 << "), r = " << cr1 << "\n";
  std::cout << "  steps         : " << steps << "\n"
            << "  output every  : " << every << " (0 = off)\n"
            << "  output prefix : " << out_pref << "\n"
            << "  probe at      : (" << px << ", " << py << ")\n"
            << "  probe csv     : " << probe_csv << "\n"
            << "  probe every   : " << probe_every << " (0 = off)\n";

  if (every > 0) {
    std::cout << "  timing note   : output is enabled, so Wall time/MLUPS include "
              << "HDF5 output and host post-processing. Use every=0 for CUDA performance sweeps.\n";
  }

  XDMFWriter writer(out_pref, nx, ny);
  if (every > 0) writer.write_mask(solver);

  using clk = std::chrono::high_resolution_clock;
  const auto t0 = clk::now();

  for (std::size_t step = 1; step <= steps; ++step) {
    solver.step(step);

    if (every > 0 && step % every == 0) {
      writer.write_snapshot(solver, double(step));
      std::cout << "\r  step " << step << " / " << steps << std::flush;
    }
  }

  // Make sure all asynchronous CUDA work is complete before stopping the timer.
  solver.synchronize();

  if (every > 0) std::cout << "\n";

  const double dt    = std::chrono::duration<double>(clk::now() - t0).count();
  const double mlups = double(nx) * double(ny) * double(steps) / dt / 1.0e6;
  std::cout << "Wall time : " << dt    << " s\n"
            << "MLUPS     : " << mlups << "\n";

  // Copy all probe samples back once, after timing.
  solver.write_probe_csv(probe_csv);

  return 0;
  } catch (const std::exception & e) {
    std::cerr << "Error: " << e.what() << '\n';
    return 1;
  }
}
