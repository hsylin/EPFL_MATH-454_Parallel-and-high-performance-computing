#ifndef LBM_HH
#define LBM_HH

#include <cstddef>
#include <cstdint>
#include <vector>

/**
 * @brief 2D Lattice Boltzmann solver, D2Q9 lattice with BGK collision.
 *
 * The solver simulates incompressible flow past one (or two) circular
 * cylinders inside a rectangular channel with no-slip top and bottom walls.
 * The inlet (x = 0) prescribes a uniform horizontal velocity; the outlet
 * (x = nx - 1) is a simple zero-gradient copy from the column to its left.
 *
 * Distributions are stored in structure-of-arrays layout:
 *   f_[ i * (nx*ny) + y*nx + x ]   for direction i in [0, 9).
 *
 * Lattice units are used throughout (dx = dt = 1, c = 1, c_s^2 = 1/3).
 */
class LBM
{
public:
  static constexpr int Q = 9;

  static const int    cx[Q];   ///< Discrete velocity x-components.
  static const int    cy[Q];   ///< Discrete velocity y-components.
  static const double w[Q];    ///< Equilibrium weights.
  static const int    opp[Q];  ///< Index of the opposite direction.

  /**
   * @param nx       Number of cells along x.
   * @param ny       Number of cells along y.
   * @param u_in     Inlet velocity in lattice units (must be << 1/sqrt(3)).
   * @param Re       Target Reynolds number, based on cylinder diameter.
   * @param cyl_x    Center of the (first) cylinder along x, in cell units.
   * @param cyl_y    Center of the (first) cylinder along y, in cell units.
   * @param cyl_r    Radius of the (first) cylinder, in cell units.
   */
  LBM(std::size_t nx, std::size_t ny,
      double u_in, double Re,
      double cyl_x, double cyl_y, double cyl_r);

  /// Add a second circular obstacle. No-op if r2 <= 0.
  void add_second_cylinder(double cyl2_x, double cyl2_y, double cyl2_r);

  /// Set f to the equilibrium distribution with rho = 1, u = (u_in, 0)
  /// on every fluid cell, and (0, 0) on solid cells.
  void initialize();

  /// Advance the simulation by one time step.
  void step();

  // Macroscopic accessors.
  double rho      (std::size_t x, std::size_t y) const;
  double ux       (std::size_t x, std::size_t y) const;
  double uy       (std::size_t x, std::size_t y) const;
  double vorticity(std::size_t x, std::size_t y) const;
  bool   is_solid (std::size_t x, std::size_t y) const;

  std::size_t nx()   const { return nx_; }
  std::size_t ny()   const { return ny_; }
  double      tau()  const { return tau_; }
  double      u_in() const { return u_in_; }

private:
  std::size_t idx (std::size_t x, std::size_t y)         const { return y * nx_ + x; }
  std::size_t fidx(int i, std::size_t x, std::size_t y)  const { return i * nx_ * ny_ + idx(x, y); }

  void mark_obstacle (double c_x, double c_y, double r);
  void collide       ();
  void bounce_back   ();
  void stream        ();
  void apply_inlet   ();
  void apply_outlet  ();

  std::size_t nx_, ny_;
  double u_in_;
  double tau_;

  std::vector<double>  f_;      ///< Current distributions, size 9*nx*ny.
  std::vector<double>  ftmp_;   ///< Scratch buffer for streaming.
  std::vector<uint8_t> solid_;  ///< 0 = fluid, 1 = solid.  Size nx*ny.
};

#endif  // LBM_HH
