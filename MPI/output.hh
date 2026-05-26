#ifndef LBM_OUTPUT_HH
#define LBM_OUTPUT_HH

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

class LBM;

/**
 * @brief Writes LBM snapshots to HDF5 + a single XDMF index file
 *        for visualization in ParaView or PyVista.
 *
 * The XDMF file references one HDF5 file per snapshot (rho, ux, uy,
 * vorticity) plus a one-time HDF5 file containing the solid mask.
 */
class XDMFWriter
{
public:
  /**
   * @param prefix  Filename prefix, e.g., "out/lbm". The directory must
   *                already exist. The XDMF file will be "<prefix>.xdmf".
   * @param nx, ny  Grid dimensions.
   */
  XDMFWriter(const std::string & prefix, std::size_t nx, std::size_t ny);

  /// Write the static solid mask once. Call before the first snapshot.
  void write_mask(const LBM & solver);

  /// Append a snapshot at simulation time t.
  void write_snapshot(const LBM & solver, double t);

  /// Write a full global solid mask already gathered on rank 0.
  void write_mask_data(const std::vector<std::uint8_t> & mask);

  /// Append a full global snapshot already gathered on rank 0.
  void write_snapshot_data(const std::vector<double> & rho_v,
                           const std::vector<double> & ux_v,
                           const std::vector<double> & uy_v,
                           const std::vector<double> & vor_v,
                           double t);

private:
  void write_root_xdmf() const;

  std::string  prefix_;
  std::string  basename_;       ///< Prefix without leading directory.
  std::size_t  nx_, ny_;
  bool         has_mask_;
  std::vector<double> times_;
};

#endif  // LBM_OUTPUT_HH
