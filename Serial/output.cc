#include "output.hh"
#include "lbm.hh"

#include <hdf5.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

namespace {

void
write_h5_double(const std::string & fname,
                const std::string & dset,
                const std::vector<double> & data,
                std::size_t nx, std::size_t ny,
                bool truncate)
{
  hid_t file = truncate
    ? H5Fcreate(fname.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT)
    : H5Fopen  (fname.c_str(), H5F_ACC_RDWR,                H5P_DEFAULT);
  if (file < 0) {
    std::cerr << "Could not open HDF5 file " << fname << "\n";
    return;
  }
  hsize_t dims[2] = { hsize_t(ny), hsize_t(nx) };
  hid_t   space   = H5Screate_simple(2, dims, nullptr);
  hid_t   dataset = H5Dcreate2(file, dset.c_str(), H5T_IEEE_F64LE,
                               space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
  H5Dclose(dataset);
  H5Sclose(space);
  H5Fclose(file);
}

void
write_h5_uint8(const std::string & fname,
               const std::string & dset,
               const std::vector<std::uint8_t> & data,
               std::size_t nx, std::size_t ny)
{
  hid_t file = H5Fcreate(fname.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  if (file < 0) {
    std::cerr << "Could not create HDF5 file " << fname << "\n";
    return;
  }
  hsize_t dims[2] = { hsize_t(ny), hsize_t(nx) };
  hid_t   space   = H5Screate_simple(2, dims, nullptr);
  hid_t   dataset = H5Dcreate2(file, dset.c_str(), H5T_STD_U8LE,
                               space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  H5Dwrite(dataset, H5T_NATIVE_UINT8, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
  H5Dclose(dataset);
  H5Sclose(space);
  H5Fclose(file);
}

std::string
strip_dir(const std::string & p)
{
  const auto pos = p.find_last_of("/\\");
  return (pos == std::string::npos) ? p : p.substr(pos + 1);
}

std::string
parent_dir(const std::string & p)
{
  const auto pos = p.find_last_of("/\\");
  return (pos == std::string::npos) ? std::string{} : p.substr(0, pos);
}

// Recursive `mkdir -p`. Errors other than "already exists" are silently
// ignored so that the file-creation step downstream surfaces the real
// problem to the user.
void
mkdir_p(const std::string & path)
{
  if (path.empty()) return;
  std::string acc;
  for (char c : path) {
    if ((c == '/' || c == '\\') && !acc.empty()) {
      ::mkdir(acc.c_str(), 0755);
    }
    acc.push_back(c);
  }
  if (!acc.empty()) ::mkdir(acc.c_str(), 0755);
}

std::string
indexed_suffix(std::size_t i)
{
  std::ostringstream os;
  os << std::setw(6) << std::setfill('0') << i;
  return os.str();
}

}  // namespace

XDMFWriter::XDMFWriter(const std::string & prefix, std::size_t nx, std::size_t ny)
  : prefix_(prefix), basename_(strip_dir(prefix)),
    nx_(nx), ny_(ny), has_mask_(false)
{
  mkdir_p(parent_dir(prefix_));
}

void
XDMFWriter::write_mask(const LBM & solver)
{
  std::vector<std::uint8_t> mask(nx_ * ny_);
  for (std::size_t y = 0; y < ny_; ++y)
    for (std::size_t x = 0; x < nx_; ++x)
      mask[y * nx_ + x] = solver.is_solid(x, y) ? 1 : 0;

  write_h5_uint8(prefix_ + "_mask.h5", "solid", mask, nx_, ny_);
  has_mask_ = true;
  write_root_xdmf();
}

void
XDMFWriter::write_snapshot(const LBM & solver, double t)
{
  const std::size_t step_idx = times_.size();
  const std::string fname    = prefix_ + "_" + indexed_suffix(step_idx) + ".h5";

  std::vector<double> rho_v (nx_ * ny_);
  std::vector<double> ux_v  (nx_ * ny_);
  std::vector<double> uy_v  (nx_ * ny_);
  std::vector<double> vor_v (nx_ * ny_);
  for (std::size_t y = 0; y < ny_; ++y) {
    for (std::size_t x = 0; x < nx_; ++x) {
      const std::size_t k = y * nx_ + x;
      rho_v[k] = solver.rho(x, y);
      ux_v [k] = solver.ux (x, y);
      uy_v [k] = solver.uy (x, y);
      vor_v[k] = solver.vorticity(x, y);
    }
  }

  write_h5_double(fname, "rho",       rho_v, nx_, ny_, /*truncate=*/true);
  write_h5_double(fname, "ux",        ux_v,  nx_, ny_, /*truncate=*/false);
  write_h5_double(fname, "uy",        uy_v,  nx_, ny_, /*truncate=*/false);
  write_h5_double(fname, "vorticity", vor_v, nx_, ny_, /*truncate=*/false);

  times_.push_back(t);
  write_root_xdmf();
}

void
XDMFWriter::write_root_xdmf() const
{
  const std::string xmf_path = prefix_ + ".xdmf";
  std::ofstream xmf(xmf_path);
  if (!xmf) {
    std::cerr << "Could not open " << xmf_path << "\n";
    return;
  }

  xmf << "<?xml version=\"1.0\" ?>\n"
      << "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n"
      << "<Xdmf Version=\"3.0\">\n"
      << "  <Domain>\n"
      << "    <Grid Name=\"LBM\" GridType=\"Collection\" CollectionType=\"Temporal\">\n";

  for (std::size_t i = 0; i < times_.size(); ++i) {
    const std::string tag = indexed_suffix(i);
    const std::string h5  = basename_ + "_" + tag + ".h5";

    xmf << "      <Grid Name=\"step_" << tag << "\" GridType=\"Uniform\">\n"
        << "        <Time Value=\"" << times_[i] << "\"/>\n"
        << "        <Topology TopologyType=\"2DCoRectMesh\" Dimensions=\""
        << ny_ << " " << nx_ << "\"/>\n"
        << "        <Geometry GeometryType=\"ORIGIN_DXDY\">\n"
        << "          <DataItem Format=\"XML\" Dimensions=\"2\">0.0 0.0</DataItem>\n"
        << "          <DataItem Format=\"XML\" Dimensions=\"2\">1.0 1.0</DataItem>\n"
        << "        </Geometry>\n";

    auto write_attr = [&](const char * name, const char * dset) {
      xmf << "        <Attribute Name=\"" << name
          << "\" AttributeType=\"Scalar\" Center=\"Node\">\n"
          << "          <DataItem Format=\"HDF\" Dimensions=\""
          << ny_ << " " << nx_ << "\" NumberType=\"Float\" Precision=\"8\">"
          << h5 << ":/" << dset
          << "</DataItem>\n"
          << "        </Attribute>\n";
    };
    write_attr("rho",       "rho");
    write_attr("ux",        "ux");
    write_attr("uy",        "uy");
    write_attr("vorticity", "vorticity");

    if (has_mask_) {
      xmf << "        <Attribute Name=\"solid\" AttributeType=\"Scalar\" Center=\"Node\">\n"
          << "          <DataItem Format=\"HDF\" Dimensions=\""
          << ny_ << " " << nx_ << "\" NumberType=\"UChar\">"
          << basename_ << "_mask.h5:/solid"
          << "</DataItem>\n"
          << "        </Attribute>\n";
    }

    xmf << "      </Grid>\n";
  }

  xmf << "    </Grid>\n"
      << "  </Domain>\n"
      << "</Xdmf>\n";
}
