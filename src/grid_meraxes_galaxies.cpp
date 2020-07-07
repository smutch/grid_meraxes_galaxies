#include <H5Cpp.h>
#include <ProgressBar.hpp>
#include <boost/program_options.hpp>
#include <fmt/format.h>
#include <fmt/ostream.h>
#include <iostream>
#include <string>

struct Galaxy {
  std::array<float, 3> Pos;
  float value;
};

auto read_meraxes(const std::string &fname, const int snapshot,
                  const std::string &prop, std::vector<Galaxy> &galaxies)
    -> std::tuple<double, double> {
  H5::H5File file(fname, H5F_ACC_RDONLY);

  int n_cores = 0;
  {
    auto attr = file.openAttribute("NCores");
    attr.read(attr.getDataType(), &n_cores);
  }

  double box_size = 0.0;
  {
    auto attr = file.openGroup("InputParams").openAttribute("BoxSize");
    attr.read(attr.getDataType(), &box_size);
  }

  double hubble_h = 0.0;
  {
    auto attr = file.openGroup("InputParams").openAttribute("Hubble_h");
    attr.read(attr.getDataType(), &hubble_h);
  }

  size_t total_n_galaxies = 0;
  for (int core = 0; core < n_cores; ++core) {
    auto dataset = file.openDataSet(
        fmt::format("/Snap{:03d}/Core{}/Galaxies", snapshot, core));
    total_n_galaxies += dataset.getSpace().getSimpleExtentNpoints();
  }

  galaxies.resize(total_n_galaxies);
  H5::CompType galaxy_type(sizeof(Galaxy));
  const std::array<hsize_t, 1> size_ = {3};

  galaxy_type.insertMember(
      "Pos", HOFFSET(Galaxy, Pos),
      H5::ArrayType(H5::DataType(H5::PredType::NATIVE_FLOAT), 1, size_.data()));
  galaxy_type.insertMember(prop, HOFFSET(Galaxy, value),
                           H5::PredType::NATIVE_FLOAT);

  size_t n_galaxies = 0;
  progresscpp::ProgressBar progress_bar(n_cores, 70, '#', '-');
  for (int core = 0; core < n_cores; ++core, ++progress_bar) {
    auto dataset = file.openDataSet(
        fmt::format("/Snap{:03d}/Core{}/Galaxies", snapshot, core));
    dataset.read(&galaxies[n_galaxies], galaxy_type);
    n_galaxies += dataset.getSpace().getSimpleExtentNpoints();
    progress_bar.display();
  }

  progress_bar.done();

  return std::make_tuple(box_size, hubble_h);
}

class Grid {
public:
  std::vector<double> data;
  std::array<size_t, 3> dim;
  std::array<double, 3> box_size;
  std::array<double, 3> fac;
  size_t n_cell;
  double hubble_h;

  Grid(const std::array<size_t, 3> dim_, const std::array<double, 3> box_size_,
       const double hubble_h_)
      : dim{dim_}, n_cell{dim_[0] * dim_[1] * dim_[2]},
        fac{dim[0] / box_size_[0], dim[1] / box_size_[1],
            dim[2] / box_size_[2]},
        box_size{box_size_}, hubble_h{hubble_h_} {
    data.assign(n_cell, 0.0);
  };

  constexpr auto index(const size_t i, const size_t j, const size_t k)
      -> size_t {
    return (k % dim[2]) + dim[1] * ((j % dim[1]) + dim[0] * (i % dim[0]));
  }

  auto at(const size_t i, const size_t j, const size_t k) -> double {
    return data.at(index(i, j, k));
  }

  void assign_CIC(const std::array<float, 3> pos, const float value) {
    // Workout the CIC coefficients (taken from SWIFT)
    auto i = (int)(fac[0] * pos[0]);
    if (i >= dim[0]) {
      i = dim[0] - 1;
    }
    const double dx = fac[0] * pos[0] - i;
    const double tx = 1. - dx;

    auto j = (int)(fac[1] * pos[1]);
    if (j >= dim[1]) {
      j = dim[1] - 1;
    }
    const double dy = fac[1] * pos[1] - j;
    const double ty = 1. - dy;

    int k = (int)(fac[2] * pos[2]);
    if (k >= dim[2]) {
      k = dim[2] - 1;
    }
    const double dz = fac[2] * pos[2] - k;
    const double tz = 1. - dz;

#pragma omp critical
    {
      data[index(i + 0, j + 0, k + 0)] += value * tx * ty * tz;
      data[index(i + 0, j + 0, k + 1)] += value * tx * ty * dz;
      data[index(i + 0, j + 1, k + 0)] += value * tx * dy * tz;
      data[index(i + 0, j + 1, k + 1)] += value * tx * dy * dz;
      data[index(i + 1, j + 0, k + 0)] += value * dx * ty * tz;
      data[index(i + 1, j + 0, k + 1)] += value * dx * ty * dz;
      data[index(i + 1, j + 1, k + 0)] += value * dx * dy * tz;
      data[index(i + 1, j + 1, k + 1)] += value * dx * dy * dz;
    }
  }

  void write(const std::string &fname, const int snapshot) {
    H5::H5File file;
    try {
      H5::Exception::dontPrint();
      file = H5::H5File(fname, H5F_ACC_RDWR);
    } catch (const H5::FileIException &) {
      file = H5::H5File(fname, H5F_ACC_TRUNC);
    }

    std::array<hsize_t, 3> size_;
    for (int ii = 0; ii < 3; ++ii) {
      size_[ii] = (hsize_t)dim[ii];
    }

    auto plist = H5::DSetCreatPropList();
    size_[0] = 1;
    plist.setChunk(3, size_.data());
    plist.setDeflate(7);

    size_[0] = (hsize_t)dim[0];
    auto ds = file.createDataSet(fmt::format("Snap{:03d}", snapshot),
                                 H5::PredType::NATIVE_DOUBLE,
                                 H5::DataSpace(3, size_.data()), plist);
    ds.write(data.data(), H5::PredType::NATIVE_DOUBLE);

    std::array<int, 3> h5dims;
    for (int ii = 0; ii < 3; ++ii) {
      h5dims[ii] = dim[ii];
    }

    size_[0] = 3;
    auto box_size = this->box_size;
    auto hubble_h = this->hubble_h;
    std::for_each(box_size.begin(), box_size.end(), [hubble_h](auto &v){return v /= hubble_h;});
    ds.createAttribute("dim", H5::PredType::NATIVE_INT,
                       H5::DataSpace(1, size_.data()))
        .write(H5::PredType::NATIVE_INT, h5dims.data());
    ds.createAttribute("box_size", H5::PredType::NATIVE_DOUBLE,
                       H5::DataSpace(1, size_.data()))
        .write(H5::PredType::NATIVE_DOUBLE, box_size.data());

    const std::string units = "log10(M/Msun)";
    const H5::StrType str_type(H5::PredType::C_S1, units.length());
    ds.createAttribute("units", str_type, H5::DataSpace())
        .write(str_type, units.data());
  }

  void update_units() {
#pragma omp parallel for default(none) shared(data)                            \
    firstprivate(n_cell, hubble_h)
    for (int ii = 0; ii < n_cell; ++ii) {
      double &val = data.at(ii);
      val = log10(val / hubble_h) + 10.0;
    }
  }
};

auto main(int argc, char *argv[]) -> int {

  namespace po = boost::program_options;

  int snapshot = -1;
  po::options_description desc("Allowed options");
  desc.add_options()("help,h", "produce help message")("input",
                                                       "input meraxes file")(
      "snapshot", po::value<int>(&snapshot),
      "input snapshot")("property", "galaxy property to grid")(
      "output,o",
      "output grid file (default {input file directory}/{prop}_grids.h5)")(
      "dim", po::value<size_t>(), "grid dimensionality");

  po::positional_options_description pos_desc;
  pos_desc.add("input", 1).add("snapshot", 1).add("property", 1).add("dim", 1);

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv)
                .options(desc)
                .positional(pos_desc)
                .run(),
            vm);
  po::notify(vm);

  if (vm.count("help") || (vm.size() == 0)) {
    fmt::print("Usage: grid_meraxes_galaxies [input meraxes file] "
               "[snapshot] [property] [dim]\n\n");
    fmt::print("{}", desc);
    return 1;
  }

  std::string property = vm["property"].as<std::string>();
  std::string output;
  if (vm["output"].empty()) {
    auto input = vm["input"].as<std::string>();
    auto pos = input.rfind("/");
    if (pos != std::string::npos) {
      output = fmt::format("{}/{}_grids.h5", input.substr(0, pos), property);
    } else {
      output = fmt::format("{}_grids.h5", property);
    }
  }

  fmt::print("Reading galaxies...\n");

  std::vector<Galaxy> galaxies;
  const auto [box_size, hubble_h] =
      read_meraxes(vm["input"].as<std::string>(), snapshot, property, galaxies);
  const size_t n_galaxies = galaxies.size();

  std::array<size_t, 3> dim;
  dim.fill(vm["dim"].as<size_t>());
  Grid grid(dim, {box_size, box_size, box_size}, hubble_h);

  fmt::print("Assigning galaxies to grid (CIC).\n");

#pragma omp parallel for default(none) shared(grid, galaxies)                  \
    firstprivate(n_galaxies)
  for (int ii = 0; ii < n_galaxies; ++ii) {
    grid.assign_CIC(galaxies[ii].Pos, galaxies[ii].value);
  }

  grid.update_units();

  fmt::print("Creating output file: {}\n", output);

  grid.write(output, snapshot);

  return 0;
}

// vim: set shiftwidth=2 tabstop=2:
