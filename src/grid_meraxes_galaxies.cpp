#include <ProgressBar.hpp>
#include <H5Cpp.h>
#include <fmt/format.h>
#include <fmt/ostream.h>
#include <iostream>
#include <string>
#include <boost/program_options.hpp>

struct Galaxy {
  std::array<float, 3> Pos;
  float StellarMass;
};

auto read_meraxes(const std::string &fname, const int snapshot,
                  std::vector<Galaxy> &galaxies) -> double {
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
  galaxy_type.insertMember("StellarMass", HOFFSET(Galaxy, StellarMass),
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

  return box_size;
}

class Grid {
public:
  std::vector<double> data;
  std::array<size_t, 3> dim;
  std::array<double, 3> box_size;
  std::array<double, 3> fac;
  size_t n_cell;

  Grid(const std::array<size_t, 3> dim_, const std::array<double, 3> box_size_)
      : dim{dim_}, n_cell{dim_[0] * dim_[1] * dim_[2]},
        fac{dim[0] / box_size_[0], dim[1] / box_size_[1],
            dim[2] / box_size_[2]},
        box_size{box_size_} {
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

  void write(const std::string &fname) {
    H5::H5File file(fname, H5F_ACC_TRUNC);

    std::array<hsize_t, 1> size_ = {3};

    std::array<int, 3> data = {0};
    for (int ii = 0; ii < 3; ++ii) {
      data[ii] = this->dim[ii];
    }
    file.createAttribute("dim", H5::PredType::NATIVE_INT,
                         H5::DataSpace(1, size_.data()))
        .write(H5::PredType::NATIVE_INT, data.data());

    size_[0] = this->n_cell;
    file.createDataSet("StellarMass", H5::PredType::NATIVE_DOUBLE,
                       H5::DataSpace(1, size_.data()))
        .write(this->data.data(), H5::PredType::NATIVE_DOUBLE);
  }
};

auto main(int argc, char *argv[]) -> int {

  namespace po = boost::program_options;

  po::options_description desc("Allowed options");
  desc.add_options()("help,h", "produce help message")
                    ("input", po::value<std::string>(), "input meraxes file")
                    ("output", po::value<std::string>(), "output grid file")
                    ("dim", po::value<size_t>()->default_value(128), "grid dimensionality");

  po::positional_options_description pos_desc;
  pos_desc.add("input", 1).add("output", 1).add("dim", 1);

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).positional(pos_desc).run(), vm);
  po::notify(vm);

  if (vm.count("help") || (vm.size() == 0)) {
    fmt::print("Usage:\n  grid_meraxes_galaxies [input meraxes file] [output grid file] [dim]\n\n");
    fmt::print("{}", desc);
    return 1;
  }

  fmt::print("Reading galaxies...\n");

  std::vector<Galaxy> galaxies;
  const auto box_size =
      read_meraxes(vm["input"].as<std::string>(), 100, galaxies);
  const size_t n_galaxies = galaxies.size();

  std::array<size_t, 3> dim;
  dim.fill(vm["dim"].as<size_t>());
  Grid grid(dim, {box_size, box_size, box_size});

  fmt::print("Assigning galaxies to grid (CIC).\n");

#pragma omp parallel for default(none)                                         \
    shared(grid, galaxies) firstprivate(n_galaxies)
  for (int ii = 0; ii < n_galaxies; ++ii) {
    grid.assign_CIC(galaxies[ii].Pos, galaxies[ii].StellarMass);
  }

  fmt::print("Creating output.\n");

  grid.write(vm["output"].as<std::string>());

  return 0;
}

// vim: set shiftwidth=2 tabstop=2:
