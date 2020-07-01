#include <H5Cpp.h>
#include <fmt/format.h>
#include <fmt/ostream.h>
#include <string>
#include <iostream>

struct Galaxy {
  float Pos[3];
  float StellarMass;
};

std::vector<Galaxy> read_meraxes(const std::string fname) {
  H5::H5File file(fname, H5F_ACC_RDONLY);

  size_t total_n_galaxies = 0;
  for (int core=0; core < 8; ++core) {
      auto dataset = file.openDataSet(fmt::format("/Snap100/Core{}/Galaxies", core));
      total_n_galaxies += dataset.getSpace().getSimpleExtentNpoints();
  }

  std::vector<Galaxy> galaxies(total_n_galaxies);
  H5::CompType galaxy_type(sizeof(Galaxy));
  hsize_t size_[] = {3};

  galaxy_type.insertMember(
      "Pos", HOFFSET(Galaxy, Pos),
      H5::ArrayType(H5::DataType(H5::PredType::NATIVE_FLOAT), 1, size_));
  galaxy_type.insertMember("StellarMass", HOFFSET(Galaxy, StellarMass),
                           H5::PredType::NATIVE_FLOAT);

  size_t n_galaxies = 0;
  for (int core=0; core < 8; ++core) {
      auto dataset = file.openDataSet(fmt::format("/Snap100/Core{}/Galaxies", core));
      dataset.read(&galaxies[n_galaxies], galaxy_type);
      n_galaxies += dataset.getSpace().getSimpleExtentNpoints();
  }

  return galaxies;
}


class Grid {
    public:
        std::vector<double> data;
        std::array<size_t,3> dim;
        std::array<double,3> box_size;
        std::array<double,3> fac;
        size_t n_cell;

    Grid(const std::array<size_t, 3> dim_, const std::array<double, 3> box_size_)
    : dim { dim_ }
    , n_cell {dim_[0] * dim_[1] * dim_[2]}
    , fac {dim[0] / box_size_[0], dim[1] / box_size_[1], dim[2] / box_size_[2]}
    , box_size { box_size_ } {data.assign(n_cell, 0.0);};

    constexpr size_t index(const size_t i, const size_t j, const size_t k)
    {
        return k + dim[1] * (j + dim[0] * i);
    }

    double at(const size_t i, const size_t j, const size_t k) {
        return data.at(index(i, j, k));
    }

    void assign_CIC(const float pos[3], float value) {
      // Workout the CIC coefficients (taken from SWIFT)
      auto i = (int)(fac[0] * pos[0]);
      if (i >= dim[0]) i = dim[0] - 1;
      const double dx = fac[0] * pos[0] - i;
      const double tx = 1. - dx;

      auto j = (int)(fac[1] * pos[1]);
      if (j >= dim[1]) j = dim[1] - 1;
      const double dy = fac[1] * pos[1] - j;
      const double ty = 1. - dy;

      int k = (int)(fac[2] * pos[2]);
      if (k >= dim[2]) k = dim[2] - 1;
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
    
};

int main(int argc, char *argv[]) {

  fmt::print("Reading galaxies...");
  std::cout.flush();

  auto galaxies =
      read_meraxes("/home/smutch/freddos/meraxes/mhysa_paper/tiamat_runs/"
                   "smf_only/the_bathroom_sink/single_run/output/meraxes.hdf5");
  size_t n_galaxies = galaxies.size();

  fmt::print(" done\n");
  std::cout.flush();

  // {
  //   int ii = 0;
  //   for (const auto galaxy : galaxies) {
  //     fmt::print("galaxy[{}] = ([{}], {})\n", ii, fmt::join(galaxy.Pos, ","), galaxy.StellarMass);
  //     ii++;
  //     if (ii == 10)
  //       break;
  //   }
  // }

  std::array<size_t, 3> dim = {128, 128, 128};
  std::array<double, 3> box_size = {67.8, 67.8, 67.8};
  Grid grid(dim, box_size);

  fmt::print("Assigning galaxies to grid (CIC)...");
  std::cout.flush();


#pragma omp parallel for shared(grid, galaxies) private(n_galaxies)
  for (int ii=0; ii<n_galaxies; ++ii) {
      grid.assign_CIC(galaxies[ii].Pos, galaxies[ii].StellarMass);
  }

  fmt::print(" done\n");

  return 0;
}
