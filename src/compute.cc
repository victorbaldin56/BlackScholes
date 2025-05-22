#include "bs/compute.hh"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <random>

#include "omp.h"

namespace bs {

double computeScalar(double S0, double sigma, double r, double T, double K,
                     std::size_t N) {
  double CT = 0;
  std::mt19937_64 rng;

  double sigma2 = sigma * sigma;
  std::lognormal_distribution<double> dist{(r - sigma2 / 2) * T, sigma2 * T};
  double norm = std::exp(-r * T) / N;

  for (std::size_t i = 0; i < N; ++i) {
    double Y = dist(rng);
    double SiT = Y * S0;
    double rem = norm * std::max(0.0, SiT - K);
    CT += rem;
  }

  return CT;
}

double computeVector(double S0, double sigma, double r, double T, double K,
                     std::size_t N) {
  constexpr std::size_t kVectorSize = 8;
  constexpr std::size_t kAlignment = kVectorSize * sizeof(double);
  assert(N % kVectorSize == 0);

  std::array<std::mt19937_64, kVectorSize> rngs;
  std::array<std::lognormal_distribution<double>, kVectorSize> dists;

  double sigma2 = sigma * sigma;
  double norm = std::exp(-r * T) / N;

#pragma unroll
  for (std::size_t lane = 0; lane < kVectorSize; ++lane) {
    rngs[lane].seed(lane);
    dists[lane] =
        std::lognormal_distribution<double>{(r - sigma2 / 2) * T, sigma2 * T};
  }

  alignas(kAlignment) double CTs[kVectorSize]{};

  for (std::size_t i = 0; i < N; i += kVectorSize) {
    alignas(kAlignment) double Ys[kVectorSize];

#pragma omp simd
    for (std::size_t lane = 0; lane < kVectorSize; ++lane) {
      Ys[lane] = dists[lane](rngs[lane]);
    }

#pragma omp simd
    for (std::size_t lane = 0; lane < kVectorSize; ++lane) {
      double SiT = Ys[lane] * S0;
      double rem = norm * std::max(0.0, SiT - K);
      CTs[lane] += rem;
    }
  }

  return std::accumulate(std::begin(CTs), std::end(CTs), 0.0);
}
}  // namespace bs
