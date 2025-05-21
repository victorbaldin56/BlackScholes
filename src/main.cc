#include <cmath>
#include <random>

#include "benchmark/benchmark.h"
#include "bs/compute.hh"
#include "omp.h"

constexpr std::size_t kNumTrajectories = 100000;
constexpr std::size_t kNumOptions = 100;

namespace {

void fillParams(std::array<double, kNumOptions>& p,
                std::uniform_real_distribution<double>& dist,
                std::mt19937_64& rng) {
  std::generate(p.begin(), p.end(), [&] { return dist(rng); });
}
}  // namespace

class BSBenchmark : public benchmark::Fixture {
 public:
  void SetUp(const benchmark::State& state) noexcept override {
    std::mt19937_64 rng;
    std::uniform_real_distribution<double> S0_dist{50, 200};
    std::uniform_real_distribution<double> sigma_dist{0.05, 0.2};
    std::uniform_real_distribution<double> r_dist{0.01, 0.1};
    std::uniform_real_distribution<double> T_dist{0.5, 2};
    std::uniform_real_distribution<double> K_dist{50, 200};

    fillParams(S0s_, S0_dist, rng);
    fillParams(sigmas_, sigma_dist, rng);
    fillParams(rs_, r_dist, rng);
    fillParams(Ts_, T_dist, rng);
    fillParams(Ks_, K_dist, rng);
  }

  void measureScalar(benchmark::State& state) {
    for (auto _ : state) {
#pragma omp parallel for
      for (std::size_t i = 0; i < kNumOptions; ++i) {
        bs::computeScalar(S0s_[i], sigmas_[i], rs_[i], Ts_[i], Ks_[i],
                          kNumTrajectories);
      }
    }
  }

  void measureVector(benchmark::State& state) {
    for (auto _ : state) {
#pragma omp parallel for
      for (std::size_t i = 0; i < kNumOptions; ++i) {
        bs::computeVector(S0s_[i], sigmas_[i], rs_[i], Ts_[i], Ks_[i],
                          kNumTrajectories);
      }
    }
  }

 protected:
  std::array<double, kNumOptions> S0s_;
  std::array<double, kNumOptions> sigmas_;
  std::array<double, kNumOptions> rs_;
  std::array<double, kNumOptions> Ts_;
  std::array<double, kNumOptions> Ks_;
};

BENCHMARK_DEFINE_F(BSBenchmark, scalar)(benchmark::State& state) {
  measureScalar(state);
}

BENCHMARK_DEFINE_F(BSBenchmark, vector)(benchmark::State& state) {
  measureVector(state);
}

BENCHMARK_REGISTER_F(BSBenchmark, scalar);
BENCHMARK_REGISTER_F(BSBenchmark, vector);

BENCHMARK_MAIN();
