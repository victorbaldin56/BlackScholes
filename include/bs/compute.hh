#pragma once

#include <cstddef>

namespace bs {

double computeScalar(double S0, double sigma, double r, double T, double K,
                     std::size_t N);
double computeVector(double S0, double sigma, double r, double T, double K,
                     std::size_t N);
}
