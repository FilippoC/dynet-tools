#pragma once

#include "dynet/expr.h"

namespace dytools
{

dynet::Expression make_sinusoidal_position_encoding(dynet::ComputationGraph &cg, const dynet::Dim& dim);

}