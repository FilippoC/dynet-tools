#pragma once

#include "dynet/expr.h"

namespace dytools
{

dynet::Expression main_diagonal_mask(dynet::ComputationGraph& cg, const dynet::Dim dim, const float value=1.f);

dynet::Expression all_but_first_vector_mask(dynet::ComputationGraph& cg, const unsigned size);

}