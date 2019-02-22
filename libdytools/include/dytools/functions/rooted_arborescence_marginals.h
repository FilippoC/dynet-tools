#pragma once

#include "dynet/expr.h"

namespace dytools
{

dynet::Expression rooted_arborescence_marginals(
        dynet::ComputationGraph& cg,
        const dynet::Expression& arc_weights,
        const std::vector<unsigned>& n_vertices
);

dynet::Expression rooted_arborescence_marginals(
        dynet::ComputationGraph& cg,
        const dynet::Expression& arc_weights,
        const std::vector<unsigned>* n_vertices = nullptr
);

}