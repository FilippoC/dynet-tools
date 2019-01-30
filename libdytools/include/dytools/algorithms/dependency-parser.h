#pragma once

#include <vector>

namespace dytools
{


std::vector<unsigned> non_projective_dependency_parser(const unsigned size, const std::vector<float>& arc_weights);


void RunCLE(
        const unsigned length_,
        const std::vector<float>& scores,
        std::vector<int> *heads,
        float *value
);

void RunChuLiuEdmondsIteration(
        std::vector<bool> *disabled,
        std::vector<std::vector<int> > *candidate_heads,
        std::vector<std::vector<float> >
        *candidate_scores,
        std::vector<int> *heads,
        float *value
);

}