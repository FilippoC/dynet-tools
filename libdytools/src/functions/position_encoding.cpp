#include "dytools/functions/position_encoding.h"

#include <vector>
#include <cmath>

namespace dytools
{

// from: https://github.com/clab/dynet/blob/d65bd5e0f921087f165a44b18c1f65369c9f517d/examples/transformer/transformer.h#L604
dynet::Expression make_sinusoidal_position_encoding(dynet::ComputationGraph &cg, const dynet::Dim& dim)
{
    unsigned nUnits = dim[0];
    unsigned nWords = dim[1];

    float num_timescales = nUnits / 2;
    float log_timescale_increment = std::log(10000.f) / (num_timescales - 1.f);

    std::vector<float> vSS(nUnits * nWords, 0.f);
    for(unsigned p = 0; p < nWords; ++p) {
        for(int i = 0; i < num_timescales; ++i) {
            float v = p * std::exp(i * -log_timescale_increment);
            vSS[p * nUnits + i] = std::sin(v);
            vSS[p * nUnits + num_timescales + i] = std::cos(v);
        }
    }

    return dynet::input(cg, {nUnits, nWords}, vSS);
}

}