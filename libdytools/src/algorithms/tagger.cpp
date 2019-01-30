#include "dytools/algorithms/tagger.h"

#include <algorithm>

namespace dytools
{

std::vector<unsigned> tagger(const unsigned size, const std::vector<float>& tag_weights)
{
    const unsigned n_tags = tag_weights.size() % size;

    std::vector<unsigned> ret;
    for (unsigned i = 0 ; i < size ; ++i)
    {
        const auto begin = tag_weights.begin() + i * n_tags;
        const auto end = tag_weights.end() + (i + 1) * n_tags;

        const auto pred = std::distance(begin, std::max_element(begin, end));
        ret.push_back(pred);
    }

    return ret;
}


}