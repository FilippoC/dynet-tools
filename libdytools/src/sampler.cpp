#include "dytools/sampler.h"

#include <algorithm>

namespace dytools
{

Sampler::Sampler(const unsigned _size) :
    size(_size),
    next_id(_size)
{
    indices.reserve(size);
    for (unsigned i = 0u ; i < size ; ++i)
        indices.emplace_back(i);
}

unsigned Sampler::next()
{
    if (next_id >= (unsigned) indices.size())
    {
        next_id = 0u;
        std::random_shuffle(indices.begin(), indices.end());
    }

    const unsigned ret = next_id;
    ++next_id;
    return ret;
}

}