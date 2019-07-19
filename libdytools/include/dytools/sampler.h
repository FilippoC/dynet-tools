#pragma once

#include <vector>

namespace dytools
{

struct Sampler
{
    const unsigned size;

    unsigned next_id;
    std::vector<unsigned> indices;

    Sampler(const unsigned _size);
    unsigned next();
};

}