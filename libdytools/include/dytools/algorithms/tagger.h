#pragma once

#include <vector>

namespace dytools
{

std::vector<unsigned> tagger(const unsigned size, const std::vector<float>& tag_weights);


}