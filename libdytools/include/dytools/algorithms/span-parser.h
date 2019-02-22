#pragma once

#include <utility>
#include <vector>
#include <set>

namespace dytools
{

typedef std::pair<unsigned, unsigned> Span;
typedef std::set<Span> Tree;

Tree binary_span_parser(const unsigned size, const std::vector<float> &weights);

}