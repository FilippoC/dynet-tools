#pragma once

#include <dynet/expr.h>
#include <vector>

namespace dytools
{

dynet::Expression head_neg_log_likelihood(const dynet::Expression& inpur, const std::vector<unsigned> &heads);

}