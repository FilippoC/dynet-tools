#include "dytools/loss/dependency.h"

namespace dytools
{

dynet::Expression head_neg_log_likelihood(const dynet::Expression& input, const std::vector<unsigned> &heads)
{
    const unsigned size = heads.size();
    dynet::ComputationGraph& cg = *(input.pg);

    // mask the diagonal
    std::vector<unsigned> diag_idx;
    for (unsigned i = 1u; i < size; ++i)
        diag_idx.push_back(i + i * size);
    std::vector<float> diag_values(diag_idx.size(), -std::numeric_limits<float>::infinity());
    const auto diag_mask = dynet::input(cg, {size, size}, diag_idx, diag_values);

    const auto masked_weights = input + diag_mask;

    // construct loss
    const auto batched_weights = dynet::reshape(masked_weights, dynet::Dim({size}, size));
    const auto batched_loss = dynet::pickneglogsoftmax(batched_weights, heads);

    // mask the loss of the root word
    std::vector<float> loss_mask_values(size, 1.f);
    loss_mask_values.at(0u) = 0.f;
    const auto masked_loss = batched_loss * dynet::input(cg, dynet::Dim({1u}, size), loss_mask_values);

    return dynet::sum_batches(masked_loss);
}


}