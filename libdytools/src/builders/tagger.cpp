#include "dytools/builders/tagger.h"

#include "dynet/param-init.h"

namespace dytools
{

TaggerBuilder::TaggerBuilder(dynet::ParameterCollection& pc, const TaggerSettings& settings, unsigned size, unsigned dim_input) :
    settings(settings),
    local_pc(pc.add_subcollection("tagger")),
    mlp(local_pc, settings.mlp, dim_input)
{
    p_W = local_pc.add_parameters({size, mlp.output_rows()});
    if (settings.output_bias)
        p_bias = local_pc.add_parameters({size}, dynet::ParameterInitConst(0.f));

    std::cerr
        << "Tagger\n"
        << " num classes: " << size << "\n"
        ;
}

void TaggerBuilder::new_graph(dynet::ComputationGraph& cg, bool train, bool update)
{
    _cg = &cg;
    mlp.new_graph(cg, train, update);
    if (update)
        e_W = dynet::parameter(cg, p_W);
    else
        e_W = dynet::const_parameter(cg, p_W);

    if (settings.output_bias)
    {
        if (update and !settings.fix_output_bias)
            e_bias = dynet::parameter(cg, p_bias);
        else
            e_bias = dynet::const_parameter(cg, p_bias);
    }
}

void TaggerBuilder::set_dropout(float value)
{
    mlp.set_dropout(value);
}

dynet::Expression TaggerBuilder::full_logits(const dynet::Expression &input)
{
    auto repr = mlp.apply(input);
    if (settings.output_bias)
        return dynet::affine_transform({e_bias, e_W, repr});
    else
        return e_W * repr;
}

dynet::Expression TaggerBuilder::neg_log_softmax(const dynet::Expression& input, unsigned idx)
{
    const auto repr = full_logits(input);
    return dynet::pickneglogsoftmax(repr, idx);
}


dynet::Expression TaggerBuilder::neg_log_softmax(const dynet::Expression& input, const std::vector<unsigned>& indices)
{
    const auto repr = full_logits(input);
    return dynet::pickneglogsoftmax(repr, indices);
}

dynet::Expression TaggerBuilder::masked_neg_log_softmax(const dynet::Expression& input, const std::vector<int>& words, unsigned* c, bool skip_first)
{
    std::vector<unsigned> indices;
    std::vector<float> v_mask;

    indices.reserve(words.size() + 1);
    v_mask.reserve(words.size() + 1);

    if (skip_first)
    {
        indices.push_back(0);
        v_mask.push_back(0.f);
    }

    unsigned counter = 0u;
    for (const int w : words)
    {
        if (w >= 0)
        {
            indices.push_back((unsigned) w);
            v_mask.push_back(1.f);
            ++counter;
        }
        else
        {
            indices.push_back(0); // don't care, will be masked
            v_mask.push_back(0.f);
        }
    }
    if (c != nullptr)
        *c = counter;

    const auto loss = neg_log_softmax(input, indices);
    const auto e_mask = dynet::input(*_cg, dynet::Dim({1}, v_mask.size()), v_mask);
    const auto ret = loss * e_mask;

    return ret;
}

}
