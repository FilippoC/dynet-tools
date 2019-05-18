#include "dytools/builders/tagger.h"

#include "dynet/param-init.h"

namespace dytools
{

TaggerBuilder::TaggerBuilder(dynet::ParameterCollection& pc, const TaggerSettings& settings, unsigned size, unsigned dim_input) :
    settings(settings),
    local_pc(pc.add_subcollection("tagger")),
    p_W(settings.layers), p_bias(settings.layers),
    e_W(settings.layers), e_bias(settings.layers),
    builder((settings.layers == 0 ? dim_input : settings.dim), size, local_pc, settings.output_bias)
{
    for (unsigned i = 0 ; i < settings.layers ; ++i)
    {
        p_W.at(i) = local_pc.add_parameters({settings.dim, dim_input});
        p_bias.at(i) = local_pc.add_parameters({settings.dim}, dynet::ParameterInitConst(0.f));
    }

    std::cerr
        << "Tagger\n"
        << " layer / dim: " << settings.layers << " / " << settings.dim << "\n"
        << " num classes: " << size << "\n"
        ;
}

void TaggerBuilder::new_graph(dynet::ComputationGraph& cg, bool, bool update)
{
    _cg = &cg;
    builder.new_graph(cg, update);
    for (unsigned i = 0 ; i < settings.layers ; ++i)
    {
        if (update)
        {
            e_W.at(i) = dynet::parameter(cg, p_W.at(i));
            e_bias.at(i) = dynet::parameter(cg, p_bias.at(i));
        }
        else
        {
            e_W.at(i) = dynet::const_parameter(cg, p_W.at(i));
            e_bias.at(i) = dynet::const_parameter(cg, p_bias.at(i));
        }
    }
}

dynet::Expression TaggerBuilder::full_logits(const dynet::Expression &input)
{
    auto repr = input;
    for (unsigned i = 0 ; i < settings.layers ; ++i)
        repr = dynet::tanh(e_W.at(i) * repr + e_bias.at(i));

    return builder.full_logits(repr);
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
