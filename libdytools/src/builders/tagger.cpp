#include "dytools/builders/tagger.h"

#include "dynet/param-init.h"

namespace dytools
{

TaggerBuilder::TaggerBuilder(dynet::ParameterCollection& pc, const TaggerSettings& settings, std::shared_ptr<dynet::Dict> dict, unsigned dim_input) :
    settings(settings),
    local_pc(pc.add_subcollection("tagger")),
    dict(dict),
    p_W(settings.layers), p_bias(settings.layers),
    e_W(settings.layers), e_bias(settings.layers),
    builder((settings.layers == 0 ? dim_input : settings.dim), dict->size(), local_pc, settings.output_bias)
{
    const int zero = 0;
    for (unsigned i = 0 ; i < settings.layers ; ++i)
    {
        p_W.at(i) = local_pc.add_parameters({settings.dim, dim_input});
        p_bias.at(i) = local_pc.add_parameters({settings.dim}, dynet::ParameterInitConst(0.f));
    }

    std::cerr
        << "Tagger\n"
        << " layer / dim: " << settings.layers << " / " << settings.dim << "\n"
        << " num classes: " << dict->size() << "\n"
        << " classes: " << dict->convert(zero)
        ;
    for (int i = 0 ; i < dict->size() ; ++i)
        std::cerr << "\t" << dict->convert(i);
    std::cerr << "\n\n";
}

void TaggerBuilder::new_graph(dynet::ComputationGraph& cg, bool update)
{
    builder.new_graph(cg, update);
    for (unsigned i = 0 ; i < settings.layers ; ++i)
    {
        e_W.at(i) = dynet::parameter(cg, p_W.at(i));
        e_bias.at(i) = dynet::parameter(cg, p_bias.at(i));
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

}
