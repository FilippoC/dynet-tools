#include "dytools/builders/tagger.h"

namespace dytools
{

TaggerBuilder::TaggerBuilder(dynet::ParameterCollection& pc, const TaggerSettings& settings, std::shared_ptr<dynet::Dict> dict, unsigned dim_input) :
    settings(settings),
    local_pc(pc.add_subcollection("tagger")),
    dict(dict),
    builder(dim_input, dict->size(), local_pc, settings.output_bias)
{
    std::cerr
        << "Tagger\n"
        << " num classes: " << dict->size() << "\n"
        << " classes: " << dict->convert((int) 0)
        ;
    for (int i = 0 ; i < dict->size() ; ++i)
        std::cerr << "\t" << dict->convert(i);
    std::cerr << "\n\n";
}

void TaggerBuilder::new_graph(dynet::ComputationGraph& cg, bool update)
{
    builder.new_graph(cg, update);
}

dynet::Expression TaggerBuilder::operator()(const dynet::Expression& input)
{
    return builder.full_logits(input);
}

dynet::Expression TaggerBuilder::operator()(const std::vector<dynet::Expression>& input)
{
    return (*this)(dynet::concatenate_cols(input));
}

}