#include "dytools/builders/token_embeddings.h"

namespace dytools
{

TokenEmbeddingsBuilder::TokenEmbeddingsBuilder(dynet::ParameterCollection& pc, const TokenEmbeddingsSettings& settings, std::shared_ptr<dynet::Dict> dict) :
    settings(settings),
    local_pc(pc.add_subcollection("embstoken")),
    dict(dict)
{
    lp = pc.add_lookup_parameters(dict->size(), {settings.dim});

        std::cerr
        << "Token embeddings\n"
        << " dim: " << settings.dim << "\n"
        << " vocabulary size: " << dict->size() << "\n"
        << "\n"
        ;
}

void TokenEmbeddingsBuilder::new_graph(dynet::ComputationGraph& cg, bool update)
{
    _cg = &cg;
    _update = update;
}

dynet::Expression TokenEmbeddingsBuilder::operator()(const std::string& str)
{
    if (_update)
        return dynet::lookup(*_cg, lp, dict->convert(str));
    else
        return dynet::const_lookup(*_cg, lp, dict->convert(str));
}

dynet::Expression TokenEmbeddingsBuilder::operator()(const ConllToken& token)
{
    const std::string word = token.word;
    return (*this)(word);
}

std::vector<dynet::Expression> TokenEmbeddingsBuilder::operator()(const ConllSentence& sentence)
{
    std::vector<dynet::Expression> embs;
    for (auto const& token : sentence)
        embs.push_back((*this)(token));
    return embs;
}

unsigned TokenEmbeddingsBuilder::output_rows() const
{
    return settings.dim;
}

}