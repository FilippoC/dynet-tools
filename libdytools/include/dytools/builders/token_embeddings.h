#pragma once

#include "dytools/data/conll.h"
#include "dynet/expr.h"
#include "dytools/builders/builder.h"

namespace dytools
{

struct TokenEmbeddingsSettings
{
    unsigned dim = 100;

    template<class Archive>
    void serialize(Archive& ar, const unsigned int)
    {
        ar & dim;
    }
};

struct TokenEmbeddingsBuilder : public Builder
{
    const TokenEmbeddingsSettings settings;
    dynet::ParameterCollection local_pc;
    std::shared_ptr<dynet::Dict> dict;

    dynet::LookupParameter lp;
    dynet::ComputationGraph* _cg;
    bool _update = true;

    TokenEmbeddingsBuilder(dynet::ParameterCollection& pc, const TokenEmbeddingsSettings& settings, std::shared_ptr<dynet::Dict> dict);

    void new_graph(dynet::ComputationGraph& cg, bool update=true);
    dynet::Expression operator()(const std::string& str);
    dynet::Expression operator()(const ConllToken& token);
    std::vector<dynet::Expression> operator()(const ConllSentence& sentence);

    unsigned output_rows() const;
};

}