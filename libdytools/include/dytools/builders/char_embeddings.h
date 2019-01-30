#pragma once

#include "dynet/expr.h"

#include "dytools/data/conll.h"
#include "dytools/builders/bilstm.h"
#include "dytools/builders/builder.h"

namespace dytools
{

struct CharEmbeddingsSettings
{
    unsigned dim = 100;
    BiLSTMSettings bilstm;

    template<class Archive>
    void serialize(Archive& ar, const unsigned int)
    {
        ar & dim;
        ar & bilstm;
    }
};

struct CharEmbeddingsBuilder : public Builder
{
    const CharEmbeddingsSettings settings;
    dynet::ParameterCollection local_pc;
    std::shared_ptr<dynet::Dict> dict;

    dynet::LookupParameter lp;
    BiLSTMBuilder bilstm;

    dynet::ComputationGraph* _cg;
    bool _update = true;

    CharEmbeddingsBuilder(dynet::ParameterCollection& pc, const CharEmbeddingsSettings& settings, std::shared_ptr<dynet::Dict> dict);

    void new_graph(dynet::ComputationGraph& cg, bool update=true);
    void set_is_training(bool value) override;

    dynet::Expression operator()(const std::string& c);
    dynet::Expression operator()(const ConllToken& token);
    std::vector<dynet::Expression> operator()(const ConllSentence& sentence);

    unsigned output_rows() const;
};

}