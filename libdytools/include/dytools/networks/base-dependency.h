#pragma once

#include "dytools/data/conll.h"
#include "dytools/builders/bilstm.h"
#include "dytools/builders/biaffine.h"
#include "dytools/builders/tagger.h"

namespace dytools
{

struct BaseDependencySettings
{
    BiLSTMSettings first_bilstm;
    BiLSTMSettings second_bilstm;
    TaggerSettings tagger;
    BiAffineSettings biaffine;

    template<class Archive>
    void serialize(Archive& ar, const unsigned int)
    {
        ar & first_bilstm;
        ar & second_bilstm;
        ar & tagger;
        ar & biaffine;
    }
};

struct BaseDependencyNetwork : public Builder
{
    const BaseDependencySettings settings;
    dynet::ParameterCollection local_pc;

    BiLSTMBuilder first_bilstm;
    BiLSTMBuilder second_bilstm;
    TaggerBuilder tagger;
    BiAffineBuilder biaffine;

    dynet::ComputationGraph* _cg;

    BaseDependencyNetwork(dynet::ParameterCollection& pc, const BaseDependencySettings& settingss, std::shared_ptr<dytools::Dict> tagger_dict, unsigned embeddings_size);
    void new_graph(dynet::ComputationGraph& cg, bool update = true);
    void set_is_training(bool value) override;

    virtual unsigned get_embeddings_size() const = 0;
    virtual std::vector<dynet::Expression> get_embeddings(const ConllSentence &sentence) = 0;

    std::pair<dynet::Expression, dynet::Expression> logits(const ConllSentence &sentence);
    dynet::Expression dependency_logits(const ConllSentence &sentence);
    dynet::Expression tag_logits(const ConllSentence &sentence);
};
}