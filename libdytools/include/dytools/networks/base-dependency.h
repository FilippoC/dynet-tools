#pragma once

#include <utility>

#include "dytools/data/conll.h"
#include "dytools/builders/bilstm.h"
#include "dytools/builders/biaffine.h"
#include "dytools/builders/biaffine_tagger.h"
#include "dytools/builders/tagger.h"

#include "dytools/networks/network.h"

namespace dytools
{

struct BaseDependencySettings
{
    BiLSTMSettings first_bilstm;
    BiLSTMSettings second_bilstm;
    TaggerSettings tagger;
    BiAffineSettings biaffine;
    BiAffineTaggerSettings biaffine_tagger;

    template<class Archive>
    void serialize(Archive& ar, const unsigned int)
    {
        ar & first_bilstm;
        ar & second_bilstm;
        ar & tagger;
        ar & biaffine;
        ar & biaffine_tagger;
    }
};

struct BaseDependencyNetwork : public Builder, Network<ConllSentence>
{
    const BaseDependencySettings settings;
    dynet::ParameterCollection local_pc;

    BiLSTMBuilder first_bilstm;
    BiLSTMBuilder second_bilstm;
    TaggerBuilder tagger;
    BiAffineBuilder biaffine;
    BiAffineTaggerBuilder biaffine_tagger;

    dynet::ComputationGraph* _cg;

    BaseDependencyNetwork(
            dynet::ParameterCollection& pc,
            const BaseDependencySettings& settingss,
            std::shared_ptr<dytools::Dict> tagger_dict,
            std::shared_ptr<dytools::Dict> label_dict,
            unsigned embeddings_size
    );
    virtual void new_graph(dynet::ComputationGraph& cg, bool update = true);
    void set_is_training(bool value) override;

    virtual unsigned get_embeddings_size() const = 0;
    virtual std::vector<dynet::Expression> get_embeddings(const ConllSentence &sentence) = 0;

    std::tuple<dynet::Expression, dynet::Expression, dynet::Expression> logits(const ConllSentence &sentence);
    dynet::Expression dependency_logits(const ConllSentence &sentence);
    dynet::Expression tag_logits(const ConllSentence &sentence);

    dynet::Expression labeled_loss(const dytools::ConllSentence &sentence) override;
};

struct DependencyParserEvaluator
{
    float operator()(BaseDependencyNetwork* network, const std::vector<dytools::ConllSentence>& data) const;
};


}