#pragma once

#include "dytools/data/conll.h"
#include "dytools/builders/embeddings.h"
#include "dytools/builders/bilstm.h"
#include "dytools/builders/biaffine.h"
#include "dytools/builders/tagger.h"
#include "dytools/training.h"

namespace dytools
{

struct DependencySettings
{
    EmbeddingsSettings embeddings;
    BiLSTMSettings first_bilstm;
    BiLSTMSettings second_bilstm;
    TaggerSettings tagger;
    BiAffineSettings biaffine;

    template<class Archive>
    void serialize(Archive& ar, const unsigned int)
    {
        ar & embeddings;
        ar & first_bilstm;
        ar & second_bilstm;
        ar & tagger;
        ar & biaffine;
    }
};

struct DependencyNetwork : public Builder
{
    const DependencySettings settings;
    dynet::ParameterCollection local_pc;

    EmbeddingsBuilder embeddings;
    BiLSTMBuilder first_bilstm;
    BiLSTMBuilder second_bilstm;
    TaggerBuilder tagger;
    BiAffineBuilder biaffine;

    dynet::ComputationGraph* _cg;

    DependencyNetwork(dynet::ParameterCollection& pc, const DependencySettings& settings, std::shared_ptr<dynet::Dict> token_dict, std::shared_ptr<dynet::Dict> char_dict, std::shared_ptr<dynet::Dict> tagger_dict);
    void new_graph(dynet::ComputationGraph& cg, bool update = true);
    void set_is_training(bool value) override;

    std::pair<dynet::Expression, dynet::Expression> full_logits(const ConllSentence& sentence);
};



struct DependencyParserTrainer : public Training<DependencyNetwork, ConllSentence>
{
    using Training<DependencyNetwork, ConllSentence>::network;

    DependencyParserTrainer(const TrainingSettings& settings, std::shared_ptr<DependencyNetwork> _network);

    dynet::Expression compute_loss(const dytools::ConllSentence& sentence) override;
    float evaluate(const std::vector<dytools::ConllSentence>& data) override;
};

}