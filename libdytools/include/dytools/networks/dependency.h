#pragma once

#include "dytools/builders/embeddings/embeddings.h"
#include "dytools/training.h"
#include "dytools/networks/base-dependency.h"

namespace dytools
{

struct DependencySettings : BaseDependencySettings
{
    EmbeddingsSettings embeddings;

    template<class Archive>
    void serialize(Archive& ar, const unsigned int)
    {
        ar & boost::serialization::base_object<BaseDependencySettings>(*this);
        ar & embeddings;
    }
};

struct DependencyNetwork : public BaseDependencyNetwork
{
    const DependencySettings settings;

    EmbeddingsBuilder embeddings;

    dynet::ComputationGraph* _cg;

    DependencyNetwork(
            dynet::ParameterCollection& pc,
            const DependencySettings& settings,
            std::shared_ptr<dytools::Dict> token_dict,
            std::shared_ptr<dytools::Dict> char_dict,
            std::shared_ptr<dytools::Dict> tagger_dict,
            std::shared_ptr<dytools::Dict> label_dict
            );
    void new_graph(dynet::ComputationGraph& cg, bool update = true);
    void set_is_training(bool value) override;

    unsigned get_embeddings_size() const override;
    std::vector<dynet::Expression> get_embeddings(const ConllSentence &sentence) override;
};

}