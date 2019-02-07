#pragma once

#include <memory>
#include "dynet/expr.h"

#include "dytools/data/conll.h"
#include "dytools/builders/embeddings/character.h"
#include "dytools/builders/embeddings/word.h"
#include "dytools/builders/builder.h"

namespace dytools
{

struct EmbeddingsSettings
{
    bool use_token_embeddings = true;
    bool use_char_embeddings = true;

    WordEmbeddingsSettings token_embeddings;
    CharacterEmbeddingsSettings char_embeddings;

    template<class Archive>
    void serialize(Archive& ar, const unsigned int)
    {
        ar & use_token_embeddings;
        ar & use_char_embeddings;

        ar & token_embeddings;
        ar & char_embeddings;
    }
};

struct EmbeddingsBuilder : public Builder
{
    const EmbeddingsSettings settings;
    dynet::ParameterCollection local_pc;

    std::unique_ptr<WordEmbeddingsBuilder> token_embeddings;
    std::unique_ptr<CharacterEmbeddingsBuilder> char_embeddings;

    dynet::Parameter p_root_prefix;
    dynet::Expression e_root_prefix;

    // word + char embeddings
    EmbeddingsBuilder(
            dynet::ParameterCollection& pc,
            const EmbeddingsSettings& settings,
            std::shared_ptr<dytools::Dict> token_dict,
            std::shared_ptr<dytools::Dict> char_dict
    );

    void new_graph(dynet::ComputationGraph& cg, bool update=true);
    void set_is_training(bool value) override;
    std::vector<dynet::Expression> operator()(const ConllSentence& sentence);

    unsigned output_rows() const;
};

}