#pragma once

#include <memory>
#include "dynet/expr.h"

#include "dytools/builders/embeddings/character.h"
#include "dytools/builders/embeddings/word.h"

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

    unsigned output_rows() const;
};

struct EmbeddingsBuilder
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
            const unsigned n_token,
            const unsigned n_char
    );

    void new_graph(dynet::ComputationGraph& cg, bool training, bool update);
    std::vector<dynet::Expression> operator()(const std::vector<unsigned>& tokens);
    std::vector<dynet::Expression> operator()(const std::vector<std::vector<unsigned>>& v_chars);
    std::vector<dynet::Expression> operator()(const std::vector<unsigned>& tokens, const std::vector<std::vector<unsigned>>& chars);

    unsigned output_rows() const;
};

}