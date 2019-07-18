#pragma once

#include <vector>
#include <string>

#include "dynet/expr.h"
#include "dytools/builders/bilstm.h"

namespace dytools
{

struct CharacterEmbeddingsSettings
{
    unsigned dim = 100;
    BiLSTMSettings bilstm;

    template<class Archive>
    void serialize(Archive& ar, const unsigned int)
    {
        ar & dim;
        ar & bilstm;
    }

    unsigned int output_rows() const;
};

struct CharacterEmbeddingsBuilder
{
    const CharacterEmbeddingsSettings settings;
    dynet::ParameterCollection local_pc;

    dynet::LookupParameter lp;
    BiLSTMBuilder bilstm;

    float input_dropout = 0.f;

    dynet::ComputationGraph* _cg;
    bool _update = true;
    bool _is_training = true;

    CharacterEmbeddingsBuilder(dynet::ParameterCollection& pc, const CharacterEmbeddingsSettings& settings, const unsigned n_char);

    void new_graph(dynet::ComputationGraph& cg, bool training, bool update);
    void set_dropout(float input);

    dynet::Expression get(const unsigned c);
    dynet::Expression get(const std::vector<unsigned>& word);
    std::vector<dynet::Expression> get_all_as_vector(const std::vector<std::vector<unsigned>>& words);

    unsigned output_rows() const;
};



}