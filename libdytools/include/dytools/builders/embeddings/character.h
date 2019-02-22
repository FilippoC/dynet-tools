#pragma once

#include <vector>
#include <string>

#include "dynet/expr.h"

#include "dytools/dict.h"
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
    std::shared_ptr<dytools::Dict> dict;

    dynet::LookupParameter lp;
    BiLSTMBuilder bilstm;

    dynet::ComputationGraph* _cg;
    bool _update = true;
    bool _is_training = true;

    CharacterEmbeddingsBuilder(dynet::ParameterCollection& pc, const CharacterEmbeddingsSettings& settings, std::shared_ptr<dytools::Dict> dict);

    void new_graph(dynet::ComputationGraph& cg, bool training, bool update);

    dynet::Expression get_char_embedding(const std::string& c);

    dynet::Expression get(const std::string& word);
    template<class T, class O> dynet::Expression get(const O& obj);


    std::vector<dynet::Expression> get_all_as_vector(const std::vector<std::string>& words);
    template<class T, class It> std::vector<dynet::Expression> get_all_as_vector(It begin, It end);

    unsigned output_rows() const;
};


template<class T, class O>
dynet::Expression CharacterEmbeddingsBuilder::get(const O& obj)
{
    return get(T()(obj));
}

template<class T, class It>
std::vector<dynet::Expression> CharacterEmbeddingsBuilder::get_all_as_vector(It begin, It end)
{
    std::vector<dynet::Expression> ret;
    for (; begin != end; ++begin)
        ret.push_back(get<T>(*begin));
    return ret;
}


}