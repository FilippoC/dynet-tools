#pragma once

#include "dynet/expr.h"

namespace dytools
{

struct WordEmbeddingsSettings
{
    unsigned dim = 100;

    template<class Archive>
    void serialize(Archive& ar, const unsigned int)
    {
        ar & dim;
    }

    unsigned int output_rows() const;
};

struct WordEmbeddingsBuilder
{
    const WordEmbeddingsSettings settings;
    const unsigned size;
    dynet::ParameterCollection local_pc;

    dynet::LookupParameter lp;
    dynet::ComputationGraph* _cg;
    bool _update = true;
    bool _is_training = true;

    WordEmbeddingsBuilder(dynet::ParameterCollection& pc, const WordEmbeddingsSettings& settings, const unsigned _size);

    void new_graph(dynet::ComputationGraph& cg, bool training, bool update);

    dynet::Expression get(const unsigned idx);

    dynet::Expression get(unsigned* idx);
    dynet::Expression get_all_as_expr(std::vector<unsigned>* indices);

    dynet::Expression get_all_as_expr(const std::vector<unsigned> indices);
    std::vector<dynet::Expression> get_all_as_vector(const std::vector<unsigned> indices);

    // templatized accessors
    template <class T, class It> std::vector<dynet::Expression> get_all_as_vector(It begin, It end);
    template <class T, class It> dynet::Expression get_all_as_expr(It begin, It end);

    unsigned output_rows() const;

protected:
    WordEmbeddingsBuilder(dynet::ParameterCollection& pc, const WordEmbeddingsSettings& settings, const std::string& name);
};



// templatized methods implementation

template <class T, class It>
std::vector<dynet::Expression> WordEmbeddingsBuilder::get_all_as_vector(It begin, It end)
{
    std::vector<dynet::Expression> ret;
    for (; begin != end ; ++begin)
        ret.push_back(get(*begin));
    return ret;
}

template <class T, class It>
dynet::Expression WordEmbeddingsBuilder::get_all_as_expr(It begin, It end)
{
    std::vector<unsigned> indices;
    for (; begin != end ; ++begin)
        indices.push_back(*begin);
    return get_all_as_expr(indices);
}


}