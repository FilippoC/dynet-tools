#pragma once

#include "dynet/expr.h"
#include "dytools/builders/builder.h"
#include "dytools/dict.h"

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

struct WordEmbeddingsBuilder : public Builder
{
    const WordEmbeddingsSettings settings;
    dynet::ParameterCollection local_pc;
    std::shared_ptr<dytools::Dict> dict;

    dynet::LookupParameter lp;
    dynet::ComputationGraph* _cg;
    bool _update = true;

    WordEmbeddingsBuilder(dynet::ParameterCollection& pc, const WordEmbeddingsSettings& settings, std::shared_ptr<dytools::Dict> dict);

    void new_graph(dynet::ComputationGraph& cg, bool update=true);

    dynet::Expression get(const std::string& str);
    dynet::Expression get(const unsigned idx);

    dynet::Expression get(unsigned* idx);
    dynet::Expression get_all_as_expr(std::vector<unsigned>* indices);

    dynet::Expression get_all_as_expr(const std::vector<unsigned> indices);
    std::vector<dynet::Expression> get_all_as_vector(const std::vector<unsigned> indices);

    dynet::Expression get_all_as_expr(const std::vector<std::string> strings);
    std::vector<dynet::Expression> get_all_as_vector(const std::vector<std::string> strings);

    // templatized accessors
    template <class T> dynet::Expression get(const T& token);
    template <class T, class It> std::vector<dynet::Expression> get_all_as_vector(It begin, It end);
    template <class T, class It> dynet::Expression get_all_as_expr(It begin, It end);

    unsigned output_rows() const;

protected:
    WordEmbeddingsBuilder(dynet::ParameterCollection& pc, const WordEmbeddingsSettings& settings, const std::string& name);
};



// templatized methods implementation

template <class T>
dynet::Expression WordEmbeddingsBuilder::get(const T& token)
{
    const std::string& str = T()(token);
    return get(str);
}

template <class T, class It>
std::vector<dynet::Expression> WordEmbeddingsBuilder::get_all_as_vector(It begin, It end)
{
    std::vector<dynet::Expression> ret;
    for (; begin != end ; ++begin)
    {
        const std::string& str = T()(*begin);
        ret.push_back(get(str));
    }
    return ret;
}

template <class T, class It>
dynet::Expression WordEmbeddingsBuilder::get_all_as_expr(It begin, It end)
{
    std::vector<unsigned> indices;
    for (; begin != end ; ++begin)
    {
        const std::string& str = T()(*begin);
        indices.push_back(dict->convert(str));
    }
    return get_all_as_expr(indices);
}


}