#include "dytools/builders/embeddings/word.h"

namespace dytools
{

WordEmbeddingsBuilder::WordEmbeddingsBuilder(dynet::ParameterCollection& pc, const WordEmbeddingsSettings& settings, const std::string& name) :
        settings(settings),
        local_pc(pc.add_subcollection("name"))
{}

WordEmbeddingsBuilder::WordEmbeddingsBuilder(dynet::ParameterCollection& pc, const WordEmbeddingsSettings& settings, std::shared_ptr<dytools::Dict> dict) :
    settings(settings),
    local_pc(pc.add_subcollection("embstoken")),
    dict(dict)
{
    lp = pc.add_lookup_parameters(dict->size(), {settings.dim});

        std::cerr
        << "Token embeddings\n"
        << " dim: " << settings.dim << "\n"
        << " vocabulary size: " << dict->size() << "\n"
        << "\n"
        ;
}

void WordEmbeddingsBuilder::new_graph(dynet:: ComputationGraph& cg, bool update)
{
    _cg = &cg;
    _update = update;
}

dynet::Expression WordEmbeddingsBuilder::get(const std::string& str)
{
   return get((unsigned) dict->convert(str));
}

dynet::Expression WordEmbeddingsBuilder::get(const unsigned idx)
{
    if (_update)
        return dynet::lookup(*_cg, lp, idx);
    else
        return dynet::const_lookup(*_cg, lp, idx);
}


dynet::Expression WordEmbeddingsBuilder::get(unsigned* idx)
{
    if (_update)
        return dynet::lookup(*_cg, lp, idx);
    else
        return dynet::const_lookup(*_cg, lp, idx);
}

dynet::Expression WordEmbeddingsBuilder::get_all_as_expr(std::vector<unsigned>* indices)
{
    if (_update)
        return dynet::lookup(*_cg, lp, indices);
    else
        return dynet::const_lookup(*_cg, lp, indices);
}

dynet::Expression WordEmbeddingsBuilder::get_all_as_expr(const std::vector<unsigned> indices)
{
    if (_update)
        return dynet::lookup(*_cg, lp, indices);
    else
        return dynet::const_lookup(*_cg, lp, indices);
}

std::vector<dynet::Expression> WordEmbeddingsBuilder::get_all_as_vector(const std::vector<unsigned> indices)
{
    std::vector<dynet::Expression> ret;
    for (const auto u : indices)
        ret.push_back(get(u));
    return ret;
}


dynet::Expression WordEmbeddingsBuilder::get_all_as_expr(const std::vector<std::string> strings)
{
    std::vector<unsigned> indices;
    for (const auto& s : strings)
        indices.push_back(dict->convert(s));
    return get_all_as_expr(indices);
}

std::vector<dynet::Expression> WordEmbeddingsBuilder::get_all_as_vector(const std::vector<std::string> strings)
{
    std::vector<dynet::Expression> ret;
    for (const auto s : strings)
        ret.push_back(get(s));
    return ret;
}


unsigned WordEmbeddingsBuilder::output_rows() const
{
    return settings.dim;
}

}