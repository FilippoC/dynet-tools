#include "dytools/builders/embeddings/word.h"

namespace dytools
{


unsigned int WordEmbeddingsSettings::output_rows() const
{
    return dim;
}


WordEmbeddingsBuilder::WordEmbeddingsBuilder(dynet::ParameterCollection& pc, const WordEmbeddingsSettings& settings, const unsigned _size) :
    settings(settings),
    size(_size),
    local_pc(pc.add_subcollection("embstoken"))
{
    lp = pc.add_lookup_parameters(size, {settings.dim});

    std::cerr
        << "Token embeddings\n"
        << " dim: " << settings.dim << "\n"
        << " vocabulary size: " << size << "\n"
        << "\n"
    ;
}

void WordEmbeddingsBuilder::new_graph(dynet:: ComputationGraph& cg, bool training, bool update)
{
    _cg = &cg;
    _update = update;
    _is_training = training;
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


unsigned WordEmbeddingsBuilder::output_rows() const
{
    return settings.output_rows();
}

}