#include "dytools/builders/embeddings/character.h"


namespace dytools
{


unsigned int CharacterEmbeddingsSettings::output_rows() const
{
    return 2 * bilstm.output_rows(dim);
}

CharacterEmbeddingsBuilder::CharacterEmbeddingsBuilder(dynet::ParameterCollection& pc, const CharacterEmbeddingsSettings& settings, const unsigned n_char) :
        settings(settings),
        local_pc(pc.add_subcollection("embschar")),
        bilstm(local_pc, settings.bilstm, settings.dim)
{
    lp = pc.add_lookup_parameters(n_char, {settings.dim});

    std::cerr
        << "Character embeddings\n"
        << " dim: " << settings.dim << "\n"
        << " vocabulary size: " << n_char << "\n"
        << "\n"
        ;
}

void CharacterEmbeddingsBuilder::new_graph(dynet::ComputationGraph& cg, bool training, bool update)
{
    _cg = &cg;
    _update = update;
    _is_training = training;

    bilstm.new_graph(cg, training, update);
}

dynet::Expression CharacterEmbeddingsBuilder::get(const unsigned c)
{
    if (_update)
        return dynet::lookup(*_cg, lp, c);
    else
        return dynet::const_lookup(*_cg, lp, c);
}

dynet::Expression CharacterEmbeddingsBuilder::get(const std::vector<unsigned>& str)
{
    std::vector<dynet::Expression> input;
    for (unsigned i = 0u; i < str.size(); ++i)
    {
        const unsigned c = str.at(i);
        input.push_back(get(c));
    }

    auto output = bilstm(input);
    return dynet::concatenate({output.at(0u), output.back()});
}

std::vector<dynet::Expression> CharacterEmbeddingsBuilder::get_all_as_vector(const std::vector<std::vector<unsigned>>& words)
{
    std::vector<dynet::Expression> ret;
    for (const std::vector<unsigned>& str : words)
        ret.push_back(get(str));
    return ret;
}

unsigned CharacterEmbeddingsBuilder::output_rows() const
{
    return settings.output_rows();
}

}