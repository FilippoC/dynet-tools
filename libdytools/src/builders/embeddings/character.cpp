#include "dytools/builders/embeddings/character.h"


namespace dytools
{

CharacterEmbeddingsBuilder::CharacterEmbeddingsBuilder(dynet::ParameterCollection& pc, const CharacterEmbeddingsSettings& settings, std::shared_ptr<dytools::Dict> dict) :
        settings(settings),
        local_pc(pc.add_subcollection("embschar")),
        dict(dict),
        bilstm(local_pc, settings.bilstm, settings.dim)
{
    lp = pc.add_lookup_parameters(dict->size(), {settings.dim});

    std::cerr
        << "Character embeddings\n"
        << " dim: " << settings.dim << "\n"
        << " vocabulary size: " << dict->size() << "\n"
        << "\n"
        ;
}

void CharacterEmbeddingsBuilder::new_graph(dynet::ComputationGraph& cg, bool update)
{
    _cg = &cg;
    _update = update;

    bilstm.new_graph(cg, update);
}

void CharacterEmbeddingsBuilder::set_is_training(bool value)
{
    Builder::set_is_training(value);
    bilstm.set_is_training(value);
}

dynet::Expression CharacterEmbeddingsBuilder::get_char_embedding(const std::string& c)
{
    if (_update)
        return dynet::lookup(*_cg, lp, dict->convert(c));
    else
        return dynet::const_lookup(*_cg, lp, dict->convert(c));
}

dynet::Expression CharacterEmbeddingsBuilder::get(const std::string& str)
{
    std::vector<dynet::Expression> input;
    for (unsigned i = 0u; i < str.size(); ++i)
    {
        const auto c = std::to_string(str[i]);
        input.push_back(get_char_embedding(c));
    }

    auto output = bilstm(input);
    return dynet::concatenate({output.at(0u), output.back()});
}

std::vector<dynet::Expression> CharacterEmbeddingsBuilder::get_all_as_vector(const std::vector<std::string>& words)
{
    std::vector<dynet::Expression> ret;
    for (auto const& str : words)
        ret.push_back(get(str));
    return ret;
}

unsigned CharacterEmbeddingsBuilder::output_rows() const
{
    return 2 * bilstm.output_rows();
}

}