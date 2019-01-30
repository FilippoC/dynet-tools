#include "dytools/builders/char_embeddings.h"


namespace dytools
{

CharEmbeddingsBuilder::CharEmbeddingsBuilder(dynet::ParameterCollection& pc, const CharEmbeddingsSettings& settings, std::shared_ptr<dynet::Dict> dict) :
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

void CharEmbeddingsBuilder::new_graph(dynet::ComputationGraph& cg, bool update)
{
    _cg = &cg;
    _update = update;

    bilstm.new_graph(cg, update);
}

void CharEmbeddingsBuilder::set_is_training(bool value)
{
    Builder::set_is_training(value);
    bilstm.set_is_training(value);
}

dynet::Expression CharEmbeddingsBuilder::operator()(const std::string& c)
{
    if (_update)
        return dynet::lookup(*_cg, lp, dict->convert(c));
    else
        return dynet::const_lookup(*_cg, lp, dict->convert(c));
}

dynet::Expression CharEmbeddingsBuilder::operator()(const ConllToken& token)
{
    std::vector<dynet::Expression> input;
    for (unsigned i = 0u; i < token.word.size(); ++i)
    {
        const auto c = std::to_string(token.word[i]);
        input.push_back((*this)(c));
    }

    auto output = bilstm(input);
    return dynet::concatenate({output.at(0u), output.back()});
}

std::vector<dynet::Expression> CharEmbeddingsBuilder::operator()(const ConllSentence& sentence)
{
    std::vector<dynet::Expression> embs;
    for (auto const& token : sentence)
        embs.push_back((*this)(token));
    return embs;
}

unsigned CharEmbeddingsBuilder::output_rows() const
{
    return 2 * bilstm.output_rows();
}

}