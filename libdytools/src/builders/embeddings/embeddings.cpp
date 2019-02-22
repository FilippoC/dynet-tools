#include "dytools/builders/embeddings/embeddings.h"

#include <stdexcept>

namespace dytools
{

unsigned EmbeddingsSettings::output_rows() const
{
    std::cerr
            << "use_token_embeddings: " << (use_token_embeddings ? "yes" : "no") << "\n"
            << "use_char_embeddings: " << (use_char_embeddings ? "yes" : "no") << "\n"
            << "token dim:" << (use_token_embeddings ? token_embeddings.output_rows() : 0u) << "\n"
            << "char dim: " << (use_char_embeddings ? char_embeddings.output_rows() : 0u) << "\n";
    return
            (use_token_embeddings ? token_embeddings.output_rows() : 0u)
            +
            (use_char_embeddings ? char_embeddings.output_rows() : 0u);
}

EmbeddingsBuilder::EmbeddingsBuilder(dynet::ParameterCollection& pc, const EmbeddingsSettings& settings, std::shared_ptr<dytools::Dict> token_dict, std::shared_ptr<dytools::Dict> char_dict) :
    settings(settings),
    local_pc(pc.add_subcollection("embeddings"))
{
    if (!settings.use_char_embeddings and !settings.use_token_embeddings)
        throw std::runtime_error("Embeddings: you should use at least one of them.");

    if (settings.use_token_embeddings)
        token_embeddings.reset(new WordEmbeddingsBuilder(local_pc, settings.token_embeddings, token_dict));
    if (settings.use_char_embeddings)
        char_embeddings.reset(new CharacterEmbeddingsBuilder(local_pc, settings.char_embeddings, char_dict));

    std::cerr
        << "Embeddings\n"
        << " use token embeddings: " << (settings.use_token_embeddings ? "yes" : "no") << "\n"
        << " use char embeddings: " << (settings.use_char_embeddings ? "yes" : "no") << "\n"
        << "\n"
        ;
}

unsigned EmbeddingsBuilder::output_rows() const
{
    return settings.output_rows();
}

void EmbeddingsBuilder::new_graph(dynet::ComputationGraph& cg, bool training, bool update)
{
    if (settings.use_token_embeddings)
        token_embeddings->new_graph(cg, training, update);
    if (settings.use_char_embeddings)
        char_embeddings->new_graph(cg, training, update);
}

std::vector<dynet::Expression> EmbeddingsBuilder::operator()(const ConllSentence& sentence)
{
    if (settings.use_token_embeddings && settings.use_char_embeddings)
    {
        auto tokens = token_embeddings->get_all_as_vector<ConllWordGetter>(sentence.begin(), sentence.end());
        auto chars = char_embeddings->get_all_as_vector<ConllWordGetter>(sentence.begin(), sentence.end());

        std::vector<dynet::Expression> ret;
        for (unsigned i = 0 ; i < sentence.size() ; ++i)
        {
            ret.push_back(dynet::concatenate({tokens.at(i), chars.at(i)}));
        }
        return ret;
    }
    else if (settings.use_token_embeddings)
    {
        auto ret = token_embeddings->get_all_as_vector<ConllWordGetter>(sentence.begin(), sentence.end());
        return ret;
    }
    else
    {
        auto ret = char_embeddings->get_all_as_vector<ConllWordGetter>(sentence.begin(), sentence.end());
        return ret;
    }
}

}