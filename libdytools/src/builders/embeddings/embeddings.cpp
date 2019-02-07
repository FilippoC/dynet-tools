#include "dytools/builders/embeddings/embeddings.h"

#include <stdexcept>

namespace dytools
{

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
    return
            (settings.use_token_embeddings ? token_embeddings->output_rows() : 0u)
            +
            (settings.use_char_embeddings ? char_embeddings->output_rows() : 0u)
            ;
}

void EmbeddingsBuilder::new_graph(dynet::ComputationGraph& cg, bool update)
{
    if (settings.use_token_embeddings)
        token_embeddings->new_graph(cg, update);
    if (settings.use_char_embeddings)
        char_embeddings->new_graph(cg, update);
}

void EmbeddingsBuilder::set_is_training(bool value)
{
    Builder::set_is_training(value);
    if (settings.use_token_embeddings)
        token_embeddings->set_is_training(value);
    if (settings.use_char_embeddings)
        char_embeddings->set_is_training(value);
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