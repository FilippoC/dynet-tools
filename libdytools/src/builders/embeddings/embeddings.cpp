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

EmbeddingsBuilder::EmbeddingsBuilder(
        dynet::ParameterCollection& pc,
        const EmbeddingsSettings& settings,
        const unsigned n_token,
        const unsigned n_char
        ) :
    settings(settings),
    local_pc(pc.add_subcollection("embeddings"))
{
    if (!settings.use_char_embeddings and !settings.use_token_embeddings)
        throw std::runtime_error("Embeddings: you should use at least one of them.");

    if (settings.use_token_embeddings)
        token_embeddings.reset(new WordEmbeddingsBuilder(local_pc, settings.token_embeddings, n_token));
    if (settings.use_char_embeddings)
        char_embeddings.reset(new CharacterEmbeddingsBuilder(local_pc, settings.char_embeddings, n_char));

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

std::vector<dynet::Expression> EmbeddingsBuilder::operator()(const std::vector<unsigned>& v_tokens)
{
    if (settings.use_char_embeddings)
        throw std::runtime_error("Characters are mandatory");
    if (!settings.use_token_embeddings)
        throw std::runtime_error("Not token embeddings");

    return token_embeddings->get_all_as_vector(v_tokens);
}

std::vector<dynet::Expression> EmbeddingsBuilder::operator()(const std::vector<std::vector<unsigned>>& v_chars)
{
    if (!settings.use_char_embeddings)
        throw std::runtime_error("Not char embeddings");
    if (settings.use_token_embeddings)
        throw std::runtime_error("Tokens are mandatory");

    return char_embeddings->get_all_as_vector(v_chars);
}


std::vector<dynet::Expression> EmbeddingsBuilder::operator()(
        const std::vector<unsigned>& v_tokens,
        const std::vector<std::vector<unsigned>>& v_chars
)
{
    if (settings.use_token_embeddings && settings.use_char_embeddings)
    {
        auto tokens = token_embeddings->get_all_as_vector(v_tokens);
        auto chars = char_embeddings->get_all_as_vector(v_chars);

        std::vector<dynet::Expression> ret;
        for (unsigned i = 0 ; i < v_tokens.size() ; ++i)
            ret.push_back(dynet::concatenate({tokens.at(i), chars.at(i)}));
        return ret;
    }
    else if (settings.use_token_embeddings)
    {
        auto ret = token_embeddings->get_all_as_vector(v_tokens);
        return ret;
    }
    else
    {
        auto ret = char_embeddings->get_all_as_vector(v_chars);
        return ret;
    }
}

}