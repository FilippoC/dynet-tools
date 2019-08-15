#include "dytools/builders/bilstm.h"

namespace dytools
{

unsigned int BiLSTMSettings::output_rows(const unsigned input_dim) const
{
    if (stacks == 0)
        return input_dim;
    else
        return 2 * dim;
}


BiLSTMBuilder::BiLSTMBuilder(dynet::ParameterCollection &pc, const BiLSTMSettings &settings, unsigned input_dim) :
        settings(settings),
        local_pc(pc.add_subcollection("bilstm")),
        input_dim(input_dim)
{
    if (settings.boundaries)
    {
        p_begin = local_pc.add_parameters({input_dim});
        p_end = local_pc.add_parameters({input_dim});
    }

    for (unsigned stack = 0; stack < settings.stacks; ++stack)
        builders.emplace_back(
                dynet::VanillaLSTMBuilder(
                        settings.layers,
                        (stack == 0 ? input_dim : 2 * settings.dim),
                        settings.dim,
                        local_pc),
                dynet::VanillaLSTMBuilder(
                        settings.layers,
                        (stack == 0 ? input_dim : 2 * settings.dim),
                        settings.dim,
                        local_pc
                )
        );

    std::cerr
            << "BiLSTM\n"
            << " input dim: " << input_dim << "\n"
            << " hidden dim: " << settings.dim << "\n"
            << " layers: " << settings.layers << "\n"
            << " stacks: " << settings.stacks << "\n"
            << "\n";
}

unsigned BiLSTMBuilder::output_rows() const
{
    return settings.output_rows(input_dim);
}

void BiLSTMBuilder::new_graph(dynet::ComputationGraph &cg, bool train, bool update)
{
    if (settings.boundaries)
    {
        if (update)
        {
            e_begin = dynet::parameter(cg, p_begin);
            e_end = dynet::parameter(cg, p_end);
        }
        else
        {
            e_begin = dynet::const_parameter(cg, p_begin);
            e_end = dynet::const_parameter(cg, p_end);
        }
    }
    for (unsigned stack = 0; stack < settings.stacks; ++stack)
    {
        builders.at(stack).first.new_graph(cg, update);
        builders.at(stack).second.new_graph(cg, update);

        if (train)
        {
            builders.at(stack).first.set_dropout(dropout);
            builders.at(stack).second.set_dropout(dropout);
        }
        else
        {
            builders.at(stack).first.disable_dropout();
            builders.at(stack).second.disable_dropout();
        }
    }
}

void BiLSTMBuilder::set_dropout(float value)
{
    dropout = value;
}

std::vector<dynet::Expression> BiLSTMBuilder::operator()(const std::vector<dynet::Expression>& embeddings, const bool keep_boundaries)
{
    if (keep_boundaries and settings.boundaries == false)
        throw std::runtime_error("Cannot keep boundaries has they were not set in the settings.");

    const auto e = unmerged(embeddings, keep_boundaries);

    std::vector<dynet::Expression> ret;
    ret.reserve(e.first.size());
    for (unsigned i = 0u ; i < e.first.size() ; ++i)
        ret.push_back(dynet::concatenate({e.first.at(i), e.second.at(i)}));
    return ret;
}

std::pair<std::vector<dynet::Expression>, std::vector<dynet::Expression>> BiLSTMBuilder::unmerged(const std::vector<dynet::Expression>& embeddings, const bool keep_boundaries)
{
    if (keep_boundaries and settings.boundaries == false)
        throw std::runtime_error("Cannot keep boundaries has they were not set in the settings.");

    std::vector<dynet::Expression> ret;
    ret.reserve(embeddings.size() + 2);
    if (settings.boundaries)
        ret.push_back(e_begin);
    std::copy(embeddings.begin(), embeddings.end(), std::back_inserter(ret));
    if (settings.boundaries)
        ret.push_back(e_end);

    const unsigned size = ret.size();

    std::vector<dynet::Expression> e_forward(size);
    std::vector<dynet::Expression> e_backward(size);

    for (unsigned stack = 0; stack < settings.stacks; ++stack)
    {
        builders.at(stack).first.start_new_sequence();
        builders.at(stack).second.start_new_sequence();

        // merge from previous layer
        if (stack > 0)
            for (unsigned i = 0u ; i < size ; ++i)
                ret.at(i) = dynet::concatenate({e_forward.at(i), e_backward.at(i)});

        for (unsigned i = 0u ; i < size ; ++i)
            e_forward.at(i) = builders.at(stack).first.add_input(ret.at(i));
        for (int i = size - 1 ; i >= 0 ; --i)
            e_backward.at(i) = builders.at(stack).second.add_input(ret.at(i));
    }

    // remove first and last elements
    if (settings.boundaries and !keep_boundaries)
    {
        for (unsigned i = 0 ; i < embeddings.size() ; ++ i)
        {
            e_forward.at(i) = e_forward.at(i + 1);
            e_backward.at(i) = e_backward.at(i + 1);
        }
        ret.resize(embeddings.size());
    }

    return {std::move(e_forward), std::move(e_backward)};
}

dynet::Expression BiLSTMBuilder::endpoints(const std::vector<dynet::Expression> &embeddings)
{
    std::vector<dynet::Expression> fixed_embs;
    fixed_embs.reserve(embeddings.size() + 2);
    if (settings.boundaries)
        fixed_embs.push_back(e_begin);
    std::copy(embeddings.begin(), embeddings.end(), std::back_inserter(fixed_embs));
    if (settings.boundaries)
        fixed_embs.push_back(e_end);

    const unsigned size = fixed_embs.size();
    if (builders.size() != 1u)
        throw std::runtime_error("Endpoints can be used only if the number of stacks=1");

    builders.at(0u).first.start_new_sequence();
    builders.at(0u).second.start_new_sequence();
    dynet::Expression f, b;

    for (unsigned i = 0u ; i < size ; ++i)
        f = builders.at(0u).first.add_input(fixed_embs.at(i));
    for (int i = size - 1 ; i >= 0 ; --i)
        b = builders.at(0u).second.add_input(fixed_embs.at(i));

    return dynet::concatenate({f, b});
}


}