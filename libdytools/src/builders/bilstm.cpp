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
            builders.at(stack).first.set_dropout(0.f);
            builders.at(stack).second.set_dropout(0.f);
        }
    }
}

void BiLSTMBuilder::set_dropout(float value)
{
    dropout = value;
}

std::vector<dynet::Expression> BiLSTMBuilder::operator()(const std::vector<dynet::Expression>& embeddings)
{
    const unsigned size = embeddings.size();
    std::vector<dynet::Expression> ret(embeddings);

    std::vector<dynet::Expression> e_forward(embeddings.size());
    std::vector<dynet::Expression> e_backward(embeddings.size());

    for (unsigned stack = 0; stack < settings.stacks; ++stack)
    {
        builders.at(stack).first.start_new_sequence();
        builders.at(stack).second.start_new_sequence();

        for (unsigned i = 0u ; i < size ; ++i)
            e_forward.at(i) = builders.at(stack).first.add_input(ret.at(i));
        for (int i = size - 1 ; i >= 0 ; --i)
            e_backward.at(i) = builders.at(stack).second.add_input(ret.at(i));
        for (unsigned i = 0u ; i < size ; ++i)
            ret.at(i) = dynet::concatenate({e_forward.at(i), e_backward.at(i)});
    }
    return ret;
}

dynet::Expression BiLSTMBuilder::endpoints(const std::vector<dynet::Expression> &embeddings)
{
    const unsigned size = embeddings.size();
    if (builders.size() != 1u)
        throw std::runtime_error("Endpoints can be used only if the number of stacks=1");

    builders.at(0u).first.start_new_sequence();
    builders.at(0u).second.start_new_sequence();
    dynet::Expression f, b;

    for (unsigned i = 0u ; i < size ; ++i)
        f = builders.at(0u).first.add_input(embeddings.at(i));
    for (int i = size - 1 ; i >= 0 ; --i)
        b = builders.at(0u).second.add_input(embeddings.at(i));

    return dynet::concatenate({f, b});
}


}