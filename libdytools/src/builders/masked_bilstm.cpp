#include "dytools/builders/masked_bilstm.h"

namespace dytools
{


MaskedBiLSTMBuilder::MaskedBiLSTMBuilder(dynet::ParameterCollection& pc, const MaskedBiLSTMSettings& t_settings, unsigned t_input_dim) :
        settings(t_settings),
        local_pc(pc.add_subcollection("maskedbilstm")),
        input_dim(t_input_dim)
{
    for (unsigned i = 0 ; i < settings.n_stack ; ++ i)
    {
        if (i == 0)
        {
            forward.emplace_back(local_pc, settings.lstm, input_dim);
            backward.emplace_back(local_pc, settings.lstm, input_dim);
        }
        else
        {
            forward.emplace_back(local_pc, settings.lstm, dim_output());
            backward.emplace_back(local_pc, settings.lstm, dim_output());
        }
    }

    if (settings.padding)
    {
        pad = local_pc.add_lookup_parameters(2, {input_dim});
    }
}

unsigned MaskedBiLSTMBuilder::dim_output()
{
    if (settings.n_stack == 0)
        return input_dim;
    else
        return settings.lstm.hidden_dim * 2;
}

void MaskedBiLSTMBuilder::new_graph(dynet::ComputationGraph &cg, bool training, bool update)
{
    for (unsigned stack = 0 ; stack < settings.n_stack ; ++stack)
    {
        forward.at(stack).new_graph(cg, training, update);
        backward.at(stack).new_graph(cg, training, update);
    }

    if (settings.padding)
    {
        e_begin = (update ? dynet::lookup(cg, pad, pad_begin) : dynet::const_lookup(cg, pad, pad_begin));
        e_end = (update ? dynet::lookup(cg, pad, pad_end) : dynet::const_lookup(cg, pad, pad_end));
    }
}

std::vector<std::pair<dynet::Expression, dynet::Expression>> MaskedBiLSTMBuilder::compute(const std::vector<dynet::Expression>& input, const dynet::Expression* mask)
{
    if (settings.n_stack == 0u)
        throw std::runtime_error("This function cannot be called if n_stack==0u");

    auto const padding = settings.padding;

    const unsigned size = input.size() + (padding ? 2u : 0u);
    std::vector<std::pair<dynet::Expression, dynet::Expression>> ret;
    ret.reserve(input.size());

    std::vector<dynet::Expression> last(size);
    std::vector<dynet::Expression> lstm_forward(size);
    std::vector<dynet::Expression> lstm_backward(size);

    unsigned i = 0u;
    if (padding)
    {
        i = 1u;
        last.at(0u) = dynet::concatenate_to_batch(std::vector<dynet::Expression>(
                input.back().dim().bd,
                e_begin
        ));
    }
    std::for_each(
            input.begin(), input.end(),
            [&] (const dynet::Expression& e)
            {
                last.at(i) = e;
                ++i;
            }
    );
    if (padding)
        last.at(i) = dynet::concatenate_to_batch(std::vector<dynet::Expression>(
                input.back().dim().bd,
                e_end
        ));

    for (unsigned stack = 0 ; stack < settings.n_stack ; ++stack)
    {
        // Forward
        auto fstate = forward.at(stack).new_state();

        // if padding, do not mask first and last
        if (mask != nullptr)
            for (unsigned i = 0u ; i < size ; ++i)
            {
                if (padding && (i == 0 || i + 1 == size))
                    lstm_forward.at(i) = forward.at(stack).add_input(fstate, last.at(i));
                else
                {
                    auto m = dynet::pick(*mask, (padding ? i - 1 : i), 1u);
                    lstm_forward.at(i) = forward.at(stack).add_input(fstate, last.at(i), &m);
                }
            }
        else
            for (unsigned i = 0u ; i < size ; ++i)
                lstm_forward.at(i) = forward.at(stack).add_input(fstate, last.at(i));

        // backward
        auto bstate = backward.at(stack).new_state();

        if (mask != nullptr)
            for (int i = size - 1 ; i >= 0 ; --i)
            {
                if (padding && (i == 0 || i == (int) size - 1))
                    lstm_backward.at(i) = backward.at(stack).add_input(bstate, last.at(i));
                else
                {
                    auto m = dynet::pick(*mask, (padding ? i - 1 : i), 1u);
                    lstm_backward.at(i) = backward.at(stack).add_input(bstate, last.at(i), &m);
                }
            }
        else
            for (int i = size - 1 ; i >= 0 ; --i)
                lstm_backward.at(i) = backward.at(stack).add_input(bstate, last.at(i));

        // concatenate
        if (stack == settings.n_stack - 1u)
        {
            for (
                    unsigned i = (settings.padding ? 1u : 0u) ;
                    i < (settings.padding ? size - 1u : size) ;
                    ++i
                    )
            {
                ret.push_back(std::make_pair(
                        lstm_forward.at(i),
                        lstm_backward.at(i)
                ));
            }
        }
        else
        {
            // concatenate both lstms output
            for (unsigned i = 0 ; i < size ; ++i)
            {
                last.at(i) =
                        dynet::concatenate({
                                                   lstm_forward.at(i),
                                                   lstm_backward.at(i)
                                           })
                        ;
            }
        }
    }
    return ret;
}

std::vector<dynet::Expression> MaskedBiLSTMBuilder::operator()(const std::vector<dynet::Expression>& input, const dynet::Expression* mask)
{
    std::vector<dynet::Expression> ret;
    if (settings.n_stack == 0u)
    {
        for (auto const& e : input)
            ret.emplace_back(e);
    }
    else
    {
        auto pairs = compute(input, mask);
        for (unsigned i = 0u ; i < pairs.size() ; ++i)
            ret.push_back(dynet::concatenate({pairs.at(i).first, pairs.at(i).second}));
    }
    return ret;
}

dynet::Expression MaskedBiLSTMBuilder::operator()(const dynet::Expression& input, const dynet::Expression* mask)
{
    if (settings.n_stack == 0u)
        return input;

    const unsigned ncols = input.dim().cols();
    std::vector<dynet::Expression> values;
    values.reserve(ncols);
    for (unsigned i = 0u ; i < ncols ; ++i)
        values.emplace_back(dynet::pick(input, i, 1));

    auto pairs = compute(values, mask);

    std::vector<dynet::Expression> ret;
    ret.reserve(ncols);
    for (unsigned i = 0u ; i < pairs.size() ; ++i)
        ret.push_back(dynet::concatenate({pairs.at(i).first, pairs.at(i).second}));
    return dynet::concatenate(ret, 1);
}

}