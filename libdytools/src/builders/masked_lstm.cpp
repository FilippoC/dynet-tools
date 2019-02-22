#include "dytools/builders/masked_lstm.h"

#include <stdexcept>

namespace dytools
{

MaskedLSTMBuilder::MaskedLSTMBuilder(dynet::ParameterCollection &pc, const MaskedLSTMSettings &t_settings,
        unsigned dim_input) :
        settings(t_settings),
        local_pc(pc.add_subcollection("maskedlstm"))
{
    if (settings.layers == 0)
        throw std::runtime_error("Masked LSTM must be at list of size 1.");

    unsigned layer_input_dim = dim_input;
    for (unsigned i = 0; i < settings.layers; ++i)
    {
        auto p_Wx = local_pc.add_parameters({settings.hidden_dim * 4, layer_input_dim});
        auto p_Wh = local_pc.add_parameters({settings.hidden_dim * 4, settings.hidden_dim});
        auto p_b = local_pc.add_parameters({settings.hidden_dim * 4}, dynet::ParameterInitConst(0.f));

        layer_input_dim = settings.hidden_dim;  // output (hidden) from 1st layer is input to next

        // use emplace back here?
        std::vector<dynet::Parameter> ps = {p_Wx, p_Wh, p_b};
        p_params.push_back(ps);

        if (settings.learn_init_state)
        {
            p_init_state_c.push_back(local_pc.add_parameters({settings.hidden_dim}));
            p_init_state_h.push_back(local_pc.add_parameters({settings.hidden_dim}));
        }
    }
}

unsigned MaskedLSTMBuilder::dim_output()
{
    return settings.hidden_dim;
}

void MaskedLSTMBuilder::new_graph(dynet::ComputationGraph &cg, bool, bool update)
{
    e_params.clear();
    e_init_state_c.clear();
    e_init_state_h.clear();
    for (unsigned i = 0; i < settings.layers; ++i)
    {
        std::vector<dynet::Expression> vars;
        for (unsigned j = 0; j < p_params.at(i).size(); ++j)
        {
            vars.push_back(update ? parameter(cg, p_params.at(i).at(j)) : const_parameter(cg, p_params.at(i).at(j)));
        }
        e_params.emplace_back(vars);
    }

    for(auto p : p_init_state_c)
        e_init_state_c.push_back(update ? parameter(cg, p) : const_parameter(cg, p));
    for(auto p : p_init_state_h)
        e_init_state_h.push_back(update ? parameter(cg, p) : const_parameter(cg, p));

    e_init_zeros = dynet::zeros(cg, {settings.hidden_dim});
}

MaskedLSTMState::MaskedLSTMState(MaskedLSTMBuilder& builder)
{
    c.emplace_back(builder.settings.layers);
    h.emplace_back(builder.settings.layers);
    if (builder.settings.learn_init_state)
    {
        for (unsigned i = 0 ; i < builder.settings.layers ; ++i)
        {
            c.at(0).at(i) = builder.e_init_state_c.at(i);
            h.at(0).at(i) = builder.e_init_state_h.at(i);
        }
    }
    else
    {
        for (unsigned i = 0 ; i < builder.settings.layers ; ++i)
        {
            c.at(0).at(i) = builder.e_init_zeros;
            h.at(0).at(i) = builder.e_init_zeros;
        }
    }
}

MaskedLSTMState MaskedLSTMBuilder::new_state()
{
    MaskedLSTMState state(*this);
    return state;
}

dynet::Expression MaskedLSTMBuilder::add_input(MaskedLSTMState& state, const dynet::Expression& input, const dynet::Expression* mask)
{
    // if we are at the initial state
    // we have to expand it wrt the batch size
    if (state.h.size() == 1 && input.dim().bd > 1)
    {
        for (unsigned i = 0u ; i < state.h.at(0).size() ; ++i)
        {
            state.h.at(0).at(i) = dynet::concatenate_to_batch(std::vector<dynet::Expression>(
                    input.dim().bd,
                    state.h.at(0).at(i)
            ));
            state.c.at(0).at(i) = dynet::concatenate_to_batch(std::vector<dynet::Expression>(
                    input.dim().bd,
                    state.c.at(0).at(i)
            ));
        }
    }

    state.c.emplace_back(settings.layers);
    state.h.emplace_back(settings.layers);
    const unsigned tm1 = state.h.size() - 2;
    const unsigned t = state.h.size() - 1;

    auto layer_input = input;
    for (unsigned i = 0; i < settings.layers; ++i) {
        const auto& vars = e_params.at(i);

        auto& i_h_tm1 = state.h.at(tm1).at(i);
        auto& i_c_tm1 = state.c.at(tm1).at(i);

        auto gates_t = dynet::vanilla_lstm_gates(
                {layer_input},
                i_h_tm1,
                vars.at(0),
                vars.at(1),
                vars.at(2)
        );
        //std::cerr << "vanilla_lstm_c: " << i_c_tm1.dim() << "\t" << gates_t << "\n";
        auto ct_i = vanilla_lstm_c(i_c_tm1, gates_t);
        auto ht_i = vanilla_lstm_h(ct_i, gates_t);
        layer_input = ht_i;

        if (mask == nullptr)
        {
            state.c.at(t).at(i) = ct_i;
            state.h.at(t).at(i) = ht_i;
        }
        else
        {
            state.c.at(t).at(i) = dynet::cmult(*mask, ct_i) + dynet::cmult((1.f - *mask), i_c_tm1);
            state.h.at(t).at(i) = dynet::cmult(*mask, ht_i) + dynet::cmult((1.f - *mask), i_h_tm1);
        }
    }
    return state.h.at(t).back();
}

}