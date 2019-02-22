#pragma once

#include <boost/serialization/vector.hpp>

#include "dynet/expr.h"
#include "dynet/dynet.h"
#include "dynet/param-init.h"
#include "dynet/lstm.h"

namespace dytools
{

struct MaskedLSTMSettings
{
    unsigned layers = 1;
    unsigned hidden_dim = 250;
    bool learn_init_state = false;

    template<class Archive> void serialize(Archive& ar, const unsigned int)
    {
        ar & layers;
        ar & hidden_dim;
        ar & learn_init_state;
    }
};

struct MaskedLSTMState;

struct MaskedLSTMBuilder
{
    const MaskedLSTMSettings settings;
    dynet::ParameterCollection local_pc;

    std::vector<std::vector<dynet::Parameter>> p_params;
    std::vector<dynet::Parameter> p_init_state_c;
    std::vector<dynet::Parameter> p_init_state_h;

    dynet::Expression e_init_zeros;
    std::vector<std::vector<dynet::Expression>> e_params;
    std::vector<dynet::Expression> e_init_state_c;
    std::vector<dynet::Expression> e_init_state_h;

    MaskedLSTMBuilder(dynet::ParameterCollection &pc, const MaskedLSTMSettings &t_settings, unsigned dim_input);

    void new_graph(dynet::ComputationGraph& cg, bool training, bool update);
    MaskedLSTMState new_state();
    dynet::Expression add_input(MaskedLSTMState& state, const dynet::Expression& input, const dynet::Expression* mask = nullptr);

    unsigned dim_output();
};

struct MaskedLSTMState
{
    MaskedLSTMState() = delete;
    MaskedLSTMState(MaskedLSTMBuilder& builder);
    std::vector<std::vector<dynet::Expression>> h, c;
};

}