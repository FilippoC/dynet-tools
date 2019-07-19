#pragma once

#include "dytools/builders/masked_lstm.h"

namespace dytools
{

struct MaskedBiLSTMSettings
{
    MaskedLSTMSettings lstm;
    unsigned n_stack = 0;
    bool padding = false;

    template<class Archive> void serialize(Archive& ar, const unsigned int)
    {
        ar & lstm;
        ar & n_stack;
        ar & padding;
    }
};

struct MaskedBiLSTMBuilder
{
    const MaskedBiLSTMSettings settings;
    dynet::ParameterCollection local_pc;

    const unsigned input_dim;

    const int pad_begin = 0;
    const int pad_end = 1;
    dynet::LookupParameter pad;

    std::vector<MaskedLSTMBuilder> forward;
    std::vector<MaskedLSTMBuilder> backward;

    dynet::Expression e_begin, e_end;

    explicit MaskedBiLSTMBuilder(dynet::ParameterCollection& pc, const MaskedBiLSTMSettings& t_settings, unsigned t_input_dim);

    void new_graph(dynet::ComputationGraph& cg, bool training, bool update);
    std::vector<std::pair<dynet::Expression,dynet::Expression>> compute(const std::vector<dynet::Expression>& input, const dynet::Expression* mask = nullptr);
    std::vector<dynet::Expression> operator()(const std::vector<dynet::Expression>& input, const dynet::Expression* mask = nullptr);
    dynet::Expression operator()(const dynet::Expression& input, const dynet::Expression* mask = nullptr);

    unsigned dim_output();
};


}