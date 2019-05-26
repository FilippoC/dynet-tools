#pragma once

#include "dynet/expr.h"
#include "dytools/builders/mlp.h"

namespace dytools
{

struct ParserSettings
{
    unsigned proj_dim = 256;
    MLPSettings input_mlp;
    MLPSettings output_mlp;

    template<class Archive>
    void serialize(Archive& ar, const unsigned int)
    {
        ar & proj_dim;
        ar & input_mlp;
        ar & output_mlp;
    }
};



struct ParserBuilder
{
    const ParserSettings settings;
    dynet::ParameterCollection local_pc;

    MLPBuilder input_head_mlp;
    MLPBuilder input_mod_mlp;
    MLPBuilder output_mlp;

    dynet::Parameter p_proj_head, p_proj_mod, p_proj_bias;
    dynet::Parameter p_output;

    dynet::Expression e_proj_head, e_proj_mod, e_proj_bias;
    dynet::Expression e_output;

    const bool has_bias;
    std::unique_ptr<MLPBuilder> output_mlp_bias;
    dynet::Parameter p_output_bias;
    dynet::Expression e_output_bias;

    ParserBuilder(
            dynet::ParameterCollection& pc,
            const ParserSettings& settings,
            const unsigned dim,
            const unsigned output_dim=1,
            bool root_prefix = true,
            bool bias = false
    );

    void new_graph(dynet::ComputationGraph &cg, bool training, bool updates);
    void set_dropout(float value);

    dynet::Expression operator()(const dynet::Expression& input);
    dynet::Expression operator()(const dynet::Expression& head_input, const dynet::Expression& mod_input);
};


}