#pragma once

#include "dynet/expr.h"
#include "dytools/builders/builder.h"

namespace dytools
{

struct BiAffineSettings
{
    unsigned proj_size = 128u;
    bool mod_bias = true;

    template<class Archive>
    void serialize(Archive& ar, const unsigned int)
    {
        ar & proj_size;
        ar & mod_bias;
    }
};



struct BiAffineBuilder : public Builder
{
    const BiAffineSettings settings;
    dynet::ParameterCollection local_pc;

    dynet::Parameter p_head_proj_W, p_head_proj_bias, p_mod_proj_W, p_mod_proj_bias;
    dynet::Parameter p_biaffine_head_mod, p_biaffine_head;

    dynet::Expression e_head_proj_W, e_head_proj_bias, e_mod_proj_W, e_mod_proj_bias;
    dynet::Expression e_biaffine_head_mod, e_biaffine_head;

    const bool root_prefix;
    dynet::Parameter p_root_prefix;
    dynet::Expression e_root_prefix;

    BiAffineBuilder(dynet::ParameterCollection& pc, const BiAffineSettings& settings, unsigned dim, bool root_prefix = true);

    void new_graph(dynet::ComputationGraph &cg, bool update = true);
    dynet::Expression operator()(const dynet::Expression& input, bool check_prefix = true);
    dynet::Expression operator()(const std::vector<dynet::Expression>& input);

};

}