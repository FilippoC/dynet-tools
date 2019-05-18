#pragma once

#include <memory>
#include "dynet/expr.h"

namespace dytools
{

struct BiAffineTaggerSettings
{
    unsigned proj_size = 128u;
    bool bias = true;
    bool label_bias = true;

    template<class Archive>
    void serialize(Archive& ar, const unsigned int)
    {
        ar & proj_size;
    }
};



struct BiAffineTaggerBuilder
{
    const BiAffineTaggerSettings settings;
    const unsigned n_labels;
    dynet::ParameterCollection local_pc;

    dynet::Parameter p_head_proj_W, p_head_proj_bias, p_mod_proj_W, p_mod_proj_bias;
    dynet::Parameter p_biaffine_head_mod, p_biaffine_bias, p_biaffine_label_bias;

    dynet::Expression e_head_proj_W, e_head_proj_bias, e_mod_proj_W, e_mod_proj_bias;
    dynet::Expression e_biaffine_head_mod, e_biaffine_bias, e_biaffine_label_bias;

    const bool root_prefix;
    dynet::Parameter p_root_prefix;
    dynet::Expression e_root_prefix;

    BiAffineTaggerBuilder(
            dynet::ParameterCollection& pc,
            const BiAffineTaggerSettings& settings,
            unsigned dim,
            const unsigned size,
            bool root_prefix = true
    );

    void new_graph(dynet::ComputationGraph& cg, bool training, bool update);
    dynet::Expression dependency_tagger(const std::vector<dynet::Expression>& input, const std::vector<unsigned>& heads);

protected:
    dynet::Expression apply(const dynet::Expression& e_head, const dynet::Expression& e_mod);
};

}