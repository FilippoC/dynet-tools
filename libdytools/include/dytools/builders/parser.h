#pragma once

#include <utility>

#include "dynet/expr.h"
#include "dytools/builders/mlp.h"

namespace dytools
{

struct ParserSettings
{
    unsigned proj_dim = 256;
    MLPSettings input_mlp;
    MLPSettings output_mlp;
    bool unlabeled_bias = false;
    bool label_bias = false;

    template<class Archive>
    void serialize(Archive& ar, const unsigned int)
    {
        ar & proj_dim;
        ar & input_mlp;
        ar & output_mlp;
        ar & unlabeled_bias;
        ar & label_bias;
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
    dynet::Parameter p_output, p_output_label_bias;

    dynet::Expression e_proj_head, e_proj_mod, e_proj_bias;
    dynet::Expression e_output, e_output_label_bias;

    const bool has_bias;
    std::unique_ptr<MLPBuilder> output_mlp_bias;
    dynet::Parameter p_output_bias;
    dynet::Expression e_output_bias;

    ParserBuilder(
            dynet::ParameterCollection& pc,
            const ParserSettings& settings,
            const unsigned dim,
            const unsigned output_dim=1,
            bool root_prefix = true
    );

    void new_graph(dynet::ComputationGraph &cg, bool training, bool updates);
    void set_dropout(float value);

    dynet::Expression operator()(const dynet::Expression& input);
    dynet::Expression operator()(const dynet::Expression& head_input, const dynet::Expression& mod_input);

    std::pair<dynet::Expression, dynet::Expression> disjoint(const dynet::Expression& input);
    std::pair<dynet::Expression, dynet::Expression> disjoint(const dynet::Expression& head_input, const dynet::Expression& mod_input);

protected:
    dynet::Expression get_representation(const dynet::Expression& head_input, const dynet::Expression& mod_input);
    dynet::Expression get_labeled_weights(const dynet::Expression& values, const unsigned n_words);
    dynet::Expression get_unlabeled_weights(const dynet::Expression& values, const unsigned n_words);

};


}