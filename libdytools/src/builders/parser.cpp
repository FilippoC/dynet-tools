#include "dytools/builders/parser.h"
#include <dynet/param-init.h>

namespace dytools
{

ParserBuilder::ParserBuilder(
        dynet::ParameterCollection& pc,
        const ParserSettings& settings,
        const unsigned dim,
        const unsigned output_dim,
        const bool root_prefix
) :
    settings(settings),
    local_pc(pc.add_subcollection("parser")),
    input_head_mlp(local_pc, settings.input_mlp, dim),
    input_mod_mlp(local_pc, settings.input_mlp, dim),
    output_mlp(local_pc, settings.output_mlp, settings.proj_dim),
    has_bias(settings.unlabeled_bias)
{
    if (output_dim == 0)
        throw std::runtime_error("output_dim must be at least one");
    if (output_dim == 1 && has_bias == true)
        throw std::runtime_error("bias is useless when output is one");
    if (output_dim == 1 && settings.label_bias == true)
        throw std::runtime_error("Label bias is useless when output is one");

    p_proj_head = local_pc.add_parameters({settings.proj_dim, input_head_mlp.output_rows()});
    p_proj_mod = local_pc.add_parameters({settings.proj_dim, input_head_mlp.output_rows()});
    p_proj_bias = local_pc.add_parameters({settings.proj_dim}, dynet::ParameterInitConst(0.f));

    p_output = local_pc.add_parameters({output_dim, output_mlp.output_rows()});

    if (settings.label_bias)
        p_output_label_bias = local_pc.add_parameters({output_dim}, dynet::ParameterInitConst(0.f));

    if (has_bias)
    {
        output_mlp_bias.reset(new MLPBuilder(local_pc, settings.output_mlp, settings.proj_dim));
        p_output_bias = local_pc.add_parameters({1u, output_mlp.output_rows()});
    }
}


void ParserBuilder::new_graph(dynet::ComputationGraph &cg, bool training, bool updates)
{
    input_head_mlp.new_graph(cg, training, updates);
    input_mod_mlp.new_graph(cg, training, updates);
    output_mlp.new_graph(cg, training, updates);
    if (has_bias)
        output_mlp_bias->new_graph(cg, training, updates);

    if (updates)
    {
        e_proj_head = dynet::parameter(cg, p_proj_head);
        e_proj_mod = dynet::parameter(cg, p_proj_mod);
        e_proj_bias = dynet::parameter(cg, p_proj_bias);
        e_output = dynet::parameter(cg, p_output);

        if (has_bias)
            e_output_bias = dynet::parameter(cg, p_output_bias);

        if (settings.label_bias)
            e_output_label_bias = dynet::parameter(cg, p_output_label_bias);
    }
    else
    {
        e_proj_head = dynet::const_parameter(cg, p_proj_head);
        e_proj_mod = dynet::const_parameter(cg, p_proj_mod);
        e_proj_bias = dynet::const_parameter(cg, p_proj_bias);
        e_output = dynet::const_parameter(cg, p_output);

        if (has_bias)
            e_output_bias = dynet::const_parameter(cg, p_output_bias);

        if (settings.label_bias)
            e_output_label_bias = dynet::const_parameter(cg, p_output_label_bias);
    }
}


void ParserBuilder::set_dropout(float value)
{
    input_head_mlp.set_dropout(value);
    input_mod_mlp.set_dropout(value);
    output_mlp.set_dropout(value);
}

dynet::Expression ParserBuilder::operator()(const dynet::Expression& input)
{
    return (*this)(input, input);
}

dynet::Expression ParserBuilder::operator()(const dynet::Expression& head_input, const dynet::Expression& mod_input)
{
    const unsigned n_words = mod_input.dim().cols();

    const auto repr = get_representation(head_input, mod_input);

    if (has_bias)
        return get_unlabeled_weights(repr, n_words) + get_labeled_weights(repr, n_words);
    else
        return get_labeled_weights(repr, n_words);
}


dynet::Expression ParserBuilder::get_representation(const dynet::Expression& head_input, const dynet::Expression& mod_input)
{
    // input dim: (feats, n_words)
    const unsigned n_words = mod_input.dim().cols();

    // dim: (feats, n_words)
    auto head_proj = e_proj_head * input_head_mlp.apply(head_input);
    auto mod_proj = e_proj_mod * input_mod_mlp.apply(mod_input);

    const unsigned dim_feats = head_proj.dim().rows();

    head_proj = dynet::reshape(head_proj, {dim_feats, 1, n_words});
    mod_proj = dynet::reshape(mod_proj, {dim_feats, n_words, 1});

    // dim: (feats, n_words, n_words)
    auto values = head_proj + mod_proj;

    // dim: (feats, n_words * n_words)
    values = dynet::reshape(values, {dim_feats, n_words * n_words});
    values = dynet::rectify(values + e_proj_bias);

    return values;
}

dynet::Expression ParserBuilder::get_labeled_weights(const dynet::Expression& values, const unsigned n_words)
{

    auto output_values = output_mlp.apply(values);

    // dim: (output, n_words * n_words)
    auto output = e_output * output_values;
    if (settings.label_bias)
        output = output + e_output_label_bias;
    output = dynet::reshape(output, {e_output.dim().rows(), n_words, n_words});

    return output;
}

dynet::Expression ParserBuilder::get_unlabeled_weights(const dynet::Expression& values, const unsigned n_words)
{

    if (!has_bias)
        throw std::runtime_error("Unlabaled bias is disabled");

    auto output_bias_values = output_mlp_bias->apply(values);
    auto output_bias = e_output_bias * output_bias_values;
    output_bias = dynet::reshape(output_bias, {1u, n_words, n_words});

    return output_bias;
}

std::pair<dynet::Expression, dynet::Expression> ParserBuilder::disjoint(const dynet::Expression& input)
{
    return disjoint(input, input);
}

std::pair<dynet::Expression, dynet::Expression> ParserBuilder::disjoint(const dynet::Expression& head_input, const dynet::Expression& mod_input)
{
    const unsigned n_words = mod_input.dim().cols();

    const auto repr = get_representation(head_input, mod_input);
    return {get_unlabeled_weights(repr, n_words), get_labeled_weights(repr, n_words)};
}


}