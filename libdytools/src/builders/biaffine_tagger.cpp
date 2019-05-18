#include "dytools/builders/biaffine_tagger.h"

#include "dynet/param-init.h"

namespace dytools
{

BiAffineTaggerBuilder::BiAffineTaggerBuilder(
        dynet::ParameterCollection& pc,
        const BiAffineTaggerSettings& settings,
        unsigned dim,
        const unsigned size,
        bool root_prefix
) :
    settings(settings),
    n_labels(size),
    local_pc(pc.add_subcollection("biaffinetagger")),
    root_prefix(root_prefix)
{

    p_head_proj_W = local_pc.add_parameters({settings.proj_size, dim});
    p_head_proj_bias = local_pc.add_parameters({settings.proj_size}, dynet::ParameterInitConst(0.f));
    p_mod_proj_W = local_pc.add_parameters({settings.proj_size, dim});
    p_mod_proj_bias = local_pc.add_parameters({settings.proj_size}, dynet::ParameterInitConst(0.f));

    p_biaffine_head_mod = local_pc.add_parameters({settings.proj_size * n_labels, settings.proj_size});

    if (settings.bias)
        p_biaffine_bias = local_pc.add_parameters({n_labels, settings.proj_size * 2});

    if (settings.label_bias)
        p_biaffine_label_bias = local_pc.add_parameters({n_labels}, dynet::ParameterInitConst(0.f));

    if (root_prefix)
        p_root_prefix = local_pc.add_parameters({dim});

    std::cerr
            << "BiAffine Tagger\n"
            << " projection size: " << settings.proj_size << "\n"
            << " bias: " << (settings.bias ? "yes" : "no") << "\n"
            << " label bias: " << (settings.bias ? "yes" : "no") << "\n"
            << " root prefix: " << (root_prefix ? "yes" : "no") << "\n"
            << " n labels: " << size << "\n";
    std::cerr << "\n";
}

void BiAffineTaggerBuilder::new_graph(dynet::ComputationGraph &cg, bool, bool update)
{
    if (update)
    {
        e_head_proj_W = parameter(cg, p_head_proj_W);
        e_head_proj_bias = parameter(cg, p_head_proj_bias);
        e_mod_proj_W = parameter(cg, p_mod_proj_W);
        e_mod_proj_bias = parameter(cg, p_mod_proj_bias);

        e_biaffine_head_mod = parameter(cg, p_biaffine_head_mod);

        if (settings.bias)
            e_biaffine_bias = parameter(cg, p_biaffine_bias);

        if (settings.label_bias)
            e_biaffine_label_bias = parameter(cg, p_biaffine_label_bias);

        if (root_prefix)
            e_root_prefix = parameter(cg, p_root_prefix);
    }
    else
    {
        e_head_proj_W = const_parameter(cg, p_head_proj_W);
        e_head_proj_bias = const_parameter(cg, p_head_proj_bias);
        e_mod_proj_W = const_parameter(cg, p_mod_proj_W);
        e_mod_proj_bias = const_parameter(cg, p_mod_proj_bias);

        e_biaffine_head_mod = const_parameter(cg, p_biaffine_head_mod);

        if (settings.bias)
            e_biaffine_bias = const_parameter(cg, p_biaffine_bias);

        if (settings.label_bias)
            e_biaffine_label_bias = const_parameter(cg, p_biaffine_label_bias);

        if (root_prefix)
            e_root_prefix = const_parameter(cg, p_root_prefix);
    }

}

dynet::Expression BiAffineTaggerBuilder::dependency_tagger(const std::vector<dynet::Expression>& input, const std::vector<unsigned>& heads)
{
    std::vector<dynet::Expression> v_head_input;
    for (const unsigned head_index : heads)
    {
        if (root_prefix)
        {
            if (head_index == 0)
                v_head_input.push_back(e_root_prefix);
            else
                v_head_input.push_back(input.at(head_index - 1));
        }
        else
        {
            v_head_input.push_back(input.at(head_index));
        }
    }

    const auto mod_input = dynet::concatenate_to_batch(input);
    const auto head_input = dynet::concatenate_to_batch(v_head_input);

    const auto e_head = dynet::rectify(e_head_proj_W * head_input + e_head_proj_bias);
    const auto e_mod = dynet::rectify(e_mod_proj_W * mod_input + e_mod_proj_bias);

    auto weights = e_biaffine_head_mod * e_mod;
    weights = dynet::reshape(weights, dynet::Dim({settings.proj_size, n_labels}, weights.dim().batch_elems()));
    weights = dynet::transpose(e_head) * weights;
    weights = dynet::transpose(weights);

    if (settings.bias)
        weights = weights + e_biaffine_bias * dynet::concatenate({e_head, e_mod});

    if (settings.label_bias)
        weights = weights + e_biaffine_label_bias;

    return weights;
}


}