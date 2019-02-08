#include "dytools/builders/biaffine.h"

#include "dynet/param-init.h"

namespace dytools
{

BiAffineBuilder::BiAffineBuilder(dynet::ParameterCollection& pc, const BiAffineSettings& settings, unsigned dim, bool root_prefix) :
    settings(settings),
    local_pc(pc.add_subcollection("biaffine")),
    root_prefix(root_prefix)
{
    p_head_proj_W = local_pc.add_parameters({settings.proj_size, dim});
    p_head_proj_bias = local_pc.add_parameters({settings.proj_size}, dynet::ParameterInitConst(0.f));
    p_mod_proj_W = local_pc.add_parameters({settings.proj_size, dim});
    p_mod_proj_bias = local_pc.add_parameters({settings.proj_size}, dynet::ParameterInitConst(0.f));

    p_biaffine_head_mod = local_pc.add_parameters({settings.proj_size, settings.proj_size});

    if (settings.mod_bias)
        p_biaffine_head = local_pc.add_parameters({settings.proj_size, 1});

    if (root_prefix)
        p_root_prefix = local_pc.add_parameters({dim});

    std::cerr
        << "BiAffine attention\n"
        << " projection size: " << settings.proj_size << "\n"
        << " modifier bias: " << (settings.mod_bias ? "yes" : "no") << "\n"
        << " root prefix: " << (root_prefix ? "yes" : "no") << "\n"
        << "\n"
        ;
}

void BiAffineBuilder::new_graph(dynet::ComputationGraph &cg, bool update)
{
    if (update)
    {
        e_head_proj_W = parameter(cg, p_head_proj_W);
        e_head_proj_bias = parameter(cg, p_head_proj_bias);
        e_mod_proj_W = parameter(cg, p_mod_proj_W);
        e_mod_proj_bias = parameter(cg, p_mod_proj_bias);

        e_biaffine_head_mod = parameter(cg, p_biaffine_head_mod);
        if (settings.mod_bias)
            e_biaffine_head = parameter(cg, p_biaffine_head);

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
        if (settings.mod_bias)
            e_biaffine_head = const_parameter(cg, p_biaffine_head);

        if (root_prefix)
            e_root_prefix = const_parameter(cg, p_root_prefix);
    }

}

dynet::Expression BiAffineBuilder::operator()(const dynet::Expression& c_input, bool check_prefix)
{
    const dynet::Expression input = (
        check_prefix && root_prefix
        ? dynet::concatenate_cols({e_root_prefix, c_input})
        : c_input
        );

    auto e_head = dynet::transpose(dynet::tanh(e_head_proj_W * input + e_head_proj_bias));
    auto e_mod = dynet::tanh(e_mod_proj_W * input + e_mod_proj_bias);

    auto weights =
            e_head * e_biaffine_head_mod * e_mod
    ;

    if (settings.mod_bias)
        weights = weights + e_head * e_biaffine_head;

    return weights;

}

dynet::Expression BiAffineBuilder::operator()(const std::vector<dynet::Expression>& input)
{
    if (root_prefix)
    {
        std::vector<dynet::Expression> input2;
        input2.push_back(e_root_prefix);
        input2.insert(std::end(input2), std::begin(input), std::end(input));
        return (*this)(dynet::concatenate_cols(input2), false);
    }
    else
        return (*this)(dynet::concatenate_cols(input), false);
}


}