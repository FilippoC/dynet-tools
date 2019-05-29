#include "dytools/builders/mlp.h"

#include "dynet/param-init.h"

namespace dytools
{

MLPBuilder::MLPBuilder(dynet::ParameterCollection& pc, const MLPSettings& settings, unsigned dim_input) :
        settings(settings),
        local_pc(pc.add_subcollection("mlp")),
        e_W(settings.layers),
        e_bias(settings.layers)
{
    unsigned last = dim_input;
    for (unsigned i = 0 ; i < settings.layers ; ++ i)
    {
        p_W.push_back(local_pc.add_parameters({settings.dim, last}));
        p_bias.push_back(local_pc.add_parameters({settings.dim}, dynet::ParameterInitConst(0.f)));

        last = settings.dim;
    }
    _output_rows = last;
}

void MLPBuilder::new_graph(dynet::ComputationGraph& cg, bool training, bool update)
{
    _training = training;
    for (unsigned i = 0 ; i < settings.layers ; ++ i)
    {
        if (update)
        {
            e_W.at(i) = dynet::parameter(cg, p_W.at(i));
            e_bias.at(i) = dynet::parameter(cg, p_bias.at(i));
        }
        else
        {
            e_W.at(i) = dynet::const_parameter(cg, p_W.at(i));
            e_bias.at(i) = dynet::const_parameter(cg, p_bias.at(i));
        }
    }
}

dynet::Expression MLPBuilder::apply(const dynet::Expression &input)
{
    if (settings.layers == 0)
        return input;

    auto last = input;
    for (unsigned i = 0u ; i < settings.layers ; ++i)
    {
        //auto proj = dynet::colwise_add(e_W[i] * last,  e_bias[i]);
        auto proj = e_W.at(i) * last + e_bias.at(i);
        //auto proj = dynet::affine_transform({e_bias[i], e_W[i], last});
        if (dropout_rate > 0.f)
        {
            //if (_training)
            //    proj = dynet::dropout(proj, dropout_rate);
            //else
                // because of dynet bug on CPU
            //    proj = dynet::dropout(proj, 0.f);

        }
        last = dytools::activation(proj, settings.activation);
    }

    return last;
}

void MLPBuilder::set_dropout(float value)
{
    dropout_rate = value;
}

unsigned MLPBuilder::output_rows() const
{
    return _output_rows;
}

}