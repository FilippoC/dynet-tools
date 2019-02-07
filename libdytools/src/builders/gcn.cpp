#include "dytools/builders/gcn.h"

#include "dynet/param-init.h"

namespace dytools
{

GCNBuilder::GCNBuilder(dynet::ParameterCollection& pc, const GCNSettings& settings, unsigned dim_input) :
    settings(settings),
    local_pc(pc.add_subcollection("gcn")),
    e_W_parents(settings.layers),
    e_b_parents(settings.layers),
    e_W_children(settings.layers),
    e_b_children(settings.layers),
    e_W_self(settings.layers),
    e_b_self(settings.layers)
{
    unsigned last = dim_input;
    for (unsigned i = 0 ; i < settings.layers ; ++ i)
    {
        p_W_parents.push_back(local_pc.add_parameters({settings.dim, last}));
        p_b_parents.push_back(local_pc.add_parameters({settings.dim}, dynet::ParameterInitConst(0.f)));
        p_W_children.push_back(local_pc.add_parameters({settings.dim, last}));
        p_b_children.push_back(local_pc.add_parameters({settings.dim}, dynet::ParameterInitConst(0.f)));
        p_W_self.push_back(local_pc.add_parameters({settings.dim, last}));
        p_b_self.push_back(local_pc.add_parameters({settings.dim}, dynet::ParameterInitConst(0.f)));

        if (settings.dense)
            last = settings.dim + last;
        else
            last = settings.dim;
    }
    _output_rows = last;
}

void GCNBuilder::new_graph(dynet::ComputationGraph& cg, bool update)
{
    for (unsigned i = 0 ; i < settings.layers ; ++ i)
    {
        if (update)
        {
            e_W_parents.at(i) = dynet::parameter(cg, p_W_parents.at(i));
            e_b_parents.at(i) = dynet::parameter(cg, p_b_parents.at(i));
            e_W_children.at(i) = dynet::parameter(cg, p_W_children.at(i));
            e_b_children.at(i) = dynet::parameter(cg, p_b_children.at(i));
            e_W_self.at(i) = dynet::parameter(cg, p_W_self.at(i));
            e_b_self.at(i) = dynet::parameter(cg, p_b_self.at(i));
        }
        else
        {
            e_W_parents.at(i) = dynet::const_parameter(cg, p_W_parents.at(i));
            e_b_parents.at(i) = dynet::const_parameter(cg, p_b_parents.at(i));
            e_W_children.at(i) = dynet::const_parameter(cg, p_W_children.at(i));
            e_b_children.at(i) = dynet::const_parameter(cg, p_b_children.at(i));
            e_W_self.at(i) = dynet::const_parameter(cg, p_W_self.at(i));
            e_b_self.at(i) = dynet::const_parameter(cg, p_b_self.at(i));
        }
    }
}

dynet::Expression GCNBuilder::apply(const dynet::Expression &input, const dynet::Expression& graph)
{
    if (settings.layers == 0)
        return input;

    auto t_graph = dynet::transpose(graph);
    auto last = input;
    for (unsigned i = 0u ; i < settings.layers ; ++i)
    {
        auto current = dynet::colwise_add(e_W_self[i] * last,  e_b_self[i]);

        auto parents = dynet::colwise_add(e_W_parents[i] * last, e_b_parents[i]);
        current = current + parents * graph;

        auto children = dynet::colwise_add(e_W_children[i] * last, e_b_children[i]);
        current = current + children * t_graph;

        current = dynet::tanh(current);

        if (settings.dense)
            last = dynet::concatenate({current, last});
        else
            last = current;
    }

    return last;
}

unsigned GCNBuilder::output_rows() const
{
    return _output_rows;
}

}