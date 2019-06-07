//
// Created by Caio Filippo Corro on 2019-06-07.
//

#include "dytools/dead_neurons_checker.h"

namespace dytools
{

NodeChecker::NodeChecker(dynet::ComputationGraph& cg, int rid_) :
    rid(rid_)
{
        alive_counts.resize(node(cg)->dim.batch_size(), 0u);
}

void NodeChecker::check(dynet::ComputationGraph& cg)
{
    const dynet::Node* node_ = node(cg);
    if (dynamic_cast<const dynet::Rectify *>(node_) == nullptr)
        throw std::runtime_error("Rectifiers rid has changed! :(");
    if (dim() != node_->dim.batch_size())
        throw std::runtime_error("Node size changed");

    const unsigned batch_size = dim();
    const unsigned n_batches = node_->dim.batch_elems();
    const auto values = as_vector(cg.get_value(id(cg)));


    for (unsigned batch = 0u ; batch < n_batches ; ++batch)
        for (unsigned i = 0 ; i < batch_size ; ++i)
            if (values.at(i + batch * batch_size) > 0.f)
                ++ alive_counts.at(i);
}

unsigned NodeChecker::dead_counts() const
{
    return std::count(alive_counts.begin(), alive_counts.end(), 0u);
}

unsigned NodeChecker::dim() const
{
    return alive_counts.size();
}
int NodeChecker::id(dynet::ComputationGraph& cg) const
{
    return (int) cg.nodes.size() - 1 - rid;
}

const dynet::Node* NodeChecker::node(dynet::ComputationGraph& cg) const
{
    return cg.nodes.at(id(cg));
}


GraphChecker::GraphChecker(dynet::ComputationGraph& cg)
{
    for (unsigned node_rid = 0; node_rid < cg.nodes.size(); ++node_rid)
    {
        const dynet::Node *const node = cg.nodes.at(cg.nodes.size() - 1 - node_rid);
        if (dynamic_cast<const dynet::Rectify *>(node) != nullptr)
            node_sanities.emplace_back(cg, node_rid);
    }
}

void GraphChecker::operator()(dynet::ComputationGraph& cg)
{
    for (auto& node_sanity : node_sanities)
        node_sanity.check(cg);
}

std::pair<unsigned, unsigned> GraphChecker::dead_ratio()
{
    unsigned n_dead = 0u;
    unsigned n_total = 0u;
    for (auto& node_sanity : node_sanities)
    {
        n_dead += node_sanity.dead_counts();
        n_total += node_sanity.dim();
    }
    return {n_dead,  n_total};
}


}