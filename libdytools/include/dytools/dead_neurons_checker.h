#pragma once

#include <vector>
#include <utility>
#include <algorithm>
#include <memory>

#include "dynet/expr.h"
#include "dynet/nodes-activations.h"

namespace dytools
{

struct NodeChecker
{
    const int rid;
    std::vector<unsigned> alive_counts;

    NodeChecker(dynet::ComputationGraph& cg, int rid_);

    void check(dynet::ComputationGraph& cg);
    unsigned dead_counts() const;
    unsigned dim() const;
    int id(dynet::ComputationGraph& cg) const;
    const dynet::Node* node(dynet::ComputationGraph& cg) const;
};

struct GraphChecker
{
    std::vector<NodeChecker> node_sanities;

    GraphChecker(dynet::ComputationGraph& cg);

    void operator()(dynet::ComputationGraph& cg);
    std::pair<unsigned, unsigned> dead_ratio();
};

}