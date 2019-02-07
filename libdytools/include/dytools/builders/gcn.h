#pragma once

#include <vector>

#include "dynet/model.h"
#include "dynet/expr.h"
#include "dytools/builders/builder.h"

namespace dytools
{

struct GCNSettings
{
    unsigned dim = 128;
    unsigned layers = 1;
    bool dense = false;

    template<class Archive>
    void serialize(Archive &ar, const unsigned int)
    {
        ar & dim;
        ar & layers;
        ar & dense;
    }
};

struct GCNBuilder : public Builder
{
    const GCNSettings settings;
    dynet::ParameterCollection local_pc;

    std::vector<dynet::Parameter> p_W_parents, p_b_parents, p_W_children, p_b_children, p_W_self, p_b_self;
    std::vector<dynet::Expression> e_W_parents, e_b_parents, e_W_children, e_b_children, e_W_self, e_b_self;

    unsigned _output_rows;

    GCNBuilder(dynet::ParameterCollection& pc, const GCNSettings& settings, unsigned dim_input);

    void new_graph(dynet::ComputationGraph& cg, bool update = true);
    dynet::Expression apply(const dynet::Expression &input, const dynet::Expression& graph);

    unsigned output_rows() const;
};


}