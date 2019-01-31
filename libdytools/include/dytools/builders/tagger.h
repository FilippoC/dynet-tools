#pragma once

#include <memory>
#include "dynet/dict.h"
#include "dynet/cfsm-builder.h"
#include "dytools/builders/builder.h"

namespace dytools
{

struct TaggerSettings
{
    bool output_bias = false;

    unsigned layers = 1u;
    unsigned dim = 128u;

    template<class Archive>
    void serialize(Archive& ar, const unsigned int)
    {
        ar & output_bias;
        ar & layers;
        ar & dim;
    }
};

struct TaggerBuilder : public Builder
{
    const TaggerSettings settings;
    dynet::ParameterCollection local_pc;
    std::shared_ptr<dynet::Dict> dict;

    std::vector<dynet::Parameter> p_W, p_bias;
    std::vector<dynet::Expression> e_W, e_bias;

    dynet::StandardSoftmaxBuilder builder;

    TaggerBuilder(dynet::ParameterCollection& pc, const TaggerSettings& settings, std::shared_ptr<dynet::Dict> dict, unsigned dim_input);
    void new_graph(dynet::ComputationGraph& cg, bool update = true);

    dynet::Expression full_logits(const dynet::Expression &input);
    dynet::Expression neg_log_softmax(const dynet::Expression& input, unsigned index);
};

}