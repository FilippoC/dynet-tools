#pragma once

#include <memory>
#include "dytools/builders/mlp.h"
#include "dynet/cfsm-builder.h"

namespace dytools
{

struct TaggerSettings
{
    bool output_bias = false;
    MLPSettings mlp;

    template<class Archive>
    void serialize(Archive& ar, const unsigned int)
    {
        ar & output_bias;
        ar & mlp;
    }
};

struct TaggerBuilder
{
    const TaggerSettings settings;
    dynet::ParameterCollection local_pc;

    MLPBuilder mlp;
    dynet::StandardSoftmaxBuilder builder;
    dynet::ComputationGraph* _cg;

    TaggerBuilder(dynet::ParameterCollection& pc, const TaggerSettings& settings, unsigned size, unsigned dim_input);
    void new_graph(dynet::ComputationGraph& cg, bool training, bool update);
    void set_dropout(float value);

    dynet::Expression full_logits(const dynet::Expression &input);
    dynet::Expression neg_log_softmax(const dynet::Expression& input, unsigned index);
    dynet::Expression neg_log_softmax(const dynet::Expression& input, const std::vector<unsigned>& words);

    // mask words that are not in the dictionnary
    dynet::Expression masked_neg_log_softmax(const dynet::Expression& input, const std::vector<int>& words, unsigned* c = nullptr, bool skip_first=false);
};

}