#pragma once

#include "dynet/model.h"
#include "dynet/lstm.h"

namespace dytools
{

struct BiLSTMSettings
{
    unsigned stacks = 1u;
    unsigned layers = 1u;
    unsigned dim = 128u;
    bool boundaries = false;

    template<class Archive>
    void serialize(Archive& ar, const unsigned int)
    {
        ar & stacks;
        ar & layers;
        ar & dim;
        ar & boundaries;
    }

    unsigned int output_rows(const unsigned input_dim) const;
};

struct BiLSTMBuilder
{
    const BiLSTMSettings settings;
    dynet::ParameterCollection local_pc;
    const unsigned input_dim;
    float dropout = 0.f;

    std::vector<std::pair<dynet::VanillaLSTMBuilder, dynet::VanillaLSTMBuilder>> builders;
    dynet::Parameter p_begin, p_end;
    dynet::Expression e_begin, e_end;

    BiLSTMBuilder(dynet::ParameterCollection& pc, const BiLSTMSettings& settings, unsigned input_dim);

    void new_graph(dynet::ComputationGraph& cg, bool training, bool update);
    std::vector<dynet::Expression> operator()(const std::vector<dynet::Expression>& embeddings);
    dynet::Expression endpoints(const std::vector<dynet::Expression>& embeddings);
    void set_dropout(float value);

    unsigned output_rows() const;
};

}