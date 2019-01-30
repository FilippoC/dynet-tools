#pragma once

#include <vector>
#include <memory>

#include "dynet/expr.h"
#include "dynet/training.h"

namespace dytools
{

struct TrainingSettings
{
    unsigned n_epoch = 100u;
    unsigned n_updates_per_epoch = 10000u;
    unsigned batch_size = 1u;

    unsigned patience = 0u;
    unsigned max_trials = 0u;
    float lr_decay = 0.5;
    bool reload_parameters = false;

    std::string model_path;
};

template <class Network, class DataType>
struct Training
{
    const TrainingSettings settings;
    std::shared_ptr<Network> network;

    Training(std::shared_ptr<Network> _network);
    Training(const TrainingSettings& settings, std::shared_ptr<Network> _network);

    virtual void optimize(dynet::Trainer& trainer, std::vector<DataType>& train_data, const std::vector<DataType>& dev_data);
    virtual float forward_backward(dynet::ComputationGraph& cg, typename std::vector<DataType>::const_iterator begin_data, typename std::vector<DataType>::const_iterator end_data);
    virtual float evaluate(const std::vector<DataType>& data);

    virtual dynet::Expression compute_loss(typename std::vector<DataType>::const_iterator begin_data, typename std::vector<DataType>::const_iterator end_data);
    virtual dynet::Expression compute_loss(const DataType& data);
    virtual float evaluate(const DataType& data);

    virtual void save();
    virtual void load();
};

}

#include "dytools/training_impl.h"