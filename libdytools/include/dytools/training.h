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
    bool save_at_each_epoch = false;
};

template <class Network, class DataType, class Evaluator>
struct Training
{
    const TrainingSettings settings;
    std::shared_ptr<Network> network;

    Training(std::shared_ptr<Network> _network);
    Training(const TrainingSettings& settings, std::shared_ptr<Network> _network);

    virtual void optimize_supervised(
            dynet::Trainer &trainer,
            std::vector<DataType> &labeled_data,
            const std::vector<DataType> &dev_data
    );
    virtual void optimize_unsupervised(
            dynet::Trainer &trainer,
            std::vector<DataType> &unlabeled_data,
            const std::vector<DataType> &dev_data
            );

    virtual void optimize_semisupervised(
            dynet::Trainer &trainer,
            std::vector<DataType> &labeled_data,
            std::vector<DataType> &unlabeled_data,
            const std::vector<DataType> &dev_data
            );

    virtual dynet::Expression labeled_loss(const DataType &data);
    virtual dynet::Expression unlabeled_loss(const DataType &data);

    virtual float evaluate(const std::vector<DataType>& data);

    virtual float forward_backward(
            dynet::ComputationGraph& cg,
            typename std::vector<DataType>::const_iterator begin_labelled_data,
            typename std::vector<DataType>::const_iterator end_labelled_data,
            typename std::vector<DataType>::const_iterator begin_unlabelled_data,
            typename std::vector<DataType>::const_iterator end_unlabelled_data
    );

    virtual void save();
    virtual void save(const std::string& path);
    virtual void load();
    virtual void load(const std::string& path);

protected:
    virtual void training_settings(const std::string& mode);
    virtual void optimize(
            dynet::Trainer &trainer,
            std::vector<DataType> &labeled_data,
            std::vector<DataType> &unlabeled_data,
            const std::vector<DataType> &dev_data
    );
};

}

#include "dytools/training_impl.h"