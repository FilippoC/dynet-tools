#pragma once

#include <chrono>
#include <limits>
#include <dynet/io.h>

namespace dytools
{

template <class Network, class DataType, class Evaluator, class Epoch>
Training<Network, DataType, Evaluator, Epoch>::Training(std::shared_ptr<Network> _network) :
    network(std::move(_network))
{}


template <class Network, class DataType, class Evaluator, class Epoch>
Training<Network, DataType, Evaluator, Epoch>::Training(const TrainingSettings& settings, std::shared_ptr<Network> _network) :
settings(settings),
network(std::move(_network))
{}

template <class Network, class DataType, class Evaluator, class Epoch>
void Training<Network, DataType, Evaluator, Epoch>::optimize_supervised(
        dynet::Trainer &trainer,
        std::vector<DataType> &labeled_data,
        const std::vector<DataType> &dev_data
)
{
    training_settings("Supervised training");
    std::vector<DataType> fake_container;
    optimize(trainer, labeled_data, fake_container, dev_data);
}

template <class Network, class DataType, class Evaluator, class Epoch>
void Training<Network, DataType, Evaluator, Epoch>::optimize_unsupervised(
        dynet::Trainer &trainer,
        std::vector<DataType> &unlabeled_data,
        const std::vector<DataType> &dev_data
)
{
    training_settings("Unsupervised training");
    std::vector<DataType> fake_container;
    optimize(trainer, fake_container, unlabeled_data, dev_data);
}

template <class Network, class DataType, class Evaluator, class Epoch>
void Training<Network, DataType, Evaluator, Epoch>::optimize_semisupervised(
        dynet::Trainer &trainer,
        std::vector<DataType> &labeled_data,
        std::vector<DataType> &unlabeled_data,
        const std::vector<DataType> &dev_data
)
{
    training_settings("Semi-supervised training");
    optimize(trainer, labeled_data, unlabeled_data, dev_data);
}

template <class Network, class DataType, class Evaluator, class Epoch>
void Training<Network, DataType, Evaluator, Epoch>::optimize(
        dynet::Trainer &trainer,
        std::vector<DataType> &labeled_data,
        std::vector<DataType> &unlabeled_data,
        const std::vector<DataType> &dev_data
)
{
    float best_dev_score = -std::numeric_limits<float>::infinity();
    unsigned best_dev_epoch = 0u;
    unsigned n_trials = 0u;
    unsigned n_epoch_without_improvement = 0u;

    Epoch epoch_optimizer(network, trainer);
    for (unsigned epoch = 0; epoch < settings.n_epoch; ++epoch)
    {
        std::cerr << "\nEpoch " << epoch << "/" << settings.n_epoch << std::endl;

        auto start_epoch = std::chrono::steady_clock::now();

        const float epoch_loss = epoch_optimizer.optimize(
                labeled_data,
                unlabeled_data,
                settings.n_updates_per_epoch,
                settings.batch_size
        );

        auto end_epoch = std::chrono::steady_clock::now();
        std::cerr
                << "Epoch loss: " << epoch_loss
                << "\t/\tDuration: "
                << std::chrono::duration_cast<std::chrono::seconds>(end_epoch - start_epoch).count()
                << std::endl;

        if (settings.save_at_each_epoch)
            save(settings.model_path + "." + std::to_string(epoch));

        // evaluate on dev data
        float dev_score = evaluate(dev_data);
        if (dev_score > best_dev_score)
        {
            std::cerr << "dev score as increased: " << dev_score << " > " << best_dev_score << std::endl;
            best_dev_score = dev_score;
            best_dev_epoch = epoch;
            n_epoch_without_improvement = 0u;

            save();
        }
        else
        {
            ++ n_epoch_without_improvement;
        }

        if (settings.patience > 0 && n_epoch_without_improvement >= settings.patience)
        {
            ++ n_trials;
            if (settings.max_trials > 0 && n_trials >= settings.max_trials)
            {
                std::cerr << "Maximum number of trial! Abort." << std::endl;
                break;
            }
            n_epoch_without_improvement = 0u;
            trainer.learning_rate = trainer.learning_rate * settings.lr_decay;
            std::cerr
                << "Dev score did not improve in the last " << settings.patience << " epochs, "
                << "annealing the lr.\n"
                << "New learning rate: " << trainer.learning_rate
                << std::endl;

            if (settings.reload_parameters)
                load();
        }
    }
    std::cerr
        << "\n"
        << "Training finished.\n"
        << "Best dev score was on epoch " << best_dev_epoch << " (" << best_dev_score << ")"
        << std::endl;
}

template <class Network, class DataType, class Evaluator, class Epoch>
float Training<Network, DataType, Evaluator, Epoch>::evaluate(const std::vector<DataType>& data)
{
    return Evaluator()(network, data);
}

template <class Network, class DataType, class Evaluator, class Epoch>
void Training<Network, DataType, Evaluator, Epoch>::save(const std::string& path)
{
    std::cerr << "Saving model to: " << path << std::endl;
    dynet::TextFileSaver s(path);
    s.save(network->local_pc);
}

template <class Network, class DataType, class Evaluator, class Epoch>
void Training<Network, DataType, Evaluator, Epoch>::save()
{
    if (settings.model_path.size() > 0)
        save(settings.model_path);
}

template <class Network, class DataType, class Evaluator, class Epoch>
void Training<Network, DataType, Evaluator, Epoch>::load(const std::string& path)
{
    std::cerr << "Loading model from: " << path << std::endl;
    dynet::TextFileLoader s(path);
    s.populate(network->local_pc);
}

template <class Network, class DataType, class Evaluator, class Epoch>
void Training<Network, DataType, Evaluator, Epoch>::load()
{
    if (settings.model_path.size() > 0)
        load(settings.model_path);
}


template <class Network, class DataType, class Evaluator, class Epoch>
void Training<Network, DataType, Evaluator, Epoch>::training_settings(const std::string& mode)
{
    std::cerr
        << mode << ":\n"
        << " n. epochs: " << settings.n_epoch << "\n"
        << " n. updates per epoch: " << settings.n_updates_per_epoch << "\n"
        << " batch size: " << settings.batch_size << "\n"
        << " patience: " << settings.patience << "\n"
        << " max trials: " << settings.max_trials << "\n"
        << " lr decay: " << settings.lr_decay << "\n"
        << " reload params: " << (settings.reload_parameters ? "yes" : "no") << "\n"
        << " output path: " << settings.model_path << "\n"
        << " save params after each epoch: " << (settings.save_at_each_epoch ? "yes" : "no") << "\n"
        << std::endl;
}


//
// Dynamic Graph Optimizer
//


template <class Network, class DataType>
DynamicGraphEpoch<Network, DataType>::DynamicGraphEpoch(std::shared_ptr<Network> _network, dynet::Trainer& _trainer) :
    network(_network),
    trainer(_trainer)
{}

template <class Network, class DataType>
dynet::Expression DynamicGraphEpoch<Network, DataType>::labeled_loss(const DataType &data)
{
    return network->labeled_loss(data);
}

template <class Network, class DataType>
dynet::Expression DynamicGraphEpoch<Network, DataType>::unlabeled_loss(const DataType &data)
{
    return network->unlabeled_loss(data);
}

template <class Network, class DataType>
float DynamicGraphEpoch<Network, DataType>::forward_backward(
        dynet::ComputationGraph& cg,
        typename std::vector<DataType>::const_iterator begin_labelled_data,
        typename std::vector<DataType>::const_iterator end_labelled_data,
        typename std::vector<DataType>::const_iterator begin_unlabelled_data,
        typename std::vector<DataType>::const_iterator end_unlabelled_data
)
{
    std::vector<dynet::Expression> losses;

    for (; begin_labelled_data != end_labelled_data; ++begin_labelled_data)
        losses.push_back(labeled_loss(*begin_labelled_data));

    for (; begin_unlabelled_data != end_unlabelled_data; ++begin_unlabelled_data)
        losses.push_back(unlabeled_loss(*begin_unlabelled_data));

    if (losses.size() == 0u)
        throw std::runtime_error("No training data for the update");

    auto e_loss = (losses.size() == 1u ? losses.at(0u) : dynet::sum(losses) / (float) losses.size());

    const auto update_loss = as_scalar(cg.forward(e_loss));
    cg.backward(e_loss);

    return update_loss;
}

template <class Network, class DataType>
float DynamicGraphEpoch<Network, DataType>::optimize(
        std::vector<DataType> &labeled_data,
        std::vector<DataType> &unlabeled_data,
        const unsigned n_updates_per_epoch,
        const unsigned batch_size
)
{
    unsigned next_labeled_instance_index = labeled_data.size();
    unsigned next_unlabeled_instance_index = unlabeled_data.size();

    float epoch_loss = 0.f;
    for (unsigned update = 0; update < n_updates_per_epoch; ++update)
    {
        auto labeled_data_begin = labeled_data.end();
        auto labeled_data_end = labeled_data.end();
        auto unlabeled_data_begin = unlabeled_data.end();
        auto unlabeled_data_end = unlabeled_data.end();

        if (labeled_data.size() > 0)
        {
            if (labeled_data.size() < batch_size)
                throw std::runtime_error("Not enough labeled data");
            if (next_labeled_instance_index >= labeled_data.size())
            {
                std::random_shuffle(labeled_data.begin(), labeled_data.end());
                next_labeled_instance_index = 0u;
            }

            labeled_data_begin = labeled_data.begin() + next_labeled_instance_index;
            labeled_data_end = labeled_data.begin() + next_labeled_instance_index + batch_size;
            next_labeled_instance_index += batch_size;
        }

        if (unlabeled_data.size() > 0)
        {
            if (unlabeled_data.size() < batch_size)
                throw std::runtime_error("Not enough labeled data");
            if (next_unlabeled_instance_index >= unlabeled_data.size())
            {
                std::random_shuffle(unlabeled_data.begin(), unlabeled_data.end());
                next_unlabeled_instance_index = 0u;
            }

            unlabeled_data_begin = unlabeled_data.begin() + next_unlabeled_instance_index;
            unlabeled_data_end = unlabeled_data.begin() + next_unlabeled_instance_index + batch_size;
            next_unlabeled_instance_index += batch_size;
        }

        // build new computation graph
        dynet::ComputationGraph cg;
        network->new_graph(cg, true, true); // train & update

        // compute the loss of each instance in the batch
        const auto update_loss = forward_backward(
                cg,
                labeled_data_begin, labeled_data_end,
                unlabeled_data_begin, unlabeled_data_end
        );
        trainer.update();

        epoch_loss += update_loss;
    }
    return epoch_loss;
}


//
// Static Graph Optimizer
//


template <class Network, class DataType>
StaticGraphEpoch<Network, DataType>::StaticGraphEpoch(std::shared_ptr<Network> _network, dynet::Trainer& _trainer) :
        network(_network),
        trainer(_trainer)
{}

template <class Network, class DataType>
float StaticGraphEpoch<Network, DataType>::forward_backward(
        dynet::ComputationGraph& cg,
        typename std::vector<DataType>::const_iterator begin_labelled_data,
        typename std::vector<DataType>::const_iterator end_labelled_data,
        typename std::vector<DataType>::const_iterator begin_unlabelled_data,
        typename std::vector<DataType>::const_iterator end_unlabelled_data
)
{
    std::vector<dynet::Expression> losses;

    network->update_input(
            begin_labelled_data,
            end_labelled_data,
            begin_unlabelled_data,
            end_unlabelled_data
    );

    const auto e_loss = network->get_loss();
    const auto update_loss = as_scalar(cg.forward(e_loss));
    cg.backward(e_loss);

    return update_loss;
}

template <class Network, class DataType>
float StaticGraphEpoch<Network, DataType>::optimize(
        std::vector<DataType> &labeled_data,
        std::vector<DataType> &unlabeled_data,
        const unsigned n_updates_per_epoch,
        const unsigned batch_size
)
{
    unsigned next_labeled_instance_index = labeled_data.size();
    unsigned next_unlabeled_instance_index = unlabeled_data.size();

    // build new computation graph
    dynet::ComputationGraph cg;
    network->new_graph(cg, true, true); // train & update

    float epoch_loss = 0.f;
    for (unsigned update = 0; update < n_updates_per_epoch; ++update)
    {
        auto labeled_data_begin = labeled_data.end();
        auto labeled_data_end = labeled_data.end();
        auto unlabeled_data_begin = unlabeled_data.end();
        auto unlabeled_data_end = unlabeled_data.end();

        if (labeled_data.size() > 0)
        {
            if (labeled_data.size() < batch_size)
                throw std::runtime_error("Not enough labeled data");
            if (next_labeled_instance_index >= labeled_data.size())
            {
                std::random_shuffle(labeled_data.begin(), labeled_data.end());
                next_labeled_instance_index = 0u;
            }

            labeled_data_begin = labeled_data.begin() + next_labeled_instance_index;
            labeled_data_end = labeled_data.begin() + next_labeled_instance_index + batch_size;
            next_labeled_instance_index += batch_size;
        }

        if (unlabeled_data.size() > 0)
        {
            if (unlabeled_data.size() < batch_size)
                throw std::runtime_error("Not enough labeled data");
            if (next_unlabeled_instance_index >= unlabeled_data.size())
            {
                std::random_shuffle(unlabeled_data.begin(), unlabeled_data.end());
                next_unlabeled_instance_index = 0u;
            }

            unlabeled_data_begin = unlabeled_data.begin() + next_unlabeled_instance_index;
            unlabeled_data_end = unlabeled_data.begin() + next_unlabeled_instance_index + batch_size;
            next_unlabeled_instance_index += batch_size;
        }

        // compute the loss of each instance in the batch
        const auto update_loss = forward_backward(
                cg,
                labeled_data_begin, labeled_data_end,
                unlabeled_data_begin, unlabeled_data_end
        );
        trainer.update();

        epoch_loss += update_loss;
    }
    return epoch_loss;
}
}