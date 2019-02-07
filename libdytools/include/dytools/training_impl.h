#pragma once

#include <chrono>
#include <limits>
#include <dynet/io.h>

namespace dytools
{

template <class Network, class DataType>
Training<Network, DataType>::Training(std::shared_ptr<Network> _network) :
    network(std::move(_network))
{}


template <class Network, class DataType>
Training<Network, DataType>::Training(const TrainingSettings& settings, std::shared_ptr<Network> _network) :
settings(settings),
network(std::move(_network))
{}

template <class Network, class DataType>
dynet::Expression Training<Network, DataType>::compute_loss(typename std::vector<DataType>::const_iterator begin_data, typename std::vector<DataType>::const_iterator end_data)
{
    std::vector<dynet::Expression> losses;
    for (; begin_data != end_data ; ++begin_data)
        losses.push_back(compute_loss(*begin_data));

    if (losses.size() == 0u)
        throw std::runtime_error("No training data for the update");
    else if (losses.size() == 1u)
        return losses.at(0u);
    else
        return dynet::sum(losses) / (float) losses.size();
}

template <class Network, class DataType>
dynet::Expression Training<Network, DataType>::compute_loss(const DataType&)
{
    throw std::runtime_error("Not implemented: compute_loss()");
}

template <class Network, class DataType>
float Training<Network, DataType>::evaluate(const DataType&)
{
    throw std::runtime_error("Not implemented: evaluate()");
}

template <class Network, class DataType>
float Training<Network, DataType>::forward_backward(dynet::ComputationGraph& cg, typename std::vector<DataType>::const_iterator begin_data, typename std::vector<DataType>::const_iterator end_data)
{
    auto e_loss = compute_loss(begin_data, end_data);
    const auto update_loss = as_scalar(cg.forward(e_loss));
    cg.backward(e_loss);

    return update_loss;
}

template <class Network, class DataType>
void Training<Network, DataType>::optimize(dynet::Trainer& trainer, std::vector<DataType>& train_data, const std::vector<DataType>& dev_data)
{
    std::cerr
        << "Training!\n"
        << " train dataset size: " << train_data.size() << "\n"
        << " dev dataset size: " << dev_data.size() << "\n"
        << std::endl;

    unsigned next_instance_index = train_data.size();

    float best_dev_score = -std::numeric_limits<float>::infinity();
    unsigned best_dev_epoch = 0u;
    unsigned n_trials = 0u;
    unsigned n_epoch_without_improvement = 0u;
    for (unsigned epoch = 0; epoch < settings.n_epoch; ++epoch)
    {
        std::cerr << "\nEpoch " << epoch << "/" << settings.n_epoch << std::endl;

        network->train();
        float epoch_loss = 0.f;
        auto start_epoch = std::chrono::steady_clock::now();
        for (unsigned update = 0; update < settings.n_updates_per_epoch; ++update)
        {
            if (next_instance_index >= train_data.size())
            {
                std::random_shuffle(train_data.begin(), train_data.end());
                next_instance_index = 0u;
            }

            auto data_begin = train_data.begin() + next_instance_index;
            auto data_end = train_data.begin() + next_instance_index + settings.batch_size;
            next_instance_index += settings.batch_size;

            // build new computation graph
            dynet::ComputationGraph cg;
            network->new_graph(cg);

            // compute the loss of each instance in the batch
            const auto update_loss = forward_backward(cg, data_begin, data_end);
            trainer.update();

            epoch_loss += update_loss;
        }
        auto end_epoch = std::chrono::steady_clock::now();
        std::cerr
                << "Epoch loss: " << epoch_loss
                << "\t/\tDuration: "
                << std::chrono::duration_cast<std::chrono::seconds>(end_epoch - start_epoch).count()
                << std::endl;

        // evaluate on dev data
        network->eval();
        float dev_score = evaluate(dev_data);
        std::cerr
            << "Dev evaluation: "
            << dev_score
            << std::endl;
        if (dev_score > best_dev_score)
        {
            std::cerr << "dev score as increased: " << dev_score << " > " << dev_score << std::endl;
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

template <class Network, class DataType>
float Training<Network, DataType>::evaluate(const std::vector<DataType>& data)
{
    float total = 0.f;
    network->eval();

    const auto start_dev = std::chrono::steady_clock::now();
    for (auto const &instance : data)
    {
        std::cerr << "EVAL\n";
        total += evaluate(instance);
    }
    const auto end_dev = std::chrono::steady_clock::now();

    const auto dev_score = total / (float) data.size();
    std::cerr
            << "Dev score: " << dev_score
            << "\t/\tDuration: " << std::chrono::duration_cast<std::chrono::seconds>(end_dev - start_dev).count()
            << std::endl;
    return dev_score;
}


template <class Network, class DataType>
void Training<Network, DataType>::save()
{
    if (settings.model_path.size() > 0)
    {
        std::cerr << "Saving model to: " << settings.model_path << std::endl;
        dynet::TextFileSaver s(settings.model_path);
        s.save(network->local_pc);
    }
}

template <class Network, class DataType>
void Training<Network, DataType>::load()
{
    if (settings.model_path.size() > 0)
    {
        std::cerr << "Loading model from: " << settings.model_path << std::endl;
        dynet::TextFileLoader s(settings.model_path);
        s.populate(network->local_pc);
    }
}

}