#include <iostream>
#include <unistd.h>
#include <string>

#include "dynet/init.h"

#include "dytools/training.h"
#include "dytools/networks/dependency.h"
#include "dytools/io.h"

bool read_command_line_args(int& argc, char**& argv, dytools::TrainingSettings& training_settings, dytools::DependencySettings& network_settings, std::string& train_path, std::string& dev_path);
void command_line_help(std::ostream& os, const std::string name);


int main(int argc, char** argv)
{
    // settings
    dytools::TrainingSettings training_settings;
    dytools::DependencySettings network_settings;
    std::string train_path;
    std::string dev_path;


    // processing the command line arguments
    dynet::initialize(argc, argv);
    if (argc == 1 || !read_command_line_args(argc, argv, training_settings, network_settings, train_path, dev_path))
    {
        command_line_help(std::cerr, std::string(argv[0]));
        return 1;
    }


    std::cerr << "Reading data..." << std::endl;
    std::vector<dytools::ConllSentence> train_data;
    std::vector<dytools::ConllSentence> dev_data;
    dytools::read(train_path, train_data);
    if (dev_path.size() > 0)
        dytools::read(dev_path, dev_data);
    else
        std::cerr << "WARNING: no validation data!" << std::endl;


    std::cerr << "Building dictionnariess..." << std::endl;
    dytools::DictSettings voc_settings;
    voc_settings.num_word = "*NUM*";
    voc_settings.unk_word = "*UNK*";
    voc_settings.lowercase = true;
    auto token_dict = dytools::build_conll_token_dict(voc_settings, train_data.begin(), train_data.end());
    auto char_dict = dytools::build_conll_char_dict(voc_settings, train_data.begin(), train_data.end());

    auto tag_dict = dytools::build_conll_tag_dict(train_data.begin(), train_data.end());
    auto label_dict = dytools::build_conll_label_dict(train_data.begin(), train_data.end());

    std::cerr << "Saving network settings..." << std::endl;
    {
        dytools::TextFileSaver out(training_settings.model_path + ".settings");

        // save dictionnaries
        out.save(*token_dict);
        out.save(*char_dict);
        out.save(*tag_dict);
        out.save(*label_dict);

        // save network settings
        out.save(network_settings);

        out.close();
    }

    std::cerr << "Building network..." << std::endl;
    dynet::ParameterCollection pc;
    auto network = std::make_shared<dytools::DependencyNetwork>(pc, network_settings, token_dict, char_dict, tag_dict, label_dict);


    std::cerr << "Training..." << std::endl;
    dynet::AdamTrainer optimizer(pc);
    dytools::Training<dytools::DependencyNetwork, dytools::ConllSentence, dytools::DependencyParserEvaluator> trainer(training_settings, network);
    trainer.optimize_supervised(optimizer, train_data, dev_data);


    std::cerr << "Done!" << std::endl;
    return 0;
}


bool read_command_line_args(int& argc, char**& argv, dytools::TrainingSettings& training_settings, dytools::DependencySettings& network_settings, std::string& train_path, std::string& dev_path)
{
    // we use a flag to set true, so force the default to false
    network_settings.biaffine.mod_bias = false;
    network_settings.tagger.output_bias = false;

    opterr = 0;
    int opt;
    while ((opt = getopt (argc, argv, "v:d:e:u:b:w:c:l:p:mo:t")) != -1)
    {
        std::unique_ptr<std::istringstream> iss;
        char c;
        switch (opt)
        {
            // training options
            case 'o':
                training_settings.model_path = std::string(optarg);
                break;
            case 'd':
                train_path = std::string(optarg);
                break;
            case 'v':
                dev_path = std::string(optarg);
                break;
            case 'e':
                iss.reset(new std::istringstream(optarg));
                *iss >> training_settings.n_epoch;
                break;
            case 'u':
                iss.reset(new std::istringstream(optarg));
                *iss >> training_settings.n_updates_per_epoch;
                break;
            case 'b':
                iss.reset(new std::istringstream(optarg));
                *iss >> training_settings.batch_size;
                break;

            // network options
            case 'w':
                // format: DIM
                iss.reset(new std::istringstream(optarg));
                *iss >> network_settings.embeddings.token_embeddings.dim;
                network_settings.embeddings.use_token_embeddings = (network_settings.embeddings.token_embeddings.dim > 0);
                break;
            case 'c':
                // format: DIM,STACK,LAYERS,HIDDEN
                iss.reset(new std::istringstream(optarg));
                *iss
                    >> network_settings.embeddings.char_embeddings.dim
                    >> c
                    >> network_settings.embeddings.char_embeddings.bilstm.stacks
                    >> c
                    >> network_settings.embeddings.char_embeddings.bilstm.layers
                    >> c
                    >> network_settings.embeddings.char_embeddings.bilstm.dim
                    ;
                network_settings.embeddings.use_char_embeddings = (network_settings.embeddings.char_embeddings.dim > 0);
                break;
            case 'l':
                // format STACK1,LAYER1,HIDDEN1,STACK2,LAYER2,HIDDEN2
                iss.reset(new std::istringstream(optarg));
                *iss
                    >> network_settings.first_bilstm.stacks
                    >> c
                    >> network_settings.first_bilstm.layers
                    >> c
                    >> network_settings.first_bilstm.dim
                    >> c
                    >> network_settings.second_bilstm.stacks
                    >> c
                    >> network_settings.second_bilstm.layers
                    >> c
                    >> network_settings.second_bilstm.dim
                    ;
                break;
            case 'p':
                // format: DIM,DIM
                iss.reset(new std::istringstream(optarg));
                *iss
                    >> network_settings.tagger.dim
                    >> c
                    >> network_settings.biaffine.proj_size
                    ;
                break;
            case 'm':
                network_settings.biaffine.mod_bias = true;
                break;
            case 't':
                network_settings.tagger.output_bias = true;
                break;
            case '?':
            default:
                return false;
        }
    }

    // it's only ok if we read all the arguments
    return optind >= argc;
}

void command_line_help(std::ostream& os, const std::string name)
{
    os
        << "usage: " << name << "\n"
        << " -d PATH\ttraining data\n"
        << " -v PATH\tvalidation data\n"
        << " -o PATH\tpath where to save the model\n"
        << "\n"
        << " -e NUM\tnumber of epochs\n"
        << " -u NUM\tnumber of updates per epoch\n"
        << " -b SIZE\tmini-batch size\n"
        << "\n"
        << " -w DIM\tdimension of word embeddings\n"
        << " -c DIM,STACK,LAYER,HIDDEN\tdimension of character embeddings and associated BiLSTM\n"
        << " -l STACK1,LAYER1,HIDDEN1,STACK2,LAYER2,HIDDEN2\tdimension of BiLSTMs\n"
        << " -p DIM,DIM\tdimension of the projection for the tagger and the biaffine network\n"
        << "\n"
        << " -m\tuse modifier bias in the biaffine network\n"
        << " -t\tuse bias in the tagger output\n"
        ;
}