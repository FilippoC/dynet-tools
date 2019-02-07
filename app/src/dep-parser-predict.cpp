#include <iostream>
#include <unistd.h>
#include <string>

#include "dynet/init.h"
#include "dynet/io.h"

#include "dytools/networks/dependency.h"
#include "dytools/io.h"
#include "dytools/algorithms/tagger.h"
#include "dytools/algorithms/dependency-parser.h"

int main(int argc, char** argv)
{
    // initialize dynet and read cmd line args
    dynet::initialize(argc, argv);
    if (argc != 3)
    {
        std::cerr << "usage: " << argv[0] << " MODEL_PATH DATA_PATH\n";
        return 1;
    }
    std::string model_path(argv[1]);
    std::string data_path(argv[2]);



    std::cerr << "Reading data..." << std::endl;
    std::vector<dytools::ConllSentence> data;
    dytools::read(data_path, data);


    std::cerr << "Reading network settings..." << std::endl;
    auto token_dict = std::make_shared<dytools::Dict>();
    auto char_dict = std::make_shared<dytools::Dict>();
    auto tag_dict = std::make_shared<dytools::Dict>();
    dytools::DependencySettings network_settings;
    {
        dytools::TextFileLoader in(model_path + ".settings");

        // read dictionnaries
        in.load(*token_dict);
        in.load(*char_dict);
        in.load(*tag_dict);

        // read network settings
        in.load(network_settings);

        in.close();
    }


    std::cerr << "Building network..." << std::endl;
    dynet::ParameterCollection pc;
    dytools::DependencyNetwork network(pc, network_settings, token_dict, char_dict, tag_dict);
    network.eval();


    std::cerr << "Loading network parameters..." << std::endl;
    {
        dynet::TextFileLoader s(model_path);
        s.populate(network.local_pc);
    }


    std::cerr << "Decoding..." << std::endl;
    for (auto& sentence : data)
    {
        dynet::ComputationGraph cg;
        network.new_graph(cg);

        const auto p_logis = network.logits(sentence);
        const auto e_tag_weights = p_logis.first;
        const auto e_arc_weights = p_logis.second;

        const auto last = e_arc_weights.i > e_tag_weights.i ? e_arc_weights : e_tag_weights;
        cg.forward(last);
        const auto v_tag_weights = as_vector(cg.get_value(e_tag_weights));
        const auto v_arc_weights = as_vector(cg.get_value(e_arc_weights));

        // decode tags
        const auto tags = dytools::tagger(sentence.size(), v_tag_weights);

        // decode dependency tree
        const auto heads = dytools::non_projective_dependency_parser(sentence.size(), v_arc_weights);

        // update and print the data
        sentence.update_tags(tags);
        sentence.update_heads(heads);
    }
    dytools::write(std::cout, data);
}