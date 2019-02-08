#include "dytools/networks/dependency.h"

#include <limits>
#include <dytools/training.h>
#include <dytools/algorithms/dependency-parser.h>

namespace dytools
{

DependencyNetwork::DependencyNetwork(
        dynet::ParameterCollection& pc,
        const DependencySettings& settings,
        std::shared_ptr<dytools::Dict> token_dict,
        std::shared_ptr<dytools::Dict> char_dict,
        std::shared_ptr<dytools::Dict> tagger_dict,
        std::shared_ptr<dytools::Dict> label_dict
) :
        BaseDependencyNetwork(pc, settings, tagger_dict, label_dict, settings.embeddings.output_rows()),
        settings(settings),
        embeddings(local_pc, settings.embeddings, token_dict, char_dict)
{}

void DependencyNetwork::new_graph(dynet::ComputationGraph& cg, bool update)
{
    BaseDependencyNetwork::new_graph(cg, update);

    embeddings.new_graph(cg, update);
    _cg = &cg;
}

void DependencyNetwork::set_is_training(bool value)
{
    BaseDependencyNetwork::set_is_training(value);

    embeddings.set_is_training(value);
}

std::vector<dynet::Expression> DependencyNetwork::get_embeddings(const dytools::ConllSentence &sentence)
{
    return embeddings(sentence);
}


unsigned DependencyNetwork::get_embeddings_size() const
{
    return embeddings.output_rows();
}

float DependencyParserEvaluator::operator()(BaseDependencyNetwork* network, const std::vector<dytools::ConllSentence>& data) const
{
    auto n_correct = 0.f;
    auto total = 0.f;
    for (auto const& sentence : data)
    {
        dynet::ComputationGraph cg;
        cg.set_immediate_compute(true);
        cg.set_check_validity(true);

        network->new_graph(cg);

        const auto e_weights = std::get<1>(network->logits(sentence));
        const auto v_weights = as_vector(cg.forward(e_weights));

        const auto heads = non_projective_dependency_parser(sentence.size(), v_weights);
        n_correct += uas(sentence, heads, false);
        total += sentence.size();
    }

    const float score =  n_correct / total;
    std::cerr << "Dev evaluation: " << score << "\t" << n_correct << "/" << total << "\n";
    return score;
}

}
