#include "dytools/networks/dependency.h"

#include <limits>
#include <dytools/training.h>
#include <dytools/loss/dependency.h>
#include <dytools/algorithms/dependency-parser.h>

namespace dytools
{

DependencyNetwork::DependencyNetwork(dynet::ParameterCollection& pc, const DependencySettings& settings, std::shared_ptr<dynet::Dict> token_dict, std::shared_ptr<dynet::Dict> char_dict, std::shared_ptr<dynet::Dict> tagger_dict) :
        settings(settings),
        local_pc(pc.add_subcollection("parser")),
        embeddings(local_pc, settings.embeddings, token_dict, char_dict),
        first_bilstm(local_pc, settings.first_bilstm, embeddings.output_rows()),
        second_bilstm(local_pc, settings.second_bilstm, first_bilstm.output_rows()),
        tagger(local_pc, settings.tagger, tagger_dict, first_bilstm.output_rows()),
        biaffine(local_pc, settings.biaffine, second_bilstm.output_rows(), true)
{}

void DependencyNetwork::new_graph(dynet::ComputationGraph& cg, bool update)
{
    embeddings.new_graph(cg, update);
    first_bilstm.new_graph(cg, update);
    second_bilstm.new_graph(cg, update);
    tagger.new_graph(cg, update);
    biaffine.new_graph(cg, update);

    _cg = &cg;
}

void DependencyNetwork::set_is_training(bool value)
{
    Builder::set_is_training(value);
    embeddings.set_is_training(value);
    first_bilstm.set_is_training(value);
    second_bilstm.set_is_training(value);
    tagger.set_is_training(value);
    biaffine.set_is_training(value);
}

std::pair<dynet::Expression, dynet::Expression> DependencyNetwork::full_logits(const ConllSentence& sentence)
{
    const auto embs = embeddings(sentence);
    const auto embs1 = first_bilstm(embs);
    const auto embs2 = second_bilstm(embs1);

    const auto tag_weights = tagger.full_logits(dynet::concatenate_cols(embs1));
    const auto arc_weights = biaffine(embs2);

    return std::make_pair(tag_weights, arc_weights);
}


DependencyParserTrainer::DependencyParserTrainer(const TrainingSettings& settings, std::shared_ptr<DependencyNetwork> _network) :
    dytools::Training<dytools::DependencyNetwork, dytools::ConllSentence>(settings, _network)
{}

dynet::Expression DependencyParserTrainer::compute_loss(const dytools::ConllSentence& sentence)
{
    const unsigned size = sentence.size() + 1;
    dynet::ComputationGraph& cg = *(network->_cg);
    const auto p_weights = network->full_logits(sentence);

    std::vector<unsigned> gold_idx;
    gold_idx.push_back(0u); // root word, will be masked
    for (unsigned i = 0u; i < sentence.size(); ++i)
    {
        const unsigned head = sentence.at(i).head == i ? 0 : sentence.at(i).head + 1;
        gold_idx.push_back(head);
    }

    return head_neg_log_likelihood(p_weights.second, gold_idx);

}
float DependencyParserTrainer::evaluate(const std::vector<dytools::ConllSentence>& data)
{
    auto n_correct{0.f};
    auto total{0.f};
    for (auto const& sentence : data)
    {
        dynet::ComputationGraph cg;
        network->new_graph(cg);

        const auto e_weights = network->full_logits(sentence).second;
        const auto v_weights = as_vector(cg.forward(e_weights));

        const auto heads = non_projective_dependency_parser(sentence.size(), v_weights);
        n_correct += uas(sentence, heads, false);
        total += sentence.size();
    }

    return n_correct / total;
}

}