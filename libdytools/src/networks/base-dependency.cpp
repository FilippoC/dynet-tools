#include "dytools/networks/base-dependency.h"

namespace dytools
{

BaseDependencyNetwork::BaseDependencyNetwork(
        dynet::ParameterCollection& pc,
        const BaseDependencySettings& settings,
        std::shared_ptr<dytools::Dict> tagger_dict,
        unsigned embeddings_size
) :
        settings(settings),
        local_pc(pc.add_subcollection("basedepparser")),
        first_bilstm(local_pc, settings.first_bilstm, embeddings_size),
        second_bilstm(local_pc, settings.second_bilstm, first_bilstm.output_rows()),
        tagger(local_pc, settings.tagger, tagger_dict, first_bilstm.output_rows()),
        biaffine(local_pc, settings.biaffine, second_bilstm.output_rows(), true)
{}

void BaseDependencyNetwork::new_graph(dynet::ComputationGraph& cg, bool update)
{
    first_bilstm.new_graph(cg, update);
    second_bilstm.new_graph(cg, update);
    tagger.new_graph(cg, update);
    biaffine.new_graph(cg, update);

    _cg = &cg;
}

void BaseDependencyNetwork::set_is_training(bool value)
{
    Builder::set_is_training(value);

    first_bilstm.set_is_training(value);
    second_bilstm.set_is_training(value);
    tagger.set_is_training(value);
    biaffine.set_is_training(value);
}

std::pair<dynet::Expression, dynet::Expression> BaseDependencyNetwork::logits(const ConllSentence &sentence)
{
    const auto embs = get_embeddings(sentence);
    const auto embs1 = first_bilstm(embs);
    const auto embs2 = second_bilstm(embs1);

    const auto tag_weights = tagger.full_logits(dynet::concatenate_cols(embs1));
    const auto arc_weights = biaffine(embs2);

    return std::make_pair(tag_weights, arc_weights);
}

dynet::Expression BaseDependencyNetwork::dependency_logits(const ConllSentence &sentence)
{
    const auto embs = get_embeddings(sentence);
    const auto embs1 = first_bilstm(embs);
    const auto embs2 = second_bilstm(embs1);
    const auto arc_weights = biaffine(embs2);
    return arc_weights;
}

dynet::Expression BaseDependencyNetwork::tag_logits(const ConllSentence &sentence)
{
    const auto embs = get_embeddings(sentence);
    const auto embs1 = first_bilstm(embs);
    const auto tag_weights = tagger.full_logits(dynet::concatenate_cols(embs1));

    return tag_weights;
}


}