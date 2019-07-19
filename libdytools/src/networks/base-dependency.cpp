#include "dytools/networks/base-dependency.h"
#include <dytools/loss/dependency.h>

namespace dytools
{

BaseDependencyNetwork::BaseDependencyNetwork(
        dynet::ParameterCollection& pc,
        const BaseDependencySettings& settings,
        std::shared_ptr<dytools::Dict> tagger_dict,
        std::shared_ptr<dytools::Dict> label_dict,
        unsigned embeddings_size
) :
        settings(settings),
        local_pc(pc.add_subcollection("basedepparser")),
        first_bilstm(local_pc, settings.first_bilstm, embeddings_size),
        second_bilstm(local_pc, settings.second_bilstm, first_bilstm.output_rows()),
        tagger(local_pc, settings.tagger, tagger_dict, first_bilstm.output_rows()),
        biaffine(local_pc, settings.biaffine, second_bilstm.output_rows(), true),
        biaffine_tagger(local_pc, settings.biaffine_tagger, second_bilstm.output_rows(), label_dict, true)
{}

void BaseDependencyNetwork::new_graph(dynet::ComputationGraph& cg, bool training, bool update)
{
    first_bilstm.new_graph(cg, training, update);
    second_bilstm.new_graph(cg, training, update);
    tagger.new_graph(cg, training, update);
    biaffine.new_graph(cg, training, update);
    biaffine_tagger.new_graph(cg, training, update);

    _cg = &cg;
}

std::tuple<dynet::Expression, dynet::Expression, dynet::Expression> BaseDependencyNetwork::logits(const ConllSentence &sentence)
{
    const auto embs = get_embeddings(sentence);
    const auto embs1 = first_bilstm(embs);
    const auto embs2 = second_bilstm(embs1);

    const auto tag_weights = tagger.full_logits(dynet::concatenate_cols(embs1));
    const auto arc_weights = biaffine(embs2);

    std::vector<unsigned> heads;
    for (unsigned mod = 0 ; mod < sentence.size() ; ++mod)
    {
        auto const& token = sentence.at(mod);
        if (token.head == mod)
            heads.push_back(0u);
        else
            heads.push_back(token.head + 1u);

    }
    const auto label_weights = biaffine_tagger.dependency_tagger(embs2, heads);

    return std::make_tuple(tag_weights, arc_weights, label_weights);
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


dynet::Expression BaseDependencyNetwork::labeled_loss(const dytools::ConllSentence &sentence)
{
    const auto t_weights = logits(sentence);
    //const auto tag_weights = std::get<0>(t_weights);
    const auto arc_weights = std::get<1>(t_weights);
    const auto labels_weight = std::get<2>(t_weights);

    std::vector<unsigned> gold_labels;
    std::vector<unsigned> gold_heads;
    gold_heads.push_back(0u); // root word, will be masked
    for (unsigned i = 0u; i < sentence.size(); ++i)
    {
        const auto& token = sentence.at(i);

        const unsigned head = token.head == i ? 0 : token.head + 1;
        gold_heads.push_back(head);

        gold_labels.push_back(biaffine_tagger.dict->convert(token.deprel));
    }

    const auto label_loss = dynet::sum_batches(
        dynet::pickneglogsoftmax(
            labels_weight,
            gold_labels
        )
    );
    const auto arc_loss = head_neg_log_likelihood(arc_weights, gold_heads);

    return label_loss + arc_loss;
}

}