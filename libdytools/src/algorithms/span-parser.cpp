#include "dytools/algorithms/span-parser.h"

#include <limits>

namespace dytools
{

Tree binary_span_parser(const unsigned size, const std::vector<float> &weights)
{
    std::vector<float> cst_weights(size * size, 0.f);
    std::vector<unsigned> back_ptr(size * size);

    // bottom-up constituency weight computation
    for (unsigned length = 1 ; length < size ; ++length)
    {
        for (unsigned left = 0 ; left < size - length; ++ left)
        {
            const unsigned right = left + length;
            float max_weight = -std::numeric_limits<float>::infinity();
            int max_k = 0;
            for (unsigned k = left ; k < right ; ++k)
            {
                const float w = cst_weights.at(left + k * size) + cst_weights.at(k + 1 + right * size);
                if (w > max_weight)
                {
                    max_weight = w;
                    max_k = k;
                }
            }
            //std::cerr << "\n";
            const float local_weight = weights.at(left + right * size);
            cst_weights.at(left + right * size) = local_weight +max_weight;
            back_ptr.at(left + right * size) = max_k;
        }
    }

    // top-down linear-time reconstruction
    std::vector<Span> queue;
    Tree tree;

    queue.emplace_back(0u, size - 1u);
    while(queue.size() > 0)
    {
        // remove from the queue and add to the tree
        const auto span = queue.back();
        queue.pop_back();

        if (span.first < span.second)
            tree.emplace(span);

        if (span.first + 1 < span.second)
        {
            const auto best_k = back_ptr.at(span.first + span.second * size);
            const auto left_antecedent = std::make_pair(span.first, best_k);
            const auto right_antecedent = std::make_pair(best_k + 1, span.second);

            queue.push_back(left_antecedent);
            queue.push_back(right_antecedent);
        }
    }

    return tree;
}

}