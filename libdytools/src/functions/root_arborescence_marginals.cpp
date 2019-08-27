#include "dytools/functions/rooted_arborescence_marginals.h"

#include "dytools/utils.h"
#include "dytools/functions/masking.h"

namespace dytools
{

dynet::Expression rooted_arborescence_marginals(dynet::ComputationGraph& cg, const dynet::Expression& arc_weights, const std::vector<unsigned>& n_vertices)
{
    return rooted_arborescence_marginals(cg, arc_weights, &n_vertices);
}
dynet::Expression rooted_arborescence_marginals(dynet::ComputationGraph& cg, const dynet::Expression& arc_weights, const std::vector<unsigned>* n_vertices)
{
    const unsigned n_max_vertices = arc_weights.dim().cols();

    const auto mask_diagonal = dytools::main_diagonal_mask(cg, {n_max_vertices, n_max_vertices});
    const auto mask_first_row = all_but_first_vector_mask(cg, n_max_vertices);
    const auto mask_first_col = dynet::transpose(mask_first_row);
    const auto root_weight = dynet::input(cg, {n_max_vertices, n_max_vertices}, {0u}, {1.f});

    auto exp_weights = dynet::exp(arc_weights);

    // mask
    dynet::Expression e_t_vertices_mask;
    if (n_vertices != nullptr)
    {
        if (n_vertices->size() != arc_weights.dim().batch_elems())
            throw std::runtime_error("Batch size does not match the graph size vector");

        std::vector<unsigned> v_mask_indices;
        for (unsigned batch = 0 ; batch < n_vertices->size() ; ++batch)
            for (unsigned i = 0 ; i < n_vertices->at(batch) ; ++i)
                v_mask_indices.push_back(i + batch * n_max_vertices);
        std::vector<float> v_mask_values(v_mask_indices.size(), 1.f);
        auto e_vertices_mask = dynet::input(cg, dynet::Dim({n_max_vertices}, n_vertices->size()), v_mask_indices, v_mask_values);
        e_t_vertices_mask = dynet::transpose(e_vertices_mask);

        // mask network of invalid arcs
        exp_weights = dynet::cmult(exp_weights, e_vertices_mask);
        exp_weights = dynet::cmult(exp_weights, e_t_vertices_mask);

        // add fake arc from the root word
        std::vector<unsigned> v_fake_arc_indices;
        for (unsigned batch = 0 ; batch < n_vertices->size() ; ++batch)
        {
            const unsigned vertices_in_batch = n_vertices->at(batch);
            for (unsigned i = vertices_in_batch; i < n_max_vertices ; ++i)
                v_fake_arc_indices.push_back(i * n_max_vertices + batch * n_max_vertices * n_max_vertices);
        }
        std::vector<float> v_fake_arc_values(v_fake_arc_indices.size(), 1.f);
        auto e_fake_arcs = dynet::input(cg, dynet::Dim({n_max_vertices, n_max_vertices}, n_vertices->size()), v_fake_arc_indices, v_fake_arc_values);

        exp_weights = exp_weights + e_fake_arcs;
    }

    const auto col_sum = dynet::sum_dim(exp_weights, {0});
    const auto col_sum_as_diag = dynet::cmult(col_sum, mask_diagonal);

    const auto laplacian = col_sum_as_diag - exp_weights;
    const auto laplacian2 = root_weight + dynet::cmult(laplacian, mask_first_row);

    // the inverse operation does not support GPU neither minibatches
    dynet::Expression inv_laplacian;
    if (laplacian2.dim().batch_elems() == 1)
    {
        inv_laplacian = dytools::force_cpu(dynet::inverse, laplacian2); // op not available on GPU
    }
    else
    {
        std::vector<dynet::Expression> all_inv_laplacian;
        all_inv_laplacian.reserve(laplacian2.dim().batch_elems());
        for (unsigned b = 0 ; b < laplacian2.dim().batch_elems() ; ++b)
        {
            all_inv_laplacian.push_back(
                    dytools::force_cpu(dynet::inverse, dynet::pick_batch_elem(laplacian2, b))
            );
        }
        inv_laplacian = dynet::concatenate_to_batch(all_inv_laplacian);
    }


    const auto inv_laplacian_diag = dynet::cmult(inv_laplacian, mask_diagonal);

    const auto output1 = dynet::cmult(
            exp_weights,
            dynet::transpose(dynet::sum_dim(inv_laplacian_diag, {0}))
    );
    const auto output2 = dynet::cmult(exp_weights, dynet::transpose(inv_laplacian));

    auto marginals = dynet::cmult(output1, mask_first_col) - dynet::cmult(output2, mask_first_row);


    if (n_vertices != nullptr)
    {
        marginals = dynet::cmult(marginals, e_t_vertices_mask);
    }

    return marginals;
}

}