#include "dytools/functions/masking.h"

#include <stdexcept>
#include <vector>

namespace dytools
{

dynet::Expression main_diagonal_mask(dynet::ComputationGraph& cg, const dynet::Dim dim, const float value)
{
    if (dim.ndims() != 2)
        throw std::runtime_error("The main diagonal mask can only be created for a matrix");
    if (dim.batch_elems() > 1)
        throw std::runtime_error("The main diagonal mask does not support multi-batching");

    const unsigned rows = dim.rows();
    const unsigned cols = dim.cols();
    const unsigned n_elems = (cols < rows ? cols : rows);

    std::vector<unsigned> indices;
    for (unsigned i = 0 ; i < n_elems ; ++i)
        indices.push_back(i + i * rows);

    std::vector<float> values(n_elems, value);

    return dynet::input(cg, dim, indices, values);
}


dynet::Expression all_but_first_vector_mask(dynet::ComputationGraph& cg, const unsigned size)
{
    std::vector<float> values(size, 1.f);
    values[0] = 0.f;

    return dynet::input(cg, {size}, values);
}

}