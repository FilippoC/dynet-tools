#include "dytools/masked_sequence.h"

namespace dytools
{

MaskedSequence::MaskedSequence(const dynet::Expression& embs, const dynet::Expression& mask) :
        embs(embs), mask(mask),
        log_mask_init(false), t_mask_init(false), t_log_mask_init(false)
{}
/*
MaskedSequence::MaskedSequence(MaskedSequence&& o) noexcept :
        embs(std::move(o.embs)),
        mask(std::move(o.mask)),
        log_mask(std::move(o.log_mask)),
        t_mask(std::move(o.t_mask)),
        t_log_mask(std::move(o.t_log_mask)),
        log_mask_init(std::move(o.log_mask_init)),
        t_mask_init(std::move(o.t_mask_init)),
        t_log_mask_init(std::move(o.t_log_mask_init))
{}
*/
void MaskedSequence::update_embs(const dynet::Expression& _embs)
{
    embs = _embs;
}

dynet::Expression MaskedSequence::get_embs()
{
    return embs;
}

dynet::Expression MaskedSequence::get_mask()
{
    return mask;
}

dynet::Expression MaskedSequence::get_log_mask()
{
    return dynet::log(mask);
    if (!log_mask_init)
    {
        log_mask_init = true;
        log_mask = dynet::log(mask);
    }
    return log_mask;
}

dynet::Expression MaskedSequence::get_t_mask()
{
    return dynet::transpose(mask);
    if (!t_mask_init)
    {
        t_mask_init = true;
        t_mask = dynet::transpose(mask);
    }
    return t_mask;
}

dynet::Expression MaskedSequence::get_t_log_mask()
{
    return dynet::transpose(dynet::log(mask));
    if(!t_log_mask_init)
    {
        t_log_mask_init = true;
        t_log_mask = dynet::transpose(get_log_mask());
    }
    return t_log_mask;
}

}
