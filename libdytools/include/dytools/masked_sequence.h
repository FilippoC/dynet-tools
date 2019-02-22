#pragma once

#include "dynet/expr.h"

namespace dytools
{

class MaskedSequence
{
private:
    dynet::Expression embs;
    dynet::Expression mask;

    dynet::Expression log_mask;
    dynet::Expression t_mask;
    dynet::Expression t_log_mask;

    bool log_mask_init, t_mask_init, t_log_mask_init;

public:
    MaskedSequence(const dynet::Expression& embs, const dynet::Expression& mask);

    // custom move constructor so we can return this object from a function
    //MaskedSequence(MaskedSequence&& o) noexcept;

    // no copy&empty constructor
    //MaskedSequence(const MaskedSequence&) = delete;
    MaskedSequence() = delete;

    void update_embs(const dynet::Expression& _embs);

    dynet::Expression get_embs();
    dynet::Expression get_mask();
    dynet::Expression get_log_mask();
    dynet::Expression get_t_mask();
    dynet::Expression get_t_log_mask();
};

}
