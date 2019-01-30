#pragma once

namespace dytools
{

struct Builder
{
    bool _is_training = false;

    virtual void set_is_training(bool value);

    void train();
    void eval();
    bool is_training() const;
    bool is_evaluating() const;
};

}