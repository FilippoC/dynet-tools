#include "dytools/builders/builder.h"

namespace dytools
{


void Builder::set_is_training(bool value)
{
    _is_training = value;
}

void Builder::train()
{
    set_is_training(true);
}

void Builder::eval()
{
    set_is_training(false);
}


bool Builder::is_training() const
{
    return _is_training;
}

bool Builder::is_evaluating() const
{
    return !_is_training;
}


}