#pragma once

#include "dynet/dict.h"
#include <boost/serialization/vector.hpp>
#include <boost/serialization/unordered_map.hpp>

namespace dytools
{

struct Dict : public dynet::Dict
{
    using dynet::Dict::size;
    using dynet::Dict::freeze;
    using dynet::Dict::is_frozen;
    using dynet::Dict::contains;

    template<class Archive>
    void serialize(Archive &ar, const unsigned int)
    {
        ar & frozen;
        ar & map_unk;
        ar & unk_id;
        ar & words_;
        ar & d_;
    }
};

}