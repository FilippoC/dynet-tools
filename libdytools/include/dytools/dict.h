#pragma once

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

#include <boost/serialization/vector.hpp>
#include <boost/serialization/unordered_map.hpp>

#include "dytools/utils.h"
#include "dynet/dict.h"

namespace dytools
{

struct Dict
{
    const std::string unk_str = "*UNK*";
    const std::string num_str = "*NUM*";

    bool has_unk;
    bool has_num;
    bool lowercase;

    unsigned unk_id = 0;
    unsigned num_id = 0;

    std::vector<std::string> id_to_word;
    std::unordered_map<std::string, unsigned> word_to_id;

    Dict(bool _lowercase=false, bool _has_num=false, bool _has_unk=false);

    std::string normalize(const std::string& word) const;
    unsigned to_id(const std::string& _word) const;
    std::string to_string(const unsigned id) const;

    void add(const std::string& _word);

    unsigned size() const;
};


}