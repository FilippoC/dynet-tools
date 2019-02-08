#pragma once

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

#include <boost/serialization/vector.hpp>
#include <boost/serialization/unordered_map.hpp>

#include "dytools/utils.h"

namespace dytools
{

struct DictSettings
{
    bool lowercase = false;
    std::string unk_word = "";
    std::string num_word = "";

    template<class Archive>
    void serialize(Archive &ar, const unsigned int)
    {
        ar & lowercase;
        ar & unk_word;
        ar & num_word;
    }
};

struct Dict
{
    typedef std::unordered_map<std::string, unsigned> Map;

    DictSettings settings;
    bool frozen = false;

    std::vector<std::string> id_to_word;
    Map word_to_id;

    Dict();
    Dict(const DictSettings& settings);

    bool has_unk() const;
    bool has_num() const;
    unsigned size() const;
    bool contains(const std::string& words) const;

    void freeze();
    bool is_frozen() const;
    bool is_special(const std::string& word) const;

    std::string normalize(const std::string& word) const;
    unsigned convert(const std::string& word);
    const std::string& convert(const unsigned id) const;
    void clear();

    template<class Archive>
    void serialize(Archive &ar, const unsigned int)
    {
        ar & settings;
        ar & frozen;
        ar & id_to_word;
        ar & word_to_id;
    }
};

std::shared_ptr<Dict> read_dict_from_file(const std::string &path);

}