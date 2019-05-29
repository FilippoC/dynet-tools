#include "dytools/dict.h"

#include <fstream>
#include "boost/algorithm/string/trim.hpp"

namespace dytools
{

Dict::Dict(bool _lowercase, bool _has_num, bool _has_unk) :
    has_unk(_has_unk),
    has_num(_has_num),
    lowercase(_lowercase)
{
    if (has_unk)
    {
        unk_id = (unsigned) id_to_word.size();
        id_to_word.push_back(unk_str);
        word_to_id.emplace(std::make_pair(unk_str, unk_id));
    }

    if (_has_num)
    {
        num_id = (unsigned) id_to_word.size();
        id_to_word.push_back(num_str);
        word_to_id.emplace(std::make_pair(num_str, num_id));
    }
}

std::string Dict::normalize(const std::string& word) const
{
    if (has_num && is_num(word))
        return num_str;
    else if (lowercase)
        return to_lower(word);
    else
        return word;
}

unsigned Dict::to_id(const std::string& _word) const
{
    const auto word = normalize(_word);

    auto it = word_to_id.find(word);
    if (it == word_to_id.end())
    {
        if (has_unk)
            return unk_id;
        else
            throw std::runtime_error("Word not in dict");
    }
    else
        return it->second;
}

unsigned Dict::to_id(const char& _char) const
{
    return to_id(std::string(1, _char));
}

std::string Dict::to_string(const unsigned id) const
{
    return id_to_word.at(id);
}

void Dict::add(const std::string& _word)
{
    const auto word = normalize(_word);

    auto it = word_to_id.find(word);
    if (it == word_to_id.end())
    {
        const unsigned id = (unsigned) word_to_id.size();
        id_to_word.push_back(word);
        word_to_id.emplace(std::make_pair(word, id));
    }
}

void Dict::add(const char& _char)
{
    add(std::string(1, _char));
}

unsigned Dict::size() const
{
    return (unsigned) id_to_word.size();
}


}