#include "dytools/dict.h"

#include <fstream>
#include "boost/algorithm/string/trim.hpp"

namespace dytools
{


Dict::Dict()
{}


Dict::Dict(const DictSettings& settings) :
        settings(settings)
{
    if (has_unk())
        convert(settings.unk_word);
    if (has_num())
        convert(settings.num_word);
}

bool Dict::has_unk() const
{
    return settings.unk_word.size() > 0;
}

bool Dict::has_num() const
{
    return settings.unk_word.size() > 0;
}

unsigned Dict::size() const
{
    return id_to_word.size();
}

bool Dict::contains(const std::string& _word) const
{
    const auto word = normalize(_word);
    return !(word_to_id.find(word) == word_to_id.end());
}

void Dict::freeze()
{
    frozen = true;
}

bool Dict::is_frozen() const
{
    return frozen;
}

std::string Dict::normalize(const std::string& word) const
{
    if (has_num() && is_num(word))
        return settings.num_word;
    else if (settings.lowercase)
        return to_lower(word);
    else
        return word;
}

bool Dict::is_special(const std::string& word) const
{
    const auto w = normalize(word);
    return w == settings.unk_word || w == settings.num_word;
}

unsigned Dict::convert(const std::string& _word)
{
    // normalize word
    const auto word = normalize(_word);

    auto it = word_to_id.find(word);
    if (it == word_to_id.end())
    {
        if (frozen)
        {
            if (has_unk())
                return word_to_id[settings.unk_word];
            else
                std::runtime_error("Unknown word encountered in frozen dictionary");
        }
        const auto id = id_to_word.size();
        word_to_id[word] = id;
        id_to_word.push_back(word);

        return id;
    }
    else
    {
        return it->second;
    }
}

const std::string& Dict::convert(const unsigned id) const
{
    if (id > id_to_word.size())
        throw std::runtime_error("id too big: not in dict!");
    return id_to_word[id];
}


void Dict::clear()
{
    word_to_id.clear();
    id_to_word.clear();
}

std::shared_ptr<Dict> read_dict_from_file(const std::string &path)
{
    std::shared_ptr<Dict> dict(new Dict());

    std::ifstream fin(path);
    if (!fin.is_open())
        throw std::runtime_error("Could not open embeddings file");

    std::string s;
    while(getline(fin, s))
    {
        boost::algorithm::trim(s);
        dict->convert(s);
    }

    return dict;
}

}