#pragma once

#include <vector>
#include <string>
#include <iostream>
#include <memory>

#include "dytools/dict.h"

namespace dytools
{

struct ConllToken
{
    std::string word;
    std::string lemma;
    std::string cpostag;
    std::string postag;
    std::string feats;
    unsigned head;
    std::string deprel;
    std::string phead;
    std::string pdeprel;

    ConllToken() = delete;
    ConllToken(
            const std::string& word,
            const std::string& lemma,
            const std::string& cpostag,
            const std::string& postag,
            const std::string& feats,
            const unsigned head,
            const std::string& deprel,
            const std::string& phead,
            const std::string& pdeprel
    );
};

struct ConllSentence : public std::vector<ConllToken>
{
    void update_tags(const std::vector<unsigned>& tags);
    void update_heads(const std::vector<unsigned>& heads);
};

float uas(const ConllSentence& sentence, const std::vector<unsigned>& heads, bool normalize);

void write(std::ostream& os, const ConllSentence& sentence);
void write(std::ostream& os, const std::vector<ConllSentence>& data);
unsigned read(const std::string&, std::vector<ConllSentence>& output);

template<class It>
std::shared_ptr<dytools::Dict> build_conll_token_dict(It begin, It end)
{
    auto dict = std::make_shared<dytools::Dict>();
    for(;begin != end; ++begin)
    {
        const ConllSentence& sentence = *begin;
        for (const ConllToken& token : sentence)
            dict->convert(token.word);
    }
    dict->convert("<NUM/>");
    dict->freeze();
    dict->set_unk("<UNK/>");
    return dict;
}

template<class It>
std::shared_ptr<dytools::Dict> build_conll_char_dict(It begin, It end)
{
    auto dict = std::make_shared<dytools::Dict>();
    for(;begin != end; ++begin)
    {
        const ConllSentence& sentence = *begin;
        for (const ConllToken& token : sentence)
        {
            const std::string &word = token.word;
            for (unsigned i = 0u; i < word.size(); ++i)
            {
                const char c = word[i];
                dict->convert(std::to_string(c));
            }
        }
    }
    dict->convert("<S>");
    dict->convert("</S>");
    dict->freeze();
    return dict;
}

template<class It>
std::shared_ptr<dytools::Dict> build_conll_tag_dict(It begin, It end)
{
    auto dict = std::make_shared<dytools::Dict>();
    for(;begin != end; ++begin)
    {
        const ConllSentence& sentence = *begin;
        for (const ConllToken& token : sentence)
            dict->convert(token.postag);
    }
    dict->freeze();
    return dict;
}

}