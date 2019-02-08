#include "dytools/data/conll.h"

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <boost/algorithm/string.hpp>

namespace dytools
{

ConllToken::ConllToken(
        const std::string& word,
        const std::string& lemma,
        const std::string& cpostag,
        const std::string& postag,
        const std::string& feats,
        const unsigned head,
        const std::string& deprel,
        const std::string& phead,
        const std::string& pdeprel
) :
        word(word),
        lemma(lemma),
        cpostag(cpostag),
        postag(postag),
        feats(feats),
        head(head),
        deprel(deprel),
        phead(phead),
        pdeprel(pdeprel)
{}

void ConllSentence::update_tags(const std::vector<unsigned>& tags)
{
    if (tags.size() != size())
        throw std::length_error("Tag list and sentence are of different size");

    for (auto i = 0u ; i < size() ; ++i)
        (*this)[i].postag = tags[i];
}
void ConllSentence::update_heads(const std::vector<unsigned>& heads)
{
    if (heads.size() != this->size())
        throw std::length_error("head list and sentence are of different size");

    for (auto i = 0u ; i < size() ; ++i)
        (*this)[i].postag = heads[i];
}

// gold sentence, we have to compute the dependency tree as a sparse matrix
dynet::Expression sentence_to_sparse_matrix(dynet::ComputationGraph &cg, const ConllSentence &sentence)
{
    const unsigned graph_width = sentence.size() + 1;

    std::vector<unsigned> idx;
    for (unsigned i = 0u; i < sentence.size(); ++i)
    {
        const unsigned mod = i + 1;
        const unsigned head = (sentence.at(i).head == i ? 0u : sentence.at(i).head + 1);
        idx.push_back(head + mod * graph_width);
    }

    std::vector<float> values(idx.size(), 1.f);
    auto arcs = dynet::input(cg, {graph_width, graph_width}, idx, values);

    return arcs;
}


float uas(const ConllSentence& sentence, const std::vector<unsigned>& heads, bool normalize)
{
    float sum = 0.f;
    for (unsigned i = 0u; i < sentence.size(); ++i)
        if (sentence.at(i).head == heads.at(i))
            sum++;

    if (normalize)
        sum = sum / (float) sentence.size();

    return sum;
}


unsigned read(const std::string& path, std::vector<ConllSentence>& output)
{
    unsigned n = 0u;
    unsigned next_token_id = 0u;

    std::string line;
    std::string id;
    std::string word;
    std::string lemma;
    std::string cpostag;
    std::string postag;
    std::string feats;
    std::string str_head;
    std::string deprel;
    std::string phead;
    std::string pdeprel;

    ConllSentence* sentence = nullptr;
    std::ifstream is(path);
    if (!is.is_open())
        throw std::runtime_error("Could not open file: " + path);

    while (std::getline(is, line)) {
        if (line.length() <= 0)
        {
            if (sentence != nullptr)
            {
                output.push_back(*sentence);
                ++ n;
                delete sentence;
                sentence = nullptr;
            }

            continue;
        }
        if (line[0] == '#')
            continue;

        if (sentence == nullptr)
        {
            sentence = new ConllSentence();
            next_token_id = 0u;
        }

        /* in swedish, a token can contain a space
        std::istringstream ss(line);
        ss
            >> id
            >> word
            >> lemma
            >> cpostag
            >> postag
            >> feats
            >> str_head
            >> deprel
            >> phead
            >> pdeprel
        ;
        */

        std::vector<std::string> strs;
        boost::split(strs, line, boost::is_any_of("\t"));
        id = strs.at(0);
        word = strs.at(1);
        lemma = strs.at(2);
        cpostag = strs.at(3);
        postag = strs.at(4);
        feats = strs.at(5);
        str_head = strs.at(6);
        deprel = strs.at(7);
        phead = strs.at(8);
        pdeprel = strs.at(9);

        // skip ids with . and - inside
        if (id.find(".") != std::string::npos || id.find("-") != std::string::npos)
            continue;

        unsigned head = std::stoul(str_head);
        if (head == 0u)
            head = next_token_id;
        else
            head -= 1u; // firt word as index=0, not 1 as in conll

        sentence->emplace_back(
                word,
                lemma,
                cpostag,
                postag,
                feats,
                head,
                deprel,
                phead,
                pdeprel
        );
        ++ next_token_id;
    }

    if (sentence != nullptr)
    {
        output.push_back(*sentence);
        ++ n;
        delete sentence;
    }

    is.close();

    return n;
}

void write(std::ostream& os, const ConllSentence& sentence)
{
    unsigned id = 0u;
    for (const auto& token : sentence)
    {
        os << id + 1u; // indexed by 1 in conll
        os << "\t";
        os << token.word;
        os << "\t";
        os << token.lemma;
        os << "\t";
        os << token.cpostag;
        os << "\t";
        os << token.postag;
        os << "\t";
        os << token.feats;
        os << "\t";
        os << (token.head == id ? 0u : token.head + 1);
        os << "\t";
        os << token.deprel;
        os << "\t";
        os << token.phead;
        os << "\t";
        os << token.pdeprel;
        os << "\n";

        ++ id;
    }
}

void write(std::ostream& os, const std::vector<ConllSentence>& data)
{
    for (auto const& sentence : data)
    {
        write(os, sentence);
        os << "\n";
    }
}

}