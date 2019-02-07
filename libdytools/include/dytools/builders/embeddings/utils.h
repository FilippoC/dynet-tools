#pragma once

#include <fstream>
#include <boost/algorithm/string.hpp>

namespace dytools
{

template<class T>
void read_embeddings_file(const std::string& path, const unsigned dim, T&& callback, bool skip_first_line=false);


// template func implementation
template<class T>
void read_embeddings_file(const std::string& path, const unsigned dim, T&& callback, bool skip_first_line)
{
    std::ifstream fin(path);
    if (!fin.is_open())
        throw std::runtime_error("Could not open embeddings file");

    std::string s;
    // skip the first line
    if (skip_first_line)
        getline(fin, s);
    while(getline(fin, s))
    {
        std::vector<std::string> fields;
        boost::algorithm::trim(s);
        if (s.size() > 0)
        {
            boost::algorithm::split(fields, s, boost::algorithm::is_any_of(" "));
            std::string word = fields[0];

            std::vector<float> values;
            for (unsigned i = 0; i < dim; ++i)
                values.push_back(std::stod(fields[1 + i]));

            callback(word, values);
        }
    }
}

}