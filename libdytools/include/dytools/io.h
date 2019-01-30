#pragma once

#include <fstream>
#include <boost/archive/impl/basic_text_iarchive.ipp>
#include <boost/archive/impl/text_iarchive_impl.ipp>
#include <boost/archive/impl/basic_text_oarchive.ipp>
#include <boost/archive/impl/text_oarchive_impl.ipp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

namespace dytools
{

struct TextFileSaver
{
    std::ofstream ostream;
    boost::archive::text_oarchive oarchive;

    TextFileSaver(const std::string& path);
    void close();

    template <class T>
    void save(const T& object);
};

struct TextFileLoader
{
    std::ifstream istream;
    boost::archive::text_iarchive iarchive;

    TextFileLoader(const std::string& path);
    void close();

    template <class T>
    void load(T& object);
};

template <class T>
void TextFileSaver::save(const T& object)
{
    oarchive << object;
}

template<typename T>
void TextFileLoader::load(T& obj)
{
    iarchive >> obj;
}

}