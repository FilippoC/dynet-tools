#include <dytools/io.h>

namespace dytools
{

TextFileSaver::TextFileSaver(const std::string& path) :
        ostream(path),
        oarchive(ostream)
{}

void TextFileSaver::close()
{
    ostream.close();
}

TextFileLoader::TextFileLoader(const std::string& path) :
        istream(path),
        iarchive(istream)
{
    if (istream.fail())
        throw std::runtime_error("Coul not open setting file: " + path);
}

void TextFileLoader::close()
{
    istream.close();
}

}