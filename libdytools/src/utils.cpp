#include "dytools/utils.h"

#include <boost/algorithm/string/case_conv.hpp>
#include <memory>
#include <cstdio>

namespace dytools
{

const boost::regex regex_num("[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+");
const boost::regex regex_punct("[[:punct:]]+");

bool is_num(const std::string& s)
{
    return boost::regex_match(s, regex_num);
}

bool is_punct(const std::string& s)
{
    return boost::regex_match(s, regex_punct);
}

std::string to_lower(const std::string& s)
{
    return boost::algorithm::to_lower_copy(s);
}

std::string exec(const char* cmd) {
    std::array<char, 128> buffer;
    std::string result;
    std::shared_ptr<FILE> pipe(popen(cmd, "r"), pclose);
    if (!pipe) throw std::runtime_error("popen() failed!");
    while (!feof(pipe.get())) {
        if (fgets(buffer.data(), 128, pipe.get()) != nullptr)
            result += buffer.data();
    }
    //pclose(&(*pipe));
    return result;
}

}