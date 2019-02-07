#include "dytools/utils.h"

#include <boost/algorithm/string/case_conv.hpp>

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

}