#pragma once

#include <boost/regex.hpp>
#include <dynet/expr.h>
#include <dynet/devices.h>

namespace dytools
{

extern const boost::regex regex_num;
extern const boost::regex regex_punct;

bool is_num(const std::string& s);
bool is_punct(const std::string& s);
std::string to_lower(const std::string& s);


template<typename T, class... Args>
dynet::Expression force_cpu(T&& f, dynet::Expression input, Args&&... args)
{
    const auto device_name = input.get_device_name();
    auto* device = dynet::get_device_manager()->get_global_device(device_name);

    if (device->type == dynet::DeviceType::GPU)
    {
        const auto cpu_input = dynet::to_device(input, dynet::get_device_manager()->get_global_device("CPU"));
        const auto cpu_output = f(cpu_input, args...);
        const auto output = dynet::to_device(cpu_output, device);

        return output;
    }
    else
    {
        return f(input, args...);
    }
}

inline
dynet::Device* get_cpu_device()
{
    return dynet::get_device_manager()->get_global_device("CPU");
}

inline
dynet::Device* get_default_device()
{
    return dynet::default_device;
}

inline
bool is_gpu_default()
{
    return dynet::default_device->type == dynet::DeviceType::GPU;
}

}