#pragma once

namespace dytools
{

template<class DataType>
struct Network
{
    virtual dynet::Expression labeled_loss(const DataType& data);
    virtual dynet::Expression unlabeled_loss(const DataType& data);
};

template<class DataType>
dynet::Expression Network<DataType>::labeled_loss(const DataType&)
{
    throw std::runtime_error("Not implemented: labeled_loss");
}

template<class DataType>
dynet::Expression Network<DataType>::unlabeled_loss(const DataType&)
{
    throw std::runtime_error("Not implemented: unlabeled_loss");
}

}