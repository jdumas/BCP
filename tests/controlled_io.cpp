#include "ControlledValidation.h"
#include <array>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc,char** argv) try {
    if(argc!=2) return 2;
    bcp::cv::init();
    const std::string mode=argv[1];
    const std::array<double,6> data={1,2,3,4,5,6};
    bcp::cv::writeF64("written","test","row",2,3,data.data(),false);
    if(mode=="duplicate") {
        const std::array<double,6> other={6,5,4,3,2,1};
        bcp::cv::writeF64("written","test","row",2,3,other.data(),false);
        throw std::runtime_error("duplicate overwrite was not rejected");
    }
    auto read=bcp::cv::readF64("input",2,3,false);
    if(mode=="good" && !std::equal(read.begin(),read.end(),data.begin()))
        throw std::runtime_error("replay memory-order conversion changed values");
    const std::array<int32_t,3> ids={0,7,9};
    bcp::cv::writeI32("ids","test","vertex",3,1,ids.data());
    return 0;
} catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
