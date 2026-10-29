#pragma once
#include <Eigen/Core>
#include <matio.h>
#include <memory>
#include <stdexcept>
#include <string>

namespace bcp {
class MatFile {
    using FilePtr = std::unique_ptr<mat_t, decltype(&Mat_Close)>;
    FilePtr file_;
public:
    explicit MatFile(const std::string& path, bool write = false)
        : file_(write ? Mat_CreateVer(path.c_str(), nullptr, MAT_FT_MAT5)
                      : Mat_Open(path.c_str(), MAT_ACC_RDONLY), &Mat_Close) {
        if (!file_) throw std::runtime_error("Cannot open MAT file: " + path);
    }
    Eigen::MatrixXd read(const char* name) {
        std::unique_ptr<matvar_t, decltype(&Mat_VarFree)> var(Mat_VarRead(file_.get(), name), &Mat_VarFree);
        if (!var || var->rank != 2 || var->isComplex || var->class_type != MAT_C_DOUBLE || !var->data)
            throw std::runtime_error(std::string("Expected a dense real double matrix: ") + name);
        return Eigen::Map<const Eigen::MatrixXd>(static_cast<const double*>(var->data), var->dims[0], var->dims[1]);
    }
    template<class Derived>
    void write(const char* name, const Eigen::MatrixBase<Derived>& value) {
        Eigen::MatrixXd data = value.template cast<double>();
        size_t dims[] = {static_cast<size_t>(data.rows()), static_cast<size_t>(data.cols())};
        std::unique_ptr<matvar_t, decltype(&Mat_VarFree)> var(
            Mat_VarCreate(name, MAT_C_DOUBLE, MAT_T_DOUBLE, 2, dims, data.data(), 0), &Mat_VarFree);
        if (!var || Mat_VarWrite(file_.get(), var.get(), MAT_COMPRESSION_NONE))
            throw std::runtime_error(std::string("Cannot write MAT variable: ") + name);
    }
};
}
