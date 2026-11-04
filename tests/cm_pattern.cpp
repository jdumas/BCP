// CSR assembly must sum every duplicate triplet, independently of sort tie order.
#include "PardisoSolver.h"
#include <Eigen/Sparse>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <vector>

class PatternProbe : public PardisoSolver<std::vector<int>, std::vector<double>> {
public:
    Eigen::MatrixXd matrix() const {
        Eigen::MatrixXd result=Eigen::MatrixXd::Zero(numRows,numRows);
        for (int r=0;r<numRows;++r)
            for (int k=ia(r)-1;k<ia(r+1)-1;++k)
                result(r,ja(k)-1)=a(k);
        return result;
    }
};

int main() try {
    constexpr int n=10;
    std::vector<std::pair<int,int>> keys;
    for (int r=0;r<n;++r) keys.emplace_back(r,r);
    for (int r=0;r<n && keys.size()<40;++r)
        for (int c=r+1;c<n && keys.size()<40;++c) keys.emplace_back(r,c);
    std::vector<int> rows,cols;
    std::vector<double> values;
    for (size_t key=0;key<keys.size();++key)
        for (int repeat=0;repeat<5;++repeat) {
            rows.push_back(keys[key].first);
            cols.push_back(keys[key].second);
            values.push_back(double(100*key+repeat+1));
        }
    for (bool reverse : {false,true}) {
        if (reverse) {
            std::reverse(rows.begin(),rows.end());
            std::reverse(cols.begin(),cols.end());
            std::reverse(values.begin(),values.end());
        }
        PatternProbe solver;
        solver.set_type(-2,true);
        solver.set_pattern(rows,cols,values);
        for (int update=0;update<2;++update) {
            std::vector<Eigen::Triplet<double>> triplets;
            for (size_t i=0;i<values.size();++i)
                triplets.emplace_back(rows[i],cols[i],values[i]);
            Eigen::SparseMatrix<double> expected(n,n);
            expected.setFromTriplets(triplets.begin(),triplets.end());
            const double error=(solver.matrix()-Eigen::MatrixXd(expected)).cwiseAbs().maxCoeff();
            if (error!=0) throw std::runtime_error(
                "Duplicate-triplet CSR values disagree with independent assembly: max error "+
                std::to_string(error));
            for (size_t i=0;i<values.size();++i) values[i]=2*values[i]+double(i%7);
            solver.update_a(values);
        }
    }
    std::cout << "CM duplicate-triplet assembly and value-update tests passed\n";
    return 0;
} catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
