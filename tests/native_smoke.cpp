#include "MatFile.h"
#include <Eigen/LU>
#include <algorithm>
#include <vector>
#include <igl/read_triangle_mesh.h>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>

#include "PardisoCompat.h"
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
std::string quote(const std::string& value) {
#ifdef _WIN32
    return "\"" + value + "\"";
#else
    std::string result = "'";
    for (char c : value) result += c == '\'' ? "'\\''" : std::string(1,c);
    return result + "'";
#endif
}
int run_command(const std::string& command) {
#ifdef _WIN32
    // cmd /c otherwise strips the executable's first quote when paths contain spaces.
    return std::system(("\"" + command + "\"").c_str());
#else
    return std::system(command.c_str());
#endif
}
void test_solver() {
    // Indefinite symmetric KKT-like system with two right-hand sides.
    void* pt[64]; int type=-2, solver=0, ip[64], error, phase, n=2, rhs=2, dummy=0, one=1;
    double dp[64], a[]={2,1,-1}, b[]={3,0,1,-1}, x[4];
    int ia[]={1,3,4}, ja[]={1,2,2};
    pardisoinit(pt,&type,&solver,ip,dp,&error);
    phase=12;
    pardiso(pt,&one,&one,&type,&phase,&n,a,ia,ja,&dummy,&rhs,ip,&dummy,b,x,&error,dp);
    require(!error,"Indefinite factorization failed");
    phase=33;
    pardiso(pt,&one,&one,&type,&phase,&n,a,ia,ja,&dummy,&rhs,ip,&dummy,b,x,&error,dp);
    require(!error,"Solve failed");
    Eigen::Matrix2d m; m << 2,1,1,-1;
    require((m * Eigen::Map<Eigen::Matrix2d>(x) - Eigen::Map<Eigen::Matrix2d>(b)).norm() < 1e-10,"Bad solve residual");
    const double unit_rhs[]={1,0,0,1};
    std::copy(unit_rhs,unit_rhs+4,b);
    phase=33;
    pardiso(pt,&one,&one,&type,&phase,&n,a,ia,ja,&dummy,&rhs,ip,&dummy,b,x,&error,dp);
    require(!error && (Eigen::Map<Eigen::Matrix2d>(x)-m.inverse()).norm()<1e-10,"Bad inverse-column solves");
    // Reuse the same factors with full and partial production-sized RHS batches.
    for (int first=0; first<19; first+=16) {
        rhs=std::min(16,19-first);
        std::vector<double> units(n*rhs,0.0), columns(n*rhs);
        for (int j=0;j<rhs;++j) units[n*j+(first+j)%n]=1.0;
        pardiso(pt,&one,&one,&type,&phase,&n,a,ia,ja,&dummy,&rhs,ip,&dummy,
                units.data(),columns.data(),&error,dp);
        require(!error,"Batched inverse-column solve failed");
        using Matrix=Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>;
        require((m*Eigen::Map<Matrix>(columns.data(),n,rhs)-
                 Eigen::Map<Matrix>(units.data(),n,rhs)).norm()<1e-10,
                "Bad full/partial batch residual");
    }
    phase=-1;
    pardiso(pt,&one,&one,&type,&phase,&n,a,ia,ja,&dummy,&rhs,ip,&dummy,b,x,&error,dp);
}
int main(int argc, char** argv) try {
    require(argc==4,"Expected CM, PT and work directory");
    test_solver();
    namespace fs=std::filesystem;
    fs::path dir=fs::path(argv[3]) / "path with spaces";
    fs::create_directories(dir);
    const auto input=dir/"input.obj", output=dir/"flat.obj", mat=dir/"uv.mat", meta=dir/"meta.txt";
    {
        std::ofstream mesh(input);
        for (int j=0;j<5;++j) for(int i=0;i<5;++i)
            mesh << "v " << i/4.0 << ' ' << j/4.0 << ' ' << 0.1*std::sin(i)*std::sin(j) << '\n';
        for(int j=0;j<4;++j) for(int i=0;i<4;++i) {
            int a=j*5+i+1;
            mesh << "f " << a << ' ' << a+1 << ' ' << a+6 << '\n';
            mesh << "f " << a << ' ' << a+6 << ' ' << a+5 << '\n';
        }
    }
    const auto command=quote(argv[1])+" "+quote(input.string())+" "+quote(output.string())+
        " "+quote(mat.string())+" "+quote(meta.string())+" 2.01 10";
    require(run_command(command)==0,"CM smoke run failed");
    Eigen::MatrixXd uv; Eigen::MatrixXi faces;
    require(igl::read_triangle_mesh(output.string(),uv,faces),"CM OBJ missing");
    bcp::MatFile result(mat.string());
    auto uvs=result.read("uvs");
    require(uv.rows()==25 && faces.rows()==32 && uvs.rows()==25 && uvs.cols()==2 &&
        uv.allFinite() && uvs.allFinite(),"Invalid UV output");
    require((uv.leftCols(2)-uvs).norm()<1e-6,"OBJ/MAT outputs disagree");
    for(int i=0;i<faces.rows();++i) {
        Eigen::Vector2d a=uvs.row(faces(i,1))-uvs.row(faces(i,0));
        Eigen::Vector2d b=uvs.row(faces(i,2))-uvs.row(faces(i,0));
        require(a.x()*b.y()-a.y()*b.x()>0,"Flipped or collapsed CM triangle");
    }
    require(run_command(command+" "+quote(output.string()))==0,"CM initialization path failed");
    const auto no_obj=quote(argv[1])+" "+quote(input.string())+" "+quote("")+" "+
        quote(mat.string())+" "+quote(meta.string())+" 2 0";
    require(run_command(no_obj)==0,"CM empty OBJ / target-runtime protocol failed");
    std::ifstream metadata(meta);
    std::string first, second, third;
    std::getline(metadata,first); std::getline(metadata,second); std::getline(metadata,third);
    require(first.find("Total time consumption: ")==0 &&
        second.find("Newton iteration count: ")==0 && third.find("Final E_SD: ")==0,"CM metadata protocol changed");
    const auto bad=dir/"closed.obj", bad_mat=dir/"closed.mat", bad_meta=dir/"closed.txt";
    {
        std::ofstream mesh(bad);
        mesh << "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\n"
                "f 1 3 2\nf 1 2 4\nf 2 3 4\nf 3 1 4\n";
    }
    fs::remove(bad_mat); fs::remove(bad_meta);
    require(run_command(quote(argv[1])+" "+quote(bad.string())+" "+quote("")+" "+
        quote(bad_mat.string())+" "+quote(bad_meta.string()))!=0,"CM accepted a closed mesh");
    require(!fs::exists(bad_mat) && !fs::exists(bad_meta),"CM wrote outputs after invalid input");
    const auto source=dir/"pt_source.mat", target=dir/"pt_result.mat";
    {
        bcp::MatFile file(source.string(),true);
        Eigen::MatrixXd f(1,3); f<<0,1,2;
        Eigen::MatrixXd rings(4,4); rings<<-1,1,3,2, -1,2,3,0, -1,0,3,1, -1,0,1,2;
        Eigen::MatrixXd angles(4,4); angles<<0,0,1,2, 0,0,1,2, 0,0,1,2, 0,0,1,2;
        file.write("f",f); file.write("one_rings",rings);
        file.write("integrated_angles",angles);
        // scipy.io.savemat serializes a one-dimensional array as a ROW vector.
        file.write("all_one_ring_angles_sums",Eigen::MatrixXd::Constant(1,4,3));
    }
    auto pt_command=quote(argv[2])+" "+quote(source.string())+" "+quote(target.string())+" 1";
    require(run_command(pt_command)==0,"PT smoke run failed");
    bcp::MatFile pt_result(target.string());
    auto angles=pt_result.read("angles_vertices_to_faces");
    require(angles.rows()==1 && angles.cols()==3 && (angles.array()-2.5).abs().maxCoeff()<1e-12,"Incorrect PT midpoint/wraparound angles");
    require(pt_result.read("total_time")(0,0)>=0,"Invalid PT time");
    require(run_command(quote(argv[2])+" "+quote((dir/"missing.mat").string())+" "+quote(target.string())+" 1")!=0,"PT accepted missing input");
    std::cout << "Native smoke tests passed\n";
    return 0;
} catch(const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
