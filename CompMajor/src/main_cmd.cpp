#include "Newton.h"
#include <igl/read_triangle_mesh.h>
#include <igl/writeOBJ.h>
#include "Clock.h"
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef BCP_PORTABLE
#include "MatFile.h"
#else
#include <Utils/MatlabInterface.h>
#include "Utils/MatlabGMMDataExchange.h"
#endif



#define BUFFER_SIZE 4000
void setMatlabPath();

double normalizeMeshArea(MatX3& V, MatX3i& F);


int main(int argc, char** argv) try
{
#ifndef BCP_PORTABLE
	setMatlabPath();
#endif
#ifdef _OPENMP
	const char* threads = std::getenv("OMP_NUM_THREADS");
	if (threads) omp_set_num_threads(std::max(std::atoi(threads), 1));
#endif

	if (argc < 5 || argc > 8)
	{
		cout << "Syntax: Parameterization_cmd.exe <Input OBJ file name> <Output OBJ file name> <Output MAT file name> <Meta data .txt file name> <target energy> <target time> <Initialization file name>" << endl;
		return EXIT_FAILURE;
	}
	// Tutte is performed anyway, even if target time is smaller
	MatX3 V;
	MatX3i F;
	cout << "Started loading mesh..." << endl;
	if (!igl::read_triangle_mesh(argv[1], V, F))
	{
		cerr << "Failed to load mesh: " << argv[1] << endl;
		return EXIT_FAILURE;
	}
	if (V.rows() < 3 || F.rows() < 1 || !V.allFinite() || F.minCoeff() < 0 || F.maxCoeff() >= V.rows())
		throw std::runtime_error("Invalid triangle mesh vertices/indices");

	MatX3 init_V;
	MatX3i init_F;

	if (argc == 8) {
		if (!igl::read_triangle_mesh(argv[7], init_V, init_F) ||
			init_V.rows() != V.rows() || init_F.rows() != F.rows() || init_F != F)
			throw std::runtime_error("Initialization must have the same vertices and faces as the input");
	}

	float target_Esd = 2.0;
	if (argc >= 6)
		target_Esd = stof(argv[5]);

	//target_Esd = 3.1;

	float target_time = 500.0;
	if (argc >= 7)
		target_time = stof(argv[6]);

	cout << "Finished loading mesh..." << endl;

	float time_Tutte_init = 0.0;
	auto start_init_time = bcp::Clock::now();

	double normalizationFactor = normalizeMeshArea(V, F);
	V /= normalizationFactor;
	Newton solver;
	if (argc <= 7)
		solver.init(V, F, time_Tutte_init);
	else
	{
		init_V /= normalizationFactor;
		solver.init_from_input(V, F, init_V);
	}

	auto end_init_time = bcp::Clock::now();

	auto cur_time = bcp::Clock::now();

	float prevF, curF;
	curF = INFINITY;
	Vec prevX = solver.m_x;

	int fcounter = 0, xcounter = 0, num_conv_iters = 3;
	float ftol = 1e-2, xtol = 1e-10;
	int iter = 0;

	prevF = solver.f / solver.energy->symDirichlet->Area.sum();

	int MAX_ITERS = 1000;
	int iters_count = 0;
	bool runStepSizeCheck = true;

	for (iter = 0; iter < MAX_ITERS; iter++)
	{
		if (solver.step() != 0)
			throw std::runtime_error("Newton step failed");

		curF = solver.linesearch();
		if (!std::isfinite(curF) || !solver.m_x.allFinite())
			throw std::runtime_error("Newton iteration produced non-finite output");

		cur_time = bcp::Clock::now();
		if (bcp::seconds(cur_time, start_init_time) >= target_time)
			break;

		iters_count++;

		if (std::isnan(prevF))
			break;

		if (abs(curF - prevF) < ftol * (solver.f + 1))
		{
			if (fcounter >= num_conv_iters)
			{
				cout << "Converged: Change in energy < tol" << endl;
				break;
			}
			else
				fcounter += 1;
		}
		else
			fcounter = 0;

		if ((solver.m_x - prevX).norm() < xtol * (solver.m_x.norm() + 1))
		{
			if (xcounter >= num_conv_iters)
			{
				cout << "Converged: Change in X < tol" << endl;
				break;
			}
			else
				xcounter += 1;
		}
		else
			xcounter = 0;

		if (curF <= target_Esd)
		{
			cout << "Converged: reached target Esd" << endl;
			break;
		}

		Vec tmp = solver.m_x;
		prevX = tmp;
		prevF = solver.f / solver.energy->symDirichlet->Area.sum();

	}
	Vec tmp = solver.m_x;
	prevX = tmp;
	cout << "Esd = " << curF << endl;
	cout << "k = " << solver.energy->symDirichlet->evaluate_small_k(prevX) / solver.energy->symDirichlet->Area.sum() << endl;

	auto end_Newton_time = bcp::Clock::now();

	cout << "Newton time consumption: " << bcp::seconds(end_Newton_time, end_init_time) << endl;
	cout << "Total time consumption: " << bcp::seconds(end_Newton_time, start_init_time) << endl;
	cout << iters_count << " iterations were performed" << endl;

	MatX3 CN;
	MatX3i FN;
	solver.uv = Eigen::Map<MatX2>(prevX.data(), prevX.size() / 2, 2);
	MatX3 UVs(V.rows(), 3);
	UVs.leftCols(2) = solver.uv;
	UVs *= normalizationFactor;
	UVs.rightCols(1).setZero();
	if (strcmp(argv[2], "") != 0)
		if (!igl::writeOBJ(argv[2], UVs, F, CN, FN, CN, FN))
			throw std::runtime_error("Failed to write output OBJ");

	
	std::string uv_mat_path = argv[3];
	if (uv_mat_path != "")
	{
#ifdef BCP_PORTABLE
		bcp::MatFile output(uv_mat_path, true);
		output.write("uvs", UVs.leftCols(2));
#else
		GMMDenseColMatrix uvs(V.rows(), 2);
		for (int i = 0; i < V.rows(); i++)
		{
			uvs(i, 0) = UVs(i, 0);
			uvs(i, 1) = UVs(i, 1);
		}
		MatlabGMMDataExchange::SetEngineDenseMatrix("uvs", uvs);
		std::string cmd_save_uv = "save('" + uv_mat_path + "', 'uvs');";
		MatlabInterface::GetEngine().Eval(cmd_save_uv.c_str());
#endif
	}
	

	std::string file_path = argv[4];

	// Create an ofstream object
	std::ofstream file(file_path);

	// Check if the file is open
	if (file.is_open()) {
		file << "Total time consumption: " << bcp::seconds(end_Newton_time, start_init_time) << endl;
		file << "Newton iteration count: " << iters_count << std::endl;
		file << "Final E_SD: " << curF << endl;
		file << "Tutte initialization time: " << time_Tutte_init << endl;
		file << "Tutte + Hessian initialization time: " << bcp::seconds(end_init_time, start_init_time) << endl;
		file << "Did Tutte was finished before target time: " << (time_Tutte_init <= target_time) << endl;


		// Close the file
		file.close();

		std::cout << "Text successfully saved to " << file_path << std::endl;
	}
	else {
		// Handle the error if the file couldn't be opened
		std::cerr << "Failed to open file " << file_path << std::endl;
		return EXIT_FAILURE;
	}
	return 0;
} catch (const std::exception& e) {
	std::cerr << "CM: " << e.what() << std::endl;
	return EXIT_FAILURE;
}


#ifndef BCP_PORTABLE
void setMatlabPath()
{
	char path[BUFFER_SIZE];
	_fullpath(path, "..\\", BUFFER_SIZE);
	int index = strlen(path);
	//path[index - 4] = '\0';
	std::string matlabPath(path);
	matlabPath += "MatlabScripts\\";
	MatlabInterface::GetEngine().AddScriptPath(matlabPath.c_str());
}

#endif

double normalizeMeshArea(MatX3& V, MatX3i& F) {
	Eigen::VectorXd M;
	// set uv coords scale
	igl::doublearea(V, F, M);
	if (!M.allFinite() || M.minCoeff() <= 0)
		throw std::runtime_error("Input has degenerate/non-finite triangles");
	M /= 2.0;

	double mesh_area = M.sum();
	return sqrt(mesh_area);
}