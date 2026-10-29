#include "Solver.h"
#include "Clock.h"

#include "Utils.h"
#include "Energy.h"

#include <iostream>
#include <igl/readOBJ.h>
#include <igl/flipped_triangles.h>
#include <igl/map_vertices_to_circle.h>
#include <igl/harmonic.h>

Solver::Solver()
	:
	energy(make_shared<Energy>()),
	param_mutex(make_unique<mutex>()),
	mesh_mutex(make_unique<shared_timed_mutex>()),
	param_cv(make_unique<condition_variable>()),
	num_steps(2147483647)
{
}

void Solver::init(const MatX3& V, const MatX3i& F, float& time_Tutte_init)
{
	using namespace placeholders;
	this->F = F;
	auto start_init_time = bcp::Clock::now();
	computeTutte(V, F, uv);
	auto end_Tutte_time = bcp::Clock::now();
	time_Tutte_init = bcp::seconds(end_Tutte_time, start_init_time);
	cout << "Tutte init time consumption: " << time_Tutte_init << endl;
	energy->init(F.rows(), V, F);
	m_x = Eigen::Map<Vec>(uv.data(), uv.rows() * uv.cols());
	ext_x = m_x;
	eval_f = bind(&Energy::evaluate_f, energy, _1, _2);
	eval_fgh = bind(&Energy::evaluate_fgh, energy, _1, _2, _3, _4);
	auto internal_init_start_time = bcp::Clock::now();
	internal_init();
	auto internal_init_end_time = bcp::Clock::now();
	cout << "internal_init time consumption:" << bcp::seconds(internal_init_end_time, internal_init_start_time) << endl;
}

void Solver::init_from_input(const MatX3& V, const MatX3i& F, const MatX3& init_uv)
{
	using namespace placeholders;
	this->F = F;
	MatX3 tmp = init_uv;
	uv = tmp.block(0, 0, tmp.rows(), 2);;
	energy->init(F.rows(), V, F);
	m_x = Eigen::Map<Vec>(uv.data(), uv.rows() * uv.cols());
	ext_x = m_x;
	eval_f = bind(&Energy::evaluate_f, energy, _1, _2);
	eval_fgh = bind(&Energy::evaluate_fgh, energy, _1, _2, _3, _4);
	internal_init();
}

void Solver::computeTutte(const MatX3& V, const MatX3i& F, MatX2 &uv_init)
{
	Eigen::VectorXi bnd; Eigen::MatrixXd bnd_uv;
	igl::boundary_loop(F,bnd);
	if (bnd.size() < 3) throw std::runtime_error("Tutte initialization requires a mesh boundary");
	map_vertices_to_circle_area_normalize(V, F, bnd, bnd_uv);

 	igl::harmonic(V,F,bnd,bnd_uv,1,uv_init);
	if (igl::flipped_triangles(uv_init,F).size() != 0) 
		igl::harmonic(F,bnd,bnd_uv,1,uv_init); // use uniform laplacian
	if (igl::flipped_triangles(uv_init, F).size() != 0)
		cout << "Error!!!!!!!!!!!!! Flips in Tutte initialization!" << endl;

}
int Solver::run()
{
	is_running = true;
	halt = false;
	int steps = 0;
	do
	{
		ret = step();
		linesearch();
		update_external_mesh();
	} while ((a_parameter_was_updated || test_progress()) && !halt && ++steps < num_steps);
	is_running = false;
	cout << "solver stopped" << endl;
	return ret;
}

void Solver::stop()
{
	wait_for_param_slot();
	halt = true;
	release_param_slot();
}

void Solver::update_external_mesh()
{
	give_param_slot();
	unique_lock<shared_timed_mutex> lock(*mesh_mutex);
	internal_update_external_mesh();
	progressed = true;
}

void Solver::get_mesh(MatX2& X)
{
	unique_lock<shared_timed_mutex> lock(*mesh_mutex);
	X = Eigen::Map<MatX2>(ext_x.data(), ext_x.rows() / 2, 2);
	progressed = false;
}

void Solver::give_param_slot()
{
	a_parameter_was_updated = false;
	unique_lock<mutex> lock(*param_mutex);
	params_ready_to_update = true;
	param_cv->notify_one();
	while (wait_for_param_update)
	{
		param_cv->wait(lock);
		a_parameter_was_updated = true;
	}
	params_ready_to_update = false;
}

void Solver::wait_for_param_slot()
{
	unique_lock<mutex> lock(*param_mutex);
	wait_for_param_update = true;
	while (!params_ready_to_update && is_running)
		param_cv->wait_for(lock, chrono::milliseconds(50));
}

void Solver::release_param_slot()
{
	wait_for_param_update = false;
	param_cv->notify_one();
}

void Solver::map_vertices_to_circle_area_normalize(
	const MatX3& V,
	const MatX3i& F,
	const Eigen::VectorXi& bnd,
	Eigen::MatrixXd& UV) {

	double radius = sqrt(1.0 / (M_PI));

	// Get sorted list of boundary vertices
	std::vector<int> interior, map_ij;
	map_ij.resize(V.rows());
	interior.reserve(V.rows() - bnd.size());

	std::vector<bool> isOnBnd(V.rows(), false);
	for (int i = 0; i < bnd.size(); i++)
	{
		isOnBnd[bnd[i]] = true;
		map_ij[bnd[i]] = i;
	}

	for (int i = 0; i < (int)isOnBnd.size(); i++)
	{
		if (!isOnBnd[i])
		{
			map_ij[i] = interior.size();
			interior.push_back(i);
		}
	}

	// Map boundary to unit circle
	std::vector<double> len(bnd.size());
	len[0] = 0.;

	for (int i = 1; i < bnd.size(); i++)
	{
		len[i] = len[i - 1] + (V.row(bnd[i - 1]) - V.row(bnd[i])).norm();
	}
	double total_len = len[len.size() - 1] + (V.row(bnd[0]) - V.row(bnd[bnd.size() - 1])).norm();

	UV.resize(bnd.size(), 2);
	for (int i = 0; i < bnd.size(); i++)
	{
		double frac = len[i] * (2. * M_PI) / total_len;
		UV.row(map_ij[bnd[i]]) << radius * cos(frac), radius* sin(frac);
	}

}