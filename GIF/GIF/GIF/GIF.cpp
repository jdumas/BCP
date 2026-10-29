#include "GIF.h"
#include "Utils/MatlabInterface.h"
#include "Utils/MatlabGMMDataExchange.h"
#include <Eigen/Sparse>
#include "Utils/CompilationAccelerator.h"
#include "Utils/MeshAlgorithms.h"
#include <CGAL/Timer.h>
#include <random>
#include <numeric>
#include <vector>
#include <iostream>
#include <cstdlib>
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Polygon_2.h>
#include <CGAL/intersections.h>
#include <CGAL/enum.h>
#include <Isometric_Injective_Energies/apps/IsoTLC_2D.cpp>
#include <igl/boundary_loop.h>


using namespace Eigen;
using namespace std;

// Quote filenames passed to the numerical-script interpreter, rather than
// treating apostrophes in a filename as executable script syntax.
static std::string matlabStringLiteral(const std::string& value) {
	std::string quoted = "'";
	for (char c : value) {
		if (c == '\n' || c == '\r') throw std::runtime_error("Newlines in MAT filenames are not supported");
		quoted += c == '\'' ? "''" : std::string(1, c);
	}
	return quoted + "'";
}

using K = CGAL::Exact_predicates_exact_constructions_kernel;
using FT = K::FT;
using Point2 = K::Point_2;
using Segment = K::Segment_2;
using Line = K::Line_2;
using Polygon = CGAL::Polygon_2<K>;

// -----------------------------------------------------------
// Normalize polygon: remove duplicate last=first vertex if present
// -----------------------------------------------------------
void normalize(Polygon& poly) {
	if (poly.size() > 1) {
		auto first = poly.vertices_begin();
		auto last = poly.vertices_end();
		--last; // last vertex
		if (*first == *last) {
			poly.erase(last);
		}
	}
}

// -----------------------------------------------------------
// Clip polygon with half-plane (defined by edge p->q, keeping the LEFT side)
// -----------------------------------------------------------
Polygon clip_with_halfplane(const Polygon& poly, const Point2& p, const Point2& q) {
	Polygon out;
	if (poly.is_empty()) return out;

	Line L(p, q);

	auto n = poly.size();
	for (std::size_t i = 0; i < n; i++) {
		Point2 A = poly[i];
		Point2 B = poly[(i + 1) % n];

		auto sideA = L.oriented_side(A);
		auto sideB = L.oriented_side(B);

		bool inA = (sideA != CGAL::ON_NEGATIVE_SIDE);
		bool inB = (sideB != CGAL::ON_NEGATIVE_SIDE);

		if (inA && inB) {
			// Both inside
			out.push_back(B);
		}
		else if (inA && !inB) {
			// Leaving half-plane
			auto inter = CGAL::intersection(Line(A, B), L);
			if (const Point2* ip = boost::get<Point2>(&*inter))
				out.push_back(*ip);
		}
		else if (!inA && inB) {
			// Entering half-plane
			auto inter = CGAL::intersection(Line(A, B), L);
			if (const Point2* ip = boost::get<Point2>(&*inter))
				out.push_back(*ip);
			out.push_back(B);
		}
		// else: both out add nothing
	}

	if (!out.is_empty() && out.orientation() == CGAL::CLOCKWISE)
		out.reverse_orientation();

	normalize(out);
	return out;
}

// -----------------------------------------------------------
// Kernel of polygon (intersection of all half-planes of edges)
// -----------------------------------------------------------
Polygon kernel_of_polygon(const Polygon& P) {
	// Start with a huge bounding box
	FT maxc = 0;
	for (auto v = P.vertices_begin(); v != P.vertices_end(); ++v) {
		maxc = std::max(maxc, CGAL::abs(v->x()));
		maxc = std::max(maxc, CGAL::abs(v->y()));
	}
	FT margin = FT(10);
	FT INF = (maxc + FT(1)) * margin;

	Polygon ker;
	ker.push_back(Point2(-INF, -INF));
	ker.push_back(Point2(INF, -INF));
	ker.push_back(Point2(INF, INF));
	ker.push_back(Point2(-INF, INF));

	// Clip with each edge half-plane (keep left of each edge)
	for (std::size_t i = 0; i < P.size(); i++) {
		Point2 a = P[i];
		Point2 b = P[(i + 1) % P.size()];
		ker = clip_with_halfplane(ker, a, b);
		if (ker.is_empty()) break;
	}

	normalize(ker);
	return ker;
}

// Area centroid of (convex) polygon; returns false if degenerate.
bool area_centroid(const Polygon& poly, Point2& C) {
	if (poly.size() < 3) return false;
	FT S = 0, Cx = 0, Cy = 0;
	for (int i = 0; i < (int)poly.size(); ++i) {
		const Point2& p = poly[i];
		const Point2& q = poly[(i + 1) % poly.size()];
		FT cr = p.x() * q.y() - q.x() * p.y();
		S += cr;
		Cx += (p.x() + q.x()) * cr;
		Cy += (p.y() + q.y()) * cr;
	}
	if (S == FT(0)) return false; // degenerate area
	FT A2 = S * FT(0.5); // signed area
	C = Point2(Cx / (FT(6) * A2), Cy / (FT(6) * A2));
	return true;
}

double extract_symmetric_dirichlet_energy(Facet_const_handle& face)
{
	// Assuming the triangle is not degenerate
	Complex e1, e2, e3;
	bool res = face->computeHalfedgesInLocalCoords(e1, e2, e3, false);

	Complex t1(imag(e1), -real(e1));
	Complex t2(imag(e2), -real(e2));
	Complex t3(imag(e3), -real(e3));

	Complex f1 = face->halfedge()->next()->vertex()->uvC();
	Complex f2 = face->halfedge()->opposite()->vertex()->uvC();
	Complex f3 = face->halfedge()->vertex()->uvC();

	double absFzbar = abs((f1 * t1 + f2 * t2 + f3 * t3) / (4 * face->area()));
	double absFz = abs((f1 * conj(t1) + f2 * conj(t2) + f3 * conj(t3)) / (4 * face->area()));

	double SIGMA = absFz + absFzbar;
	double sigma = absFz - absFzbar;

	double sd = 0.5 * (SIGMA * SIGMA + (1 / (SIGMA * SIGMA)) + sigma * sigma + (1 / (sigma * sigma)));

	return sd;
}

GIF::GIF() {
	std::cout.rdbuf(std::cerr.rdbuf()); //added since cout does not print to output window
}
bool GIF::run(std::string obj_path, std::string uv_mat_path, bool run_IDT, bool allow_MVC_fix, bool allow_local_fix, std::string edge_lengths_mat_path, int interior_faces, int boundary_segment_size, double curvature_meta_vertices_rate, double outer_termination_condition_rate, double energy_related_termination_condition)
{
	std::string cmd = "clear;";
	MatlabInterface::GetEngine().EvalToCout(cmd.c_str());
	// Pass the policy explicitly: a Windows MATLAB COM server may predate this
	// process and therefore not inherit BCP_GIF_DEVICE from its environment.
	const char* gpuMode = std::getenv("BCP_GIF_DEVICE");
	if (!gpuMode) gpuMode = "auto";
	auto& engine = MatlabInterface::GetEngine();
	if (engine.SetEngineStringArray("BCP_GPU_MODE", 1, &gpuMode) != 0)
		throw std::runtime_error("Cannot set GIF GPU policy");
	const char* evaluator = std::getenv("BCP_GIF_BACKEND");
	if (!evaluator) evaluator = "auto";
	if (engine.SetEngineStringArray("BCP_EVAL_BACKEND", 1, &evaluator) != 0)
		throw std::runtime_error("Cannot set GIF evaluator backend");
	std::cout << engine.EvalToString("[BCP_USE_GPU, BCP_EVAL_BACKEND] = bcp_compute_policy(BCP_GPU_MODE, BCP_EVAL_BACKEND);", false);
	double useGPU = 0;
	if (engine.GetEngineRealMatrix("BCP_USE_GPU", 1, 1, &useGPU) != 0)
		throw std::runtime_error("GIF GPU selection failed; check the numerical runtime diagnostics");
	engine.Eval("GIF.useGPU = logical(BCP_USE_GPU); GIF.computeBackend = BCP_EVAL_BACKEND; clear BCP_GPU_MODE BCP_USE_GPU BCP_EVAL_BACKEND;");
	std::cout << "GIF compute device: " << (useGPU != 0 ? "gpu" : "cpu") << "\n";
	mNumInternalFaces = interior_faces;
	mSegSize = boundary_segment_size;
	mPercentCurvatureMeta = curvature_meta_vertices_rate;
	mOuterRateTerminationCondition = outer_termination_condition_rate;
	mEnergyRelatedTerminationCondition = energy_related_termination_condition;
	mUseCurvatureMetaVertices = true; // this option adds the vertices with the largest Gaussian curvature to the meta vertices set
	mRunIDT = run_IDT;
	mAllowMVCFix = allow_MVC_fix;
	mMaxFoldoversForIsoTLC = 25;
	mAllowLocallFix = allow_local_fix;
	mDelaunayOutputs.clear();
	mDelaunayOutputs.resize(10, 1);

	//Load mesh
	bool res = loadMesh(obj_path, edge_lengths_mat_path);
	if (!res) {
		cout << "Failed to load mesh" << endl;
		return false;
	}
	//Get borders of mesh
	CGAL_Borders borders(mCgalMesh);
	if (borders.genus() != 0) {
		cout << "The code only supports genus 0 for now" << endl;
		return false;
	}
	if (borders.numBorders() != 1) {
		cout << "The code only supports models with one border" << endl;
		return false;
	}
	//Initialize mesh and arrays, and sent to MATLAB
	res = initialize(borders);
	if (!res) {
		cout << "Failed to initialize mesh" << endl;
		return false;
	}

	//Create KKT, calculate basis, run Newton, and find all UV's
	CGAL::Timer totalTime;
	totalTime.start();
	res = runAlgorithm(borders);
	if (!res) {
		cout << "Failed in algorithm" << endl;
		return false;
	}

	//Update UV's for halfedges in the mesh
	updatHalfedgeUVs();

	//Decide if failed, succeeded or partially succeeded
	getResult();
	if (!mRunIDT)
		fixOutliersInFinalResult();
	if (mPerformFinalGlobalScaling)
	{
		CGAL::Timer globalScalingTime;
		globalScalingTime.start();
		MatlabInterface::GetEngine().EvalToCout("global_scaling");
		globalScalingTime.stop();
		cout << "global scaling time: " << globalScalingTime.time() << endl;
		MatlabGMMDataExchange::GetEngineDenseMatrix("array1", mUVs);
		totalTime.stop();
		updatHalfedgeUVs();
	}
	else
		totalTime.stop();
	
	std::cout << "Total time: " << totalTime.time() << "\n";
	mParameters_matrix(4, 0) = totalTime.time();

	GMMDenseColMatrix uvs(mCgalMesh.size_of_vertices(), 2);
	for_each_vertex(vi, mCgalMesh)
	{
		uvs(vi->index(), 0) = real(vi->uvC());
		uvs(vi->index(), 1) = imag(vi->uvC());
	}

	vector<double> absolute_errors, relative_errors;

	for (auto he = mCgalMesh.edges_begin(); he != mCgalMesh.edges_end(); he++)
		if (!he->is_border_edge()) {
			double first_edge_length = he->targetMetric();
			double second_edge_length = he->opposite()->targetMetric();
			double absolute_error = abs(first_edge_length - second_edge_length);
			double relative_error = absolute_error / ((first_edge_length + second_edge_length) / 2);
			absolute_errors.push_back(absolute_error);
			relative_errors.push_back(relative_error);
		}

	double absolute_erros_sum = 0.0;
	for (double num : absolute_errors) {
		absolute_erros_sum += num;
	}
	double absolute_erros_average = absolute_erros_sum / absolute_errors.size();
	double sum = 0.0;
	for (double num : absolute_errors) {
		sum += (num - absolute_erros_average) * (num - absolute_erros_average);
	}
	double absolute_erros_std = std::sqrt(sum / absolute_errors.size());

	mParameters_matrix(15, 0) = absolute_erros_average;
	mParameters_matrix(16, 0) = absolute_erros_std;


	double relative_erros_sum = 0.0;
	for (double num : relative_errors) {
		relative_erros_sum += num;
	}
	double relative_erros_average = relative_erros_sum / relative_errors.size();
	sum = 0.0;
	for (double num : relative_errors) {
		sum += (num - relative_erros_average) * (num - relative_erros_average);
	}
	double relative_erros_std = std::sqrt(sum / relative_errors.size());

	mParameters_matrix(17, 0) = relative_erros_average;
	mParameters_matrix(18, 0) = relative_erros_std;

	mAverageEdgeLengths.clear();
	mAverageEdgeLengths.resize(mCgalMesh.size_of_facets(), 3);

	if (!mRunIDT)
		for (auto he = mCgalMesh.edges_begin(); he != mCgalMesh.edges_end(); he++)
			if (!he->is_border_edge()) {
				double first_edge_length = he->targetMetric();
				double second_edge_length = he->opposite()->targetMetric();
				double mean_edge_length = (first_edge_length + second_edge_length) / 2.0;
				he->targetMetric() = mean_edge_length;
				he->opposite()->targetMetric() = mean_edge_length;
			}
	for_each_facet(f, mCgalMesh)
	{
		Halfedge_handle h1 = f->halfedge();
		Halfedge_handle h2 = h1->next();
		Halfedge_handle h3 = h2->next();
		mAverageEdgeLengths(f->index(), 2) = h1->targetMetric();
		mAverageEdgeLengths(f->index(), 0) = h2->targetMetric();
		mAverageEdgeLengths(f->index(), 1) = h3->targetMetric();
	}

	double e_sd = 0;
	double total_area = 0;
	for_each_const_facet(f, mCgalMesh)
	{
		double cur_e_sd = extract_symmetric_dirichlet_energy(f);
		double cur_area = f->area();
		double cur_e_sd_area = cur_area * cur_e_sd;
		e_sd += cur_e_sd_area;
		total_area += cur_area;
	}
	e_sd = e_sd / total_area;
	cout << "Final Esd = " << e_sd << endl;


	MatlabGMMDataExchange::SetEngineDenseMatrix("uvs_ordered", uvs);
	MatlabGMMDataExchange::SetEngineDenseMatrix("IDT_outputs", mDelaunayOutputs);
	MatlabGMMDataExchange::SetEngineDenseMatrix("parametersMatrix", mParameters_matrix);
	MatlabGMMDataExchange::SetEngineDenseMatrix("flipsData", mFlipsData);
	MatlabGMMDataExchange::SetEngineDenseMatrix("mAverageEdgeLengths", mAverageEdgeLengths);
	
	std::string cmd_save_uv = "save(" + matlabStringLiteral(uv_mat_path) + ", 'uvs_ordered', 'IDT_outputs', 'parametersMatrix', 'mAverageEdgeLengths', 'flipsData');";
	MatlabInterface::GetEngine().Eval(cmd_save_uv.c_str());
	cout << "GIF succeed!" << endl;

	//openReportWindow();
	return true;
}


bool GIF::initialize(CGAL_Borders& borders)
{
	UpdateIndexOfHEinSystem();
	getBordersMapAndSetMetaVertices(borders);
	bool hasBorder = mConesAndMetaMap.size() > 0;
	if (!hasBorder) {
		cout << "Model must have border" << endl;
		return false;
	}

	int numOfBorderVertices = 0;
	for (int i = 0; i < borders.numBorders(); i++)
		numOfBorderVertices += borders.numVertices();
	mNumDOF = hasBorder ? mConesAndMetaMap.size() : mConesAndMetaMap.size() - 1;
	mNumScaffoldDOF = min(max((int)(1.1 * (double)mNumDOF), (mNumDOF + 3)), (25 + mNumDOF));
	bool res = TransferBorderFacesAndVerticesToMatlab();
	if (!res) {
		cout << "Failed to transfer to MATLAB" << endl;
		return false;
	}
	return true;
}



bool GIF::loadMesh(std::string obj_path, std::string edge_lengths_mat_path)
{
	// -----load mesh-------
	MeshBuffer m;
	mObjpath = obj_path;
	mOperationMode = 1;
	mPerformFinalGlobalScaling = false;

	cout << "Loading mesh information, creating CGAL mesh and transferring mesh to matlab..." << endl;
	bool res = Parser::loadOBJ(mObjpath.c_str(), &mMeshBuffer, &m);
	if (!res) return false;

	// -----create cgal mesh-------
	std::vector<Point_3> pVec;
	pVec.resize(mMeshBuffer.positions.size());
	for (int i = 0; i < (int)pVec.size(); ++i)
		pVec[i] = Point_3(mMeshBuffer.positions[i][0], mMeshBuffer.positions[i][1], mMeshBuffer.positions[i][2]);
	std::vector<unsigned int> faces;
	faces.resize((int)mMeshBuffer.idx_pos.size());
	for (int i = 0; i < (int)faces.size(); ++i)
		faces[i] = mMeshBuffer.idx_pos[i];
	MeshBuilder<HDS, Kernel> meshBuilder(&pVec, (std::vector<int>*)(&faces), NULL);//we assume here that the indices are not negative!
	mCgalMesh.clear();
	mCgalMesh.delegate(meshBuilder);
	mCgalMesh.updateAllGlobalIndices();


	if (mSegSize == -1)
		if (mCgalMesh.size_of_facets() > 500000)
			mSegSize = ceil(double(mCgalMesh.size_of_border_edges()) / 250);
		else
			mSegSize = 10;
		if (mCgalMesh.size_of_border_edges() < 100)
			mSegSize = 1;

		mSegSize = max(mSegSize, int(ceil(double(mCgalMesh.size_of_border_edges()) / 250)));  // Make sure extremely large boundaries don't exceed 250.

	
	if (mNumInternalFaces == -1)
		if (mCgalMesh.size_of_facets() < 8000)
			mNumInternalFaces = 50;
		else if (mCgalMesh.size_of_facets() >= 8000 && mCgalMesh.size_of_facets() < 250000)
			mNumInternalFaces = 300;
		else
			mNumInternalFaces = 500;

	if (edge_lengths_mat_path == "")
	{
		mCgalMesh.updateTargetMetricFrom3dEmbedding();
		cout << "Using original mesh's edge lengths" << endl;
	}
	else
	{
		std::string cmd = "load(" + matlabStringLiteral(edge_lengths_mat_path) + ")";
		MatlabInterface::GetEngine().EvalToCout(cmd.c_str());
		GMMDenseColMatrix edge_lengths(mCgalMesh.size_of_facets(), 3);
		MatlabGMMDataExchange::GetEngineDenseMatrix("edge_lengths", edge_lengths);

		int j = 0;
		for_each_facet(f, mCgalMesh)
		{
			Halfedge_handle h1 = f->halfedge();
			Halfedge_handle h2 = h1->next();
			Halfedge_handle h3 = h2->next();

			h1->targetMetric() = edge_lengths(j, 2);
			h2->targetMetric() = edge_lengths(j, 0);
			h3->targetMetric() = edge_lengths(j, 1);

			if (h1->is_border_edge())
				h1->opposite()->targetMetric() = edge_lengths(j, 2);
			if (h2->is_border_edge())
				h2->opposite()->targetMetric() = edge_lengths(j, 0);
			if (h3->is_border_edge())
				h3->opposite()->targetMetric() = edge_lengths(j, 1);
			j++;
		}
	}
	if (mRunIDT)
		intrinsicDelaunayTriangulationaBasedAngles(false);
	return true;
}
void GIF::getBordersMapAndSetMetaVertices(CGAL_Borders& borders)
{
	mConesAndMetaMap.clear();
	mIndicesOfMetaVerticesInUVbyGeneralIndex.clear();
	mIndicesOfMetaVerticesInUVbyGeneralIndex.resize(mCgalMesh.size_of_vertices());
	mIndicesOfAllBorderVerticesInUVbyGeneralIndex.clear();
	mIndicesOfAllBorderVerticesInUVbyGeneralIndex.resize(mCgalMesh.size_of_vertices());
	mGeneralIndicesUVbyIndicesOfAllBorderVertices.clear();
	mGeneralIndicesUVbyIndicesOfAllBorderVertices.resize(mCgalMesh.size_of_border_edges());
	mIsMeta.clear();
	mIsMeta.resize(mCgalMesh.size_of_vertices(), false);
	getVerticesThatMustBeMeta(borders);
	mMetaVerticesInBorderByHalfEdge.clear();
	mMetaVerticesInBorderByHalfEdge.resize(borders.numBorders());
	mConesVertices.clear();
	int ind1 = 0;
	//find meta vertices

	int borderSegmentSize = mSegSize;
	std::vector<Halfedge_handle>& currBorder = mMetaVerticesInBorderByHalfEdge[0];

	std::vector<Halfedge_handle> borderPath; //halfedges on boundary
	borders.getBorderHalfEdges(borders.vertex(0, 0), borderPath);
	int min_vertex_index = 0;
	for (int j = 0; j < borderPath.size(); j++)
		if (borderPath[j]->vertex()->index() < borderPath[min_vertex_index]->vertex()->index())
			min_vertex_index = j;
	std::rotate(borderPath.begin(), borderPath.begin() + (min_vertex_index + 1), borderPath.end());



	int curvatueSamples = int(mPercentCurvatureMeta * borderPath.size());
	std::vector<double> gaussianCurvatures;
	for (int j = 0; j < borderPath.size(); j++) {

		Halfedge_handle HD = borderPath[j];
		gaussianCurvatures.push_back(abs(HD->vertex()->gaussianCurvature()));
	}
	for (int j = 0; j < curvatueSamples; j++) {
		auto i = std::max_element(gaussianCurvatures.begin(), gaussianCurvatures.end());
		int i1 = std::distance(gaussianCurvatures.begin(), i);
		Halfedge_handle HD = borderPath[i1];
		mIsMeta[HD->vertex()->index()] = true;
		gaussianCurvatures[i1] = 0.0;
	}


	if (borderPath.size() < 3 * mSegSize) {//make sure we have at least 3 meta vertices in this border
		borderSegmentSize = borderPath.size() / 3;
	}

	int verticesInSeg = borderSegmentSize;
	for (int j = borderPath.size() - 1; j >= 0; j--) {

		Halfedge_handle HD = borderPath[j];
		mIndicesOfAllBorderVerticesInUVbyGeneralIndex[HD->vertex()->index()] = ind1;
		mGeneralIndicesUVbyIndicesOfAllBorderVertices[ind1] = HD->vertex()->index();
		mBoundaryVertices.insert(mIndexOfHEinSystem[HD->index()]);
		mBoundaryVerticesMap.push_back(HD->vertex());
		ind1++;
		if (verticesInSeg >= borderSegmentSize || mIsMeta[HD->vertex()->index()]) {//meta vertex
			currBorder.push_back(HD);
			verticesInSeg = 1;
			mIndicesOfMetaVerticesInUVbyGeneralIndex[HD->vertex()->index()] = mConesAndMetaMap.size();
			mConesAndMetaMap.push_back(HD->vertex());
			mConesVertices.insert(mIndexOfHEinSystem[HD->index()]);
			mIsMeta[HD->vertex()->index()] = true;
		}
		else
			verticesInSeg++;
	}
	currBorder.push_back(currBorder[0]);

}

void GIF::intrinsicDelaunayTriangulationaBasedAngles(bool allow_non_planar_edge_flips)
{
	// Retriangulating using Ptolemy flips
	// Assuming edge lengths are consistent on both sides of each edge!
	// Triangle inequality doesn't have to be satisfied
	cout << "Applying hyperbolic Delaunay Triangulation..." << endl;
	const double epsilon = 2e-14;
	CGAL::Timer idtTime;
	idtTime.start();
	int bDomain = 0;
	vector<bool> bMarked(mCgalMesh.size_of_halfedges() / 2, false);
	deque<Halfedge_handle> s;
	for (auto he = mCgalMesh.edges_begin(); he != mCgalMesh.edges_end(); he++)
		if (!he->is_border_edge()) {
			s.push_back(he);
			bMarked[he->index() / 2] = true;
		}
	int nFlips = 0;
	int nUnabledFlips = 0;
	//double max_diff = 0;
	while (!s.empty()) {
		auto he = s.back();
		s.pop_back();
		bMarked[he->index() / 2] = false;
		if ((he->cot(true) + he->opposite()->cot(true)) > -2e-14)
			continue;

		// flip
		const double l_mi = he->prev()->targetMetric();
		const double l_mj = he->next()->targetMetric();
		const double l_ij = he->targetMetric();
		const double l_ji_old = he->opposite()->targetMetric();
		const double l_kj_old = he->opposite()->prev()->targetMetric();
		const double l_ki_old = he->opposite()->next()->targetMetric();
		const double scale_ijk = l_ij / l_ji_old;

		// update ijk triangle helfedges
		he->opposite()->targetMetric() = scale_ijk * l_ji_old;
		he->opposite()->prev()->targetMetric() = scale_ijk * l_kj_old;
		he->opposite()->next()->targetMetric() = scale_ijk * l_ki_old;

		// read target metric again after update
		const double l_ji = he->opposite()->targetMetric();
		const double l_kj = he->opposite()->prev()->targetMetric();
		const double l_ki = he->opposite()->next()->targetMetric();

		/*cout << "mi: " << he->prev()->opposite()->vertex()->index() << " -> " << he->prev()->vertex()->index() << endl;
		cout << "jm: " << he->next()->opposite()->vertex()->index() << " -> " << he->next()->vertex()->index() << endl;
		cout << "ij: " << he->opposite()->vertex()->index() << " -> " << he->vertex()->index() << endl;
		cout << "ji: " << he->opposite()->opposite()->vertex()->index() << " -> " << he->opposite()->vertex()->index() << endl;
		cout << "kj: " << he->opposite()->prev()->opposite()->vertex()->index() << " -> " << he->opposite()->prev()->vertex()->index() << endl;
		cout << "ik: " << he->opposite()->next()->opposite()->vertex()->index() << " -> " << he->opposite()->next()->vertex()->index() << endl;*/

		
		double l_im_sq = l_mi * l_mi;
		double l_jk_sq = l_kj * l_kj;
		double l_jm_sq = l_mj * l_mj;
		double l_ki_sq = l_ki * l_ki;
		double l_ij_sq = l_ij * l_ij;
		double s_ijk = (l_ij + l_kj + l_ki) / 2.0;
		double a_ijk = sqrt(s_ijk * (s_ijk - l_ij) * (s_ijk - l_kj) * (s_ijk - l_ki));
		double s_jim = (l_ij + l_mi + l_mj) / 2.0;
		double a_jim = sqrt(s_jim * (s_jim - l_ij) * (s_jim - l_mi) * (s_jim - l_mj));
		double l_km_sq = 0.5 * (l_im_sq + l_jk_sq + l_jm_sq + l_ki_sq - l_ij_sq + (((l_im_sq - l_jm_sq) * (l_jk_sq - l_ki_sq)) / l_ij_sq)) + ((8.0 * a_ijk * a_jim) / l_ij_sq);
		const double l_km = sqrt(l_km_sq);

		/*const double cot_mi = he->prev()->cot(true);
		const double cot_mj = he->next()->cot(true);
		const double cot_ij = he->cot(true);
		const double cot_ji = he->opposite()->cot(true);
		const double cot_kj = he->opposite()->prev()->cot(true);
		const double cot_ki = he->opposite()->next()->cot(true);*/

		

		he = mCgalMesh.flip_edge(he);

		if (!allow_non_planar_edge_flips)
		{
			/*if (nUnabledFlips > 25)
			{
				he = mCgalMesh.flip_edge(he);
				he->targetMetric() = l_ij;
				he->opposite()->targetMetric() = l_ij;
				continue;
			}*/
			//here we check if the mesh remains planar after the edge flip

			//if (he->vertex()->index() == he->opposite()->vertex()->index())  // here we check if the new edge connects a vertex to itself
			//{
			//	cout << "irregular mesh operation" << endl;
			//	cout << he->vertex()->point() << endl;
			//	cout << mCgalMesh.size_of_vertices() << endl;
			//	he = mCgalMesh.flip_edge(he);
			//	he->targetMetric() = l_ij;
			//	he->opposite()->targetMetric() = l_ij;
			//	nUnabledFlips++;

			//	cout << "mi: " << l_mi << endl;
			//	cout << "jm: " << l_mj << endl;
			//	cout << "ij: " << l_ij << endl;
			//	cout << "ji: " << l_ji << endl;
			//	cout << "kj: " << l_kj << endl;
			//	cout << "ik: " << l_ki << endl;

			//	cout << "Splitting flipped edge..." << endl;
			//	const double he_original_length = he->targetMetric();
			//	const double he_oposite_original_length = he->opposite()->targetMetric();
			//	//Point middle_point = Point((he->vertex()->point().x() + he->opposite()->vertex()->point().x()) / 2, (he->vertex()->point().y() + he->opposite()->vertex()->point().y()) / 2, (he->vertex()->point().z() + he->opposite()->vertex()->point().z()) / 2);
			//	const double pj_length = 0.5 * sqrt(2 * he->next()->targetMetric() * he->next()->targetMetric() + 2 * he->prev()->targetMetric() * he->prev()->targetMetric() - he->targetMetric() * he->targetMetric());
			//	const double pi_length = 0.5 * sqrt(2 * he->opposite()->next()->targetMetric() * he->opposite()->next()->targetMetric() + 2 * he->opposite()->prev()->targetMetric() * he->opposite()->prev()->targetMetric() - he->opposite()->targetMetric() * he->opposite()->targetMetric());
			//	Halfedge_handle he_kp = mCgalMesh.split_edge(he); // he becomes h_pm
			//	Halfedge_handle he_pj = mCgalMesh.split_facet(he_kp, he_kp->next()->next());
			//	Halfedge_handle he_pi = mCgalMesh.split_facet(he->opposite(), he->opposite()->next()->next());
			//	//he_kp->vertex()->point() = middle_point;
			//	he_kp->targetMetric() = he_original_length / 2;
			//	he_kp->opposite()->targetMetric() = he_oposite_original_length / 2;
			//	he->targetMetric() = he_original_length / 2;
			//	he->opposite()->targetMetric() = he_oposite_original_length / 2;
			//	he_pj->targetMetric() = pj_length;
			//	he_pj->opposite()->targetMetric() = pj_length;
			//	he_pi->targetMetric() = pi_length;
			//	he_pi->opposite()->targetMetric() = pi_length;
			//	vector<Halfedge_handle> ne;
			//	for (auto he = mCgalMesh.edges_begin(); he != mCgalMesh.edges_end(); he++)
			//		if (!he->is_border_edge()) {
			//			s.push_back(he);
			//			//bMarked[he->index() / 2] = true;
			//		}


			//	continue;
			//}


			// here we check if there exist 2 edges connecting the same 2 vertices, creating a "hat" which is non-planar
			Mesh::Halfedge_around_vertex_circulator h = he->vertex()->vertex_begin();
			const Mesh::Halfedge_around_vertex_circulator hEnd = h;
			int sameEdgesAfterFlip = 0;
			CGAL_For_all(h, hEnd)
			{
				if (he->opposite()->vertex()->index() == h->opposite()->vertex()->index())
					sameEdgesAfterFlip++;
			}
			if (sameEdgesAfterFlip != 1)
			{
				cout << sameEdgesAfterFlip << " edges between 2 vertices!" << endl;
				nUnabledFlips++;


				cout << sameEdgesAfterFlip << " edges between 2 vertices!" << endl;
				cout << "Splitting flipped edge..." << endl;
				he->targetMetric() = l_km;
				he->opposite()->targetMetric() = l_km;
				const double he_original_length = he->targetMetric();
				const double he_oposite_original_length = he->opposite()->targetMetric();
				//Point middle_point = Point((he->vertex()->point().x() + he->opposite()->vertex()->point().x()) / 2, (he->vertex()->point().y() + he->opposite()->vertex()->point().y()) / 2, (he->vertex()->point().z() + he->opposite()->vertex()->point().z()) / 2);
				const double pj_length = 0.5 * sqrt(2 * he->next()->targetMetric() * he->next()->targetMetric() + 2 * he->prev()->targetMetric() * he->prev()->targetMetric() - he->targetMetric() * he->targetMetric());
				const double pi_length = 0.5 * sqrt(2 * he->opposite()->next()->targetMetric() * he->opposite()->next()->targetMetric() + 2 * he->opposite()->prev()->targetMetric() * he->opposite()->prev()->targetMetric() - he->opposite()->targetMetric() * he->opposite()->targetMetric());
				/*cout << he->opposite()->next()->targetMetric() << endl;
				cout << he->opposite()->prev()->targetMetric() << endl;
				cout << he->opposite()->targetMetric() << endl;*/
				Halfedge_handle he_kp = mCgalMesh.split_edge(he); // he becomes h_pm
				Halfedge_handle he_pj = mCgalMesh.split_facet(he_kp, he_kp->next()->next());
				Halfedge_handle he_pi = mCgalMesh.split_facet(he->opposite(), he->opposite()->next()->next());
				//he_kp->vertex()->point() = middle_point;
				he_kp->targetMetric() = he_original_length / 2;
				he_kp->opposite()->targetMetric() = he_oposite_original_length / 2;
				he->targetMetric() = he_original_length / 2;
				he->opposite()->targetMetric() = he_oposite_original_length / 2;
				he_pj->targetMetric() = pj_length;
				he_pj->opposite()->targetMetric() = pj_length;
				he_pi->targetMetric() = pi_length;
				he_pi->opposite()->targetMetric() = pi_length;
				/*cout << he_pi->targetMetric() << endl;
				cout << he_pi->next()->targetMetric() << endl;
				cout << he_pi->prev()->targetMetric() << endl;
				cout << ((-he_pi->targetMetric() + he_pi->next()->targetMetric() + he_pi->prev()->targetMetric() > 0));*/
				/*cout << ((he_pi->targetMetric() + he_pi->next()->targetMetric() - he_pi->prev()->targetMetric() > 0) && (he_pi->targetMetric() - he_pi->next()->targetMetric() + he_pi->prev()->targetMetric() > 0) && (-he_pi->targetMetric() + he_pi->next()->targetMetric() + he_pi->prev()->targetMetric() > 0)) && ((he_pi->opposite()->targetMetric() + he_pi->opposite()->next()->targetMetric() - he_pi->opposite()->prev()->targetMetric() > 0) && (he_pi->targetMetric() - he_pi->opposite()->next()->targetMetric() + he_pi->opposite()->prev()->targetMetric() > 0) && (-he_pi->opposite()->targetMetric() + he_pi->opposite()->next()->targetMetric() + he_pi->opposite()->prev()->targetMetric() > 0));
				cout << ((he_pj->targetMetric() + he_pj->next()->targetMetric() - he_pj->prev()->targetMetric() > 0) && (he_pj->targetMetric() - he_pj->next()->targetMetric() + he_pj->prev()->targetMetric() > 0) && (-he_pj->targetMetric() + he_pj->next()->targetMetric() + he_pj->prev()->targetMetric() > 0)) && ((he_pj->opposite()->targetMetric() + he_pj->opposite()->next()->targetMetric() - he_pj->opposite()->prev()->targetMetric() > 0) && (he_pj->targetMetric() - he_pj->opposite()->next()->targetMetric() + he_pj->opposite()->prev()->targetMetric() > 0) && (-he_pj->opposite()->targetMetric() + he_pj->opposite()->next()->targetMetric() + he_pj->opposite()->prev()->targetMetric() > 0));*/

				vector<Halfedge_handle> ne;
				for (auto he = mCgalMesh.edges_begin(); he != mCgalMesh.edges_end(); he++)
					if (!he->is_border_edge()) {
						s.push_back(he);
						//bMarked[he->index() / 2] = true;
					}


				continue;
			}
		}

		// after the flip the edge is in the direction of the triangle
		he->targetMetric() = l_km;
		he->opposite()->targetMetric() = l_km;


		/*double cot_km = he->cot(true);
		double cot_km_formula = (cot_mj * cot_kj - 1) / (cot_mj + cot_kj);
		double cot_mk = he->opposite()->cot(true);
		double cot_mk_formula = (cot_mi * cot_ki - 1) / (cot_mi + cot_ki);*/
		/*cout << cot_mk << endl;
		cout << cot_mk_formula << endl;

		cout << he->prev()->opposite()->vertex()->index() << endl;
		cout << he->prev()->vertex()->index() << endl;*/
		//double cot_ki_new = he->prev()->cot(true);
		/*cout << "anglw_ji = " << 180.0 / M_PI * (M_PI / 2.0 - std::atan(cot_ji)) << endl;
		cout << "anglw_ki = " << 180.0 / M_PI * (M_PI / 2.0 - std::atan(cot_ki)) << endl;
		cout << "anglw_mk_formula = " << 180.0 / M_PI * (M_PI / 2.0 - std::atan(cot_mk_formula)) << endl;
		cout << "anglw_mi = " << 180.0 / M_PI * (M_PI / 2.0 - std::atan(cot_mi)) << endl;
		cout << "anglw_ij = " << 180.0 / M_PI * (M_PI / 2.0 - std::atan(cot_ij)) << endl;*/
		/*cout << "alpha_1 = " << 180.0 / M_PI * (M_PI / 2.0 - std::atan(cot_ij)) << endl;
		cout << "alpha_2 = " << 180.0 / M_PI * (M_PI / 2.0 - std::atan(cot_mi)) << endl;
		cout << "alpha_3 = " << 180.0 / M_PI * (M_PI / 2.0 - std::atan(cot_mj)) << endl;
		cout << "beta_1 = " << 180.0 / M_PI * (M_PI / 2.0 - std::atan(cot_ji)) << endl;
		cout << "beta_2 = " << 180.0 / M_PI * (M_PI / 2.0 - std::atan(cot_ki)) << endl;
		cout << "beta_3 = " << 180.0 / M_PI * (M_PI / 2.0 - std::atan(cot_kj)) << endl;
		cout << "cot alpha_1 = " << cot_ij << endl;
		cout << "cot alpha_2 = " << cot_mi << endl;
		cout << "cot alpha_3 = " << cot_mj << endl;
		cout << "cot beta_1 = " << cot_ji << endl;
		cout << "cot beta_2 = " << cot_ki << endl;
		cout << "cot beta_3 = " << cot_kj << endl;
		cout << "cot alpha_3 + beta_3 = " << cot_mk_formula << endl;
		cout << "cot alpha_3 + beta_3 = " << cot_km_formula << endl;
		double alpha_2 = 180.0 / M_PI * (M_PI / 2.0 - std::atan(cot_mi));*/
		/*double cot_ki_formula = sqrt((1 + cot_ij * cot_ij) * (1 + cot_ki * cot_ki) * (1 + cot_km_formula * cot_km_formula) / ((1 + cot_mi * cot_mi) * (1 + cot_ji * cot_ji))) - cot_km_formula;
		max_diff = max(abs(cot_ki_new - cot_ki_formula), max_diff);*/



		bool is_cot_possitive = ((he->cot(true) + he->opposite()->cot(true)) > 0.0);
		assert(is_cot_possitive);
		bool is_triangle_inequality_satisfied_1 = ((l_mj + l_kj - l_km > 0) && (l_mj - l_kj + l_km > 0) && (-l_mj + l_kj + l_km > 0)) && ((l_mi + l_ki - l_km > 0) && (l_mi - l_ki + l_km > 0) && (-l_mi + l_ki + l_km > 0));
		assert(is_triangle_inequality_satisfied_1);
		if (!is_cot_possitive || !is_triangle_inequality_satisfied_1)
			cout << "problem!" << endl;
		nFlips++;

		// add four neighbors to stack
		vector<Halfedge_handle> ne;
		for (int i = 0; i < 2; ++i) {

			ne.push_back(he->next());
			ne.push_back(he->prev());
			he = he->opposite();
		}
		for (auto he : ne)
		{
			if (!he->is_border_edge() && !bMarked[he->index() / 2])
			{
				s.push_back(he);
				bMarked[he->index() / 2] = true;
			}
		}
	}
	idtTime.stop();
	//cout << max_diff << endl;
	cout << "The number of flips IDT performed is " << nFlips << endl;
	cout << "Intrinsic Delaunay Triangulation took " << idtTime.time() << endl;
	int negativeCotEdgesCount = 0, internalEdgesCount = 0;
	double minCotValue = 1.0, negativeCotEdgesPercent = 0.0;
	bool areAllCotPositive = true;

	mCgalMesh.updateAllGlobalIndices();
	
	for (auto heIt1 = mCgalMesh.edges_begin(); heIt1 != mCgalMesh.edges_end(); heIt1++)
	{
		if (heIt1->is_border_edge()) // checks both current halfedge or its opposite, for border edge
			continue;
		internalEdgesCount++;
		if ((((heIt1->cot(true) + heIt1->opposite()->cot(true))) / 2) < minCotValue)
			minCotValue = ((heIt1->cot(true) + heIt1->opposite()->cot(true))) / 2;
		if (((heIt1->cot(true) + heIt1->opposite()->cot(true)) < -epsilon))
		{
			negativeCotEdgesCount++;
			areAllCotPositive = false;
		}
	}
	negativeCotEdgesPercent = double(negativeCotEdgesCount) / double(internalEdgesCount);
	cout << "are all cot positive " << areAllCotPositive << endl;
	cout << "minimum cot weight is " << minCotValue << endl;
	cout << "negative cot edges count is " << negativeCotEdgesCount << endl;
	cout << "internal edges count is " << internalEdgesCount << endl;
	cout << "negative cot edges rate is " << negativeCotEdgesPercent << endl;
	bool isSimpleGraph = true;
	for_each_vertex(v1, mCgalMesh)
	{
		Mesh::Halfedge_around_vertex_circulator h = v1->vertex_begin();
		const Mesh::Halfedge_around_vertex_circulator hEnd = h;
		std::vector<int> neighbors;
		CGAL_For_all(h, hEnd)
		{
			neighbors.push_back(h->opposite()->vertex()->index());
			if (h->vertex()->index() == h->opposite()->vertex()->index())
				isSimpleGraph = false;
		}
		const int originalNeighborsCount = neighbors.size();
		sort(neighbors.begin(), neighbors.end());
		neighbors.erase(unique(neighbors.begin(), neighbors.end()), neighbors.end());
		if (originalNeighborsCount != neighbors.size())
			isSimpleGraph = false;
	}

	int triangle_inequality_not_satisfied_count = 0;
	auto facetIter = mCgalMesh.facets_begin();
	while (facetIter != mCgalMesh.facets_end())
	{
		Halfedge_handle he[3];
		facetIter->getHalfedges(he);
		double l_mj = he[0]->targetMetric();
		double l_mi = he[1]->targetMetric();
		double l_ij = he[2]->targetMetric();
		bool is_triangle_inequality_satisfied = ((l_mj + l_mi > l_ij) && (l_mj + l_ij > l_mi) && (l_mi + l_ij > l_mj));
		if (!is_triangle_inequality_satisfied)
		{
			triangle_inequality_not_satisfied_count++;
			cout << "l_mj = " << l_mj << endl;
			cout << "l_mi = " << l_mi << endl;
			cout << "l_ij = " << l_ij << endl;
			cout << "problem! triangle inequality is not satisfied" << endl;
		}
		facetIter++;
	}
	cout << "is simple graph " << isSimpleGraph << endl;
	mDelaunayOutputs(0, 0) = nFlips;
	mDelaunayOutputs(1, 0) = nUnabledFlips;
	mDelaunayOutputs(2, 0) = areAllCotPositive;
	mDelaunayOutputs(3, 0) = minCotValue;
	mDelaunayOutputs(4, 0) = negativeCotEdgesCount;
	mDelaunayOutputs(5, 0) = internalEdgesCount;
	mDelaunayOutputs(6, 0) = negativeCotEdgesPercent;
	mDelaunayOutputs(7, 0) = triangle_inequality_not_satisfied_count;
	mDelaunayOutputs(8, 0) = idtTime.time();
	mDelaunayOutputs(9, 0) = isSimpleGraph;

	auto heIt = mCgalMesh.halfedges_begin();
	while (heIt != mCgalMesh.halfedges_end())
	{
		if (heIt->is_border())
			heIt->targetMetric() = heIt->opposite()->targetMetric();
		heIt++;
	}
}


void GIF::getScaffoldBordersMapAndSetMetaVertices(CGAL_Borders& borders, int numVertices)
{
	mIsMetaScaffold.clear();
	mIsMetaScaffold.resize(numVertices, false);
	mScaffoldConesAndMetaMap.clear();
	mScaffoldMetaVerticesInBorderByHalfEdge.clear();
	mScaffoldMetaVerticesInBorderByHalfEdge.resize(borders.numBorders());

	double first_border_vertex_x = borders.vertex(0, 0)->point().x();
	double first_border_vertex_y = borders.vertex(0, 0)->point().y();
	double second_border_vertex_x = borders.vertex(1, 0)->point().x();
	double second_border_vertex_y = borders.vertex(1, 0)->point().y();
	double first_border_radius_squared = first_border_vertex_x * first_border_vertex_x + first_border_vertex_y * first_border_vertex_y;
	double second_border_radius_squared = second_border_vertex_x * second_border_vertex_x + second_border_vertex_y * second_border_vertex_y;
	int index_of_inner_border = (first_border_radius_squared < second_border_radius_squared) ? 0 : 1;
	int index_of_outer_border = 1 - index_of_inner_border;


	std::vector<Halfedge_handle>& curr_outer_border = mScaffoldMetaVerticesInBorderByHalfEdge[index_of_outer_border];
	std::vector<Halfedge_handle> outer_border_path; //halfedges on boundary
	borders.getBorderHalfEdges(borders.vertex(index_of_outer_border, 0), outer_border_path);
	std::rotate(outer_border_path.begin(), outer_border_path.begin() + (outer_border_path[0]->vertex()->index() + 1), outer_border_path.end());
	for (int j = outer_border_path.size() - 1; j >= 0; j--)
	{
		Halfedge_handle HD = outer_border_path[j];
		curr_outer_border.push_back(HD);
		mScaffoldConesAndMetaMap.push_back(HD->vertex());//update borders map
		mIsMetaScaffold[HD->vertex()->index()] = true;
	}
	curr_outer_border.push_back(curr_outer_border[0]);

	std::vector<Halfedge_handle>& curr_inner_border = mScaffoldMetaVerticesInBorderByHalfEdge[index_of_inner_border];
	std::vector<Halfedge_handle> inner_border_path; //halfedges on boundary
	borders.getBorderHalfEdges(borders.vertex(index_of_inner_border, 0), inner_border_path);
	int rotateinnerBorder = inner_border_path[inner_border_path.size() - 1]->vertex()->index() - borders.numVerticesInBorder(index_of_outer_border);
	std::rotate(inner_border_path.rbegin(), inner_border_path.rbegin() + rotateinnerBorder, inner_border_path.rend());

	for (int j = inner_border_path.size() - 1; j >= 0; j--)
	{
		Halfedge_handle HD = inner_border_path[j];
		curr_inner_border.push_back(HD);
		mScaffoldConesAndMetaMap.push_back(HD->vertex());//update borders map
		mIsMetaScaffold[HD->vertex()->index()] = true;
	}
	curr_inner_border.push_back(curr_inner_border[0]);
}


void GIF::getVerticesThatMustBeMeta(CGAL_Borders& borders)
{
	// mark vertices on bridges
	GMMDenseColMatrix isVertexOnBridge(mCgalMesh.size_of_vertices(), 1); //indicate if a vertex in the mesh is on a bridge
	gmm::clear(isVertexOnBridge);

	int numBorders = borders.numBorders();
	std::vector<Vertex_handle> firstBridgeVertexInBorder(numBorders, NULL);


	for (int i = 0; i < borders.numBorders(); i++) {
		for (int j = 0; j < borders.numVerticesInBorder(i); j++) {
			Vertex_handle v = borders.vertex(i, j);

			//if a vertex is on cut and on border, is should be meta vertex
			if (v->onCut()) {
				mIsMeta[v->index()] = true;
				continue;
			}

			//find bridges (edges whose deletion disconnects the graph) - and set the 2 vertices on the bridge to be meta vertices
			Mesh::Halfedge_around_vertex_circulator h = v->vertex_begin();
			const Mesh::Halfedge_around_vertex_circulator hEnd = h;

			double sumWeights = 0.0;
			CGAL_For_all(h, hEnd)
			{
				if (!h->is_border_edge()) {
					Vertex_handle v1 = h->vertex();
					Vertex_handle v2 = h->opposite()->vertex();
					if (v1->is_border() && v2->is_border())
					{
						isVertexOnBridge(v1->index(), 0) = 1;
						isVertexOnBridge(v2->index(), 0) = 1;

						//add bridge vertices to meta
						mIsMeta[v1->index()] = true;
						mIsMeta[v2->index()] = true;

						if (firstBridgeVertexInBorder[i] == NULL)
							firstBridgeVertexInBorder[i] = v;
					}
				}
			}
		}
	}

	//select a vertex between every 2 bridge vertices.
	for (int i = 0; i < borders.numBorders(); i++) {

		if (firstBridgeVertexInBorder[i] == NULL)
			continue;

		Halfedge_around_vertex_circulator hec = firstBridgeVertexInBorder[i]->vertex_begin();
		Halfedge_around_vertex_circulator hec_end = hec;

		CGAL_For_all(hec, hec_end)
		{
			if (hec->is_border()) break;
		}
		Halfedge_handle he = hec;

		do
		{
			Halfedge_handle he_next = he->next();
			while (!mIsMeta[he_next->vertex()->index()])
				he_next = he_next->next();
			Halfedge_handle he2 = he_next;
			while (he->vertex() != he2->opposite()->vertex() && he->next()->vertex() != he2->opposite()->vertex()) {
				he = he->next();
				he2 = he2->prev();
			}
			mIsMeta[he2->opposite()->vertex()->index()] = true;

			he = he_next;
		} while ((he->vertex() != firstBridgeVertexInBorder[i]));
	}
}

void GIF::UpdateIndexOfHEinSystem()
{
	mIndexOfHEinSystem.clear();
	mIndexOfHEinSystem.resize(mCgalMesh.size_of_halfedges());
	mIndexOfVertexByIndexOfHEinSystem.clear();
	mIndexOfVertexByIndexOfHEinSystem.resize(mCgalMesh.size_of_vertices());
	auto heIt = mCgalMesh.halfedges_begin();
	int index1, index2;
	std::vector<bool> visit;
	visit.resize(mCgalMesh.size_of_halfedges());
	for (int i = 0; i < (int)visit.size(); ++i)
		visit[i] = false;
	int num = 0;
	bool flag = false;
	while (heIt != mCgalMesh.halfedges_end())
	{
		if (visit[heIt->index()])
		{
			heIt++;
			continue;
		}

		if (!heIt->vertex()->onCut())//if the vertex is not on the cut, then all the halfedges are mapped to the same variable
		{
			auto oneRing = heIt->vertex()->vertex_begin();
			auto start = oneRing;
			index1 = oneRing->index();
			mIndexOfHEinSystem[index1] = num;
			mIndexOfVertexByIndexOfHEinSystem[num] = heIt->vertex()->index();
			visit[index1] = true;
			oneRing++;
			while (oneRing != start)
			{
				index2 = oneRing->index();
				mIndexOfHEinSystem[index2] = num;
				visit[index2] = true;
				oneRing++;
			}
			num++;
		}
		else//if it is on the cut, split the vertex
		{
			auto oneRing = heIt->vertex()->vertex_begin();
			while ((!oneRing->isCut()) && (!oneRing->opposite()->is_border()))
				oneRing++;
			auto start = oneRing;
			index1 = start->index();
			mIndexOfHEinSystem[index1] = num;
			visit[index1] = true;
			oneRing++;

			while (1)	//its o.k, the break condition have to be true
			{
				bool endFlag = false;
				while (!oneRing->isCut())
				{
					flag = true;
					index2 = oneRing->index();
					mIndexOfHEinSystem[index2] = num;
					visit[index2] = true;
					if (oneRing->is_border())
					{
						if (oneRing == start)
							endFlag = true;
						oneRing++;
						break;
					}
					oneRing++;
				}
				num++;
				if ((oneRing == start) || (endFlag))
					break;
				index1 = oneRing->index();
				mIndexOfHEinSystem[index1] = num;
				visit[index1] = true;
				oneRing++;
			}
		}
		heIt++;
	}
	mSizeOfSystemVar = num;
	GMMDenseColMatrix numOfsystemVars(1, 1);
	numOfsystemVars(0, 0) = mSizeOfSystemVar;
	MatlabGMMDataExchange::SetEngineDenseMatrix("numOfsystemVars", numOfsystemVars);
	MatlabInterface::GetEngine().EvalToCout("GIF.sysVars = numOfsystemVars(1, 1);");
}

void GIF::UpdateIndexOfHEinScaffoldSystem(Mesh& mesh1)
{

	mIndexOfHEinScaffoldSystem.clear();
	mIndexOfHEinScaffoldSystem.resize(mesh1.size_of_halfedges());
	auto heIt = mesh1.halfedges_begin();
	int index1, index2;
	std::vector<bool> visit;
	visit.resize(mesh1.size_of_halfedges());
	for (int i = 0; i < (int)visit.size(); ++i)
		visit[i] = false;
	int num = 0;
	bool flag = false;
	while (heIt != mesh1.halfedges_end())
	{
		if (visit[heIt->index()])
		{
			heIt++;
			continue;
		}

		if (!heIt->vertex()->onCut())//if the vertex is not on the cut, then all the halfedges are mapped to the same variable
		{
			auto oneRing = heIt->vertex()->vertex_begin();
			auto start = oneRing;
			index1 = oneRing->index();
			mIndexOfHEinScaffoldSystem[index1] = num;
			visit[index1] = true;
			oneRing++;
			while (oneRing != start)
			{
				index2 = oneRing->index();
				mIndexOfHEinScaffoldSystem[index2] = num;
				visit[index2] = true;
				oneRing++;
			}
			num++;
		}
		else//if it is on the cut, split the vertex
		{
			auto oneRing = heIt->vertex()->vertex_begin();
			while ((!oneRing->isCut()) && (!oneRing->opposite()->is_border()))
				oneRing++;
			auto start = oneRing;
			index1 = start->index();
			mIndexOfHEinScaffoldSystem[index1] = num;
			visit[index1] = true;
			oneRing++;

			while (1)	//its o.k, the break condition have to be true
			{
				bool endFlag = false;
				while (!oneRing->isCut())
				{
					flag = true;
					index2 = oneRing->index();
					mIndexOfHEinScaffoldSystem[index2] = num;
					visit[index2] = true;
					if (oneRing->is_border())
					{
						if (oneRing == start)
							endFlag = true;
						oneRing++;
						break;
					}
					oneRing++;
				}
				num++;
				if ((oneRing == start) || (endFlag))
					break;
				index1 = oneRing->index();
				mIndexOfHEinScaffoldSystem[index1] = num;
				visit[index1] = true;
				oneRing++;
			}
		}
		heIt++;
	}
	mSizeOfScaffoldSystemVar = num;
}

bool GIF::TransferBorderFacesAndVerticesToMatlab()
{
	//******* Tranfer duplicated Vertices by halfedges************
	mParameters_matrix.clear();
	mParameters_matrix.resize(20, 1);
	mParameters_matrix(0, 0) = mCgalMesh.size_of_vertices();
	mParameters_matrix(1, 0) = mCgalMesh.size_of_facets();
	mParameters_matrix(2, 0) = mCgalMesh.size_of_border_edges();
	mParameters_matrix(3, 0) = mNumDOF;
	mParameters_matrix(5, 0) = mSegSize;
	GMMDenseColMatrix verticesByHalfedges(mSizeOfSystemVar, 3);
	auto heIt = mCgalMesh.halfedges_begin();
	while (heIt != mCgalMesh.halfedges_end())
	{
		int index = mIndexOfHEinSystem[heIt->index()];
		auto p = heIt->vertex()->point();
		verticesByHalfedges(index, 0) = p[0];
		verticesByHalfedges(index, 1) = p[1];
		verticesByHalfedges(index, 2) = p[2];
		heIt++;
	}

	GMMDenseColMatrix facesIndices(mCgalMesh.size_of_facets(), 3);
	std::vector<int> boundaryFacesIndicesForCheck;
	std::vector<int> internalFacesIndicesForCheck;
	auto facetIter = mCgalMesh.facets_begin();
	int i = 0, boundaryFacesIndex = 0, internalFacesIndex = 0;
	while (facetIter != mCgalMesh.facets_end())
	{
		Halfedge_handle he[3];
		facetIter->getHalfedges(he);

		Vertex_handle vertices1[3];
		facetIter->getVertices(vertices1);

		facesIndices(i, 0) = mIndexOfHEinSystem[he[0]->index()] + 1;
		facesIndices(i, 1) = mIndexOfHEinSystem[he[1]->index()] + 1;
		facesIndices(i, 2) = mIndexOfHEinSystem[he[2]->index()] + 1;
		if (vertices1[0]->is_border() || vertices1[1]->is_border() || vertices1[2]->is_border())
			boundaryFacesIndicesForCheck.push_back(i + 1); // boundary faces
		else
			internalFacesIndicesForCheck.push_back(i + 1); // internal faces

		facetIter++;
		i++;
	}
	MatlabGMMDataExchange::SetEngineDenseMatrix("GIF.F1", facesIndices);

	MatlabGMMDataExchange::SetEngineDenseMatrix("GIF.V", verticesByHalfedges);

	//********* Transfer border faces corresponding to Halfedges and vector field****************
	std::unordered_set<Facet_handle> facesNearCones;//use hash to prevent duplications
	std::unordered_set<Facet_handle> facesNearCones1;

	for (int i = 0; i < mConesAndMetaMap.size(); i++) {
		Vertex_handle v = mConesAndMetaMap[i];
		int ind = v->index();
		Mesh::Halfedge_around_vertex_circulator h = v->vertex_begin();
		const Mesh::Halfedge_around_vertex_circulator hEnd = h;
		CGAL_For_all(h, hEnd)
		{
			if (!h->is_border()) {
				facesNearCones.insert(h->face());
				facesNearCones1.insert(h->face());
			}
		}
	}

	int numOuterFacesAlready = facesNearCones.size();
	if (mNumInternalFaces > 0)
	{
		GMMDenseColMatrix numFaces(1, 1);
		numFaces(0, 0) = mCgalMesh.size_of_facets();
		MatlabGMMDataExchange::SetEngineDenseMatrix("numFaces", numFaces);
		MatlabInterface::GetEngine().EvalToCout("rng(1);");
		MatlabInterface::GetEngine().EvalToCout("randomTrianglesIndices = (randperm(numFaces(1, 1)) - 1).';");
		GMMDenseColMatrix randomTriangles;
		MatlabGMMDataExchange::GetEngineDenseMatrix("randomTrianglesIndices", randomTriangles);
		int i = 0;
		while (true)
		{
			facesNearCones.insert(mCgalMesh.face(randomTriangles(i, 0)));
			i++;
			if ((facesNearCones.size() - numOuterFacesAlready) == mNumInternalFaces || facesNearCones.size() == mCgalMesh.size_of_facets())
				break;

		}
	}

	// Sampling based on area
	//vector<double> faces_areas;
	//faces_areas.reserve(mCgalMesh.size_of_facets() + 1);
	//for_each_const_facet(fi, mCgalMesh)
	//	faces_areas.push_back(fi->area());
	//vector<double> cdf(faces_areas.size());
	//std::partial_sum(faces_areas.begin(), faces_areas.end(), cdf.begin());
	//double total = cdf.back();
	//for (auto& v : cdf) v /= total;  // normalize to [0,1]
	//std::random_device rd;
	//std::mt19937 gen(rd());
	//std::uniform_real_distribution<> dis(0.0, 1.0);

	//while (true)
	//{
	//	double r = dis(gen);
	//	auto it = std::lower_bound(cdf.begin(), cdf.end(), r);
	//	int tri_idx = std::distance(cdf.begin(), it);
	//	facesNearCones.insert(mCgalMesh.face(tri_idx));
	//	cout << tri_idx << endl;
	//	if ((facesNearCones.size() - numOuterFacesAlready) == mNumInternalFaces || facesNearCones.size() == mCgalMesh.size_of_facets())
	//		break;
	//}

	mNearBoundaryVertices.clear();
	std::unordered_set<Facet_handle> facesNearCones2;
	for (int i = 0; i < mBoundaryVerticesMap.size(); i++) {
		Vertex_handle v = mBoundaryVerticesMap[i];
		int ind = v->index();
		Mesh::Halfedge_around_vertex_circulator h = v->vertex_begin();
		const Mesh::Halfedge_around_vertex_circulator hEnd = h;
		CGAL_For_all(h, hEnd)
		{
			if (!h->is_border()) {
				facesNearCones2.insert(h->face());
			}
		}
	}
	unordered_set<Facet_handle>::const_iterator itr;
	for (itr = facesNearCones2.begin(); itr != facesNearCones2.end(); ++itr) {
		Facet_handle f = (*itr);
		Halfedge_handle he[3];
		f->getHalfedges(he);

		if (!mBoundaryVertices.count(mIndexOfHEinSystem[he[0]->index()]))
			mNearBoundaryVertices.insert(mIndexOfHEinSystem[he[0]->index()]);
		if (!mBoundaryVertices.count(mIndexOfHEinSystem[he[1]->index()]))
			mNearBoundaryVertices.insert(mIndexOfHEinSystem[he[1]->index()]);
		if (!mBoundaryVertices.count(mIndexOfHEinSystem[he[2]->index()]))
			mNearBoundaryVertices.insert(mIndexOfHEinSystem[he[2]->index()]);

	}
	unordered_set<int>::const_iterator itr1;
	for (itr1 = mNearBoundaryVertices.begin(); itr1 != mNearBoundaryVertices.end(); itr1++)
	{
		mNearBoundaryVerticesVec.push_back((*itr1));
	}

	int numNearConesFaces = facesNearCones.size();
	GMMDenseColMatrix facesByHalfedges(numNearConesFaces, 3);
	GMMDenseColMatrix internalFacesIndicator(numNearConesFaces, 1);

	GMMDenseColMatrix nearConesFaces(numNearConesFaces, 1);

	mConesAndNearConesMapOfRowsInKKT.clear();
	mNearConesVertices.clear();
	i = 0;
	for (itr = facesNearCones.begin(); itr != facesNearCones.end(); ++itr) {
		Facet_handle f = (*itr);
		Halfedge_handle he[3];
		f->getHalfedges(he);
		facesByHalfedges(i, 0) = mIndexOfHEinSystem[he[0]->index()] + 1;//+1 for matlab
		facesByHalfedges(i, 1) = mIndexOfHEinSystem[he[1]->index()] + 1;
		facesByHalfedges(i, 2) = mIndexOfHEinSystem[he[2]->index()] + 1;
		internalFacesIndicator(i, 0) = facesNearCones1.count(f) ? 0 : 1;

		mConesAndNearConesMapOfRowsInKKT.insert(mIndexOfHEinSystem[he[0]->index()]);
		mConesAndNearConesMapOfRowsInKKT.insert(mIndexOfHEinSystem[he[1]->index()]);
		mConesAndNearConesMapOfRowsInKKT.insert(mIndexOfHEinSystem[he[2]->index()]);

		if (!mConesVertices.count(mIndexOfHEinSystem[he[0]->index()]))
			mNearConesVertices.insert(mIndexOfHEinSystem[he[0]->index()]);
		if (!mConesVertices.count(mIndexOfHEinSystem[he[1]->index()]))
			mNearConesVertices.insert(mIndexOfHEinSystem[he[1]->index()]);
		if (!mConesVertices.count(mIndexOfHEinSystem[he[2]->index()]))
			mNearConesVertices.insert(mIndexOfHEinSystem[he[2]->index()]);
		++i;
	}
	mNearConesVerticesVec.clear();
	for (itr1 = mNearConesVertices.begin(); itr1 != mNearConesVertices.end(); itr1++)
	{
		mNearConesVerticesVec.push_back((*itr1));
	}
	mNonBoundaryVerticesNearConesVerticesVec.clear();
	mNonMetaBoundaryVerticesNearConesVerticesVec.clear();
	for (itr1 = mNearConesVertices.begin(); itr1 != mNearConesVertices.end(); itr1++)
	{
		if (!mBoundaryVertices.count((*itr1)))
			mNonBoundaryVerticesNearConesVerticesVec.push_back((*itr1));
		else
			mNonMetaBoundaryVerticesNearConesVerticesVec.push_back((*itr1));
	}
	mMetaIndicesVec.clear();
	mIsNearMeta.clear();
	mIsNearMeta.resize(mCgalMesh.size_of_vertices(), false);
	mMetaIndicesVec.resize(mCgalMesh.size_of_vertices(), -1);
	int num = 0;
	for (int i = 0; i < mConesAndMetaMap.size(); i++)
	{
		mMetaIndicesVec[mConesAndMetaMap[i]->index()] = num;
		num++;
	}
	for (int i = 0; i < mNearConesVerticesVec.size(); i++)
	{
		mIsNearMeta[mNearConesVerticesVec[i]] = true;
	}
	mInternalVerticesIndices.clear();
	mInternalVerticesIndices.resize(mCgalMesh.size_of_vertices());
	num = 0;
	for_each_vertex(vi, mCgalMesh)
	{
		if (!vi->is_border())
		{
			mInternalVerticesIndices[vi->index()] = num;
			num++;
		}
	}
	num = 0;
	mBoundaryVerticesIndices.clear();
	mBoundaryVerticesIndices.resize(mCgalMesh.size_of_vertices());
	for (int i = 0; i < mBoundaryVerticesMap.size(); i++)
	{
		mBoundaryVerticesIndices[mBoundaryVerticesMap[i]->index()] = num;
		num++;
	}

	MatlabGMMDataExchange::SetEngineDenseMatrix("GIF.F", facesByHalfedges);
	MatlabGMMDataExchange::SetEngineDenseMatrix("GIF.internalFacesIndicator", internalFacesIndicator);
	GMMDenseColMatrix borderEdges(mBoundaryVerticesMap.size(), 3);
	for (int i = 0; i < mBoundaryVerticesMap.size(); i++) {
		Vertex_handle v = mBoundaryVerticesMap[i];
		borderEdges(i, 0) = v->point().x();
		borderEdges(i, 1) = v->point().y();
		borderEdges(i, 2) = v->point().z();
	}
	MatlabGMMDataExchange::SetEngineDenseMatrix("GIF.borderEdges", borderEdges);
	mParameters_matrix(6, 0) = facesNearCones1.size();
	mParameters_matrix(7, 0) = facesNearCones.size() - facesNearCones1.size();

	return true;
}

bool GIF::TransferScaffoldBorderFacesAndVerticesToMatlab(Mesh& mesh1)
{

	//******* Tranfer duplicated Vertices by halfedges************
	GMMDenseColMatrix verticesByHalfedges(mesh1.size_of_vertices(), 3);

	auto heIt = mesh1.halfedges_begin();
	while (heIt != mesh1.halfedges_end())
	{
		int index = mIndexOfHEinScaffoldSystem[heIt->index()];
		auto p = heIt->vertex()->point();
		verticesByHalfedges(index, 0) = p[0];
		verticesByHalfedges(index, 1) = p[1];
		verticesByHalfedges(index, 2) = p[2];
		heIt++;
	}

	MatlabGMMDataExchange::SetEngineDenseMatrix("GIF.SV ", verticesByHalfedges);

	//********* Transfer border faces corresponding to Halfedges and vector field****************
	std::unordered_set<Facet_handle> facesNearCones;//use hash to prevent duplications

	for (int i = 0; i < mScaffoldConesAndMetaMap.size(); i++) {
		Vertex_handle v = mScaffoldConesAndMetaMap[i];
		Mesh::Halfedge_around_vertex_circulator h = v->vertex_begin();
		const Mesh::Halfedge_around_vertex_circulator hEnd = h;
		CGAL_For_all(h, hEnd)
		{
			if (!h->is_border()) {
				facesNearCones.insert(h->face());
			}
		}
	}
	int numNearConesFaces = facesNearCones.size();
	GMMDenseColMatrix facesByHalfedges(numNearConesFaces, 3);


	unordered_set<Facet_handle>::const_iterator itr;
	int i = 0;
	for (itr = facesNearCones.begin(); itr != facesNearCones.end(); ++itr) {
		Facet_handle f = (*itr);
		Halfedge_handle he[3];
		f->getHalfedges(he);
		facesByHalfedges(i, 0) = mIndexOfHEinScaffoldSystem[he[0]->index()] + 1;//+1 for matlab
		facesByHalfedges(i, 1) = mIndexOfHEinScaffoldSystem[he[1]->index()] + 1;
		facesByHalfedges(i, 2) = mIndexOfHEinScaffoldSystem[he[2]->index()] + 1;
		++i;
	}


	MatlabGMMDataExchange::SetEngineDenseMatrix("GIF.SF", facesByHalfedges);

	return true;
}


bool GIF::runAlgorithm(CGAL_Borders& borders)
{
	cout << "Starting algorithm" << endl;
	//*** run algorithm to calculate UV's ***
	GMMDenseColMatrix RHS;
	int conesConstraintsStartRow;

	cout << "mSegSize = " << mSegSize << endl;
	cout << "mNumInternalFaces = " << mNumInternalFaces << endl;

	CGAL::Timer t1;
	t1.start();

	bool res = true;
	constructKKTmatrix(borders, conesConstraintsStartRow);
	constructHarmonicBasisAndSendToMATLAB(conesConstraintsStartRow);
	bool tutteSuccess = getTutteInitialValue(borders);
	GMMDenseColMatrix uvsOfBoundary(mNumDOF, 2);
	double outerRadiusOfScaffold1 = 0.0;
	MatlabInterface::GetEngine().EvalToCout("computeOptimizationRescaling");
	MatlabGMMDataExchange::GetEngineDenseMatrix("x10", uvsOfBoundary);

	mInitialUVsOfCones.clear();
	mInitialUVsOfCones.resize(mNumDOF);
	for (int i = 0; i < mNumDOF; i++)
	{
		mInitialUVsOfCones[i] = Complex(uvsOfBoundary(i, 0), uvsOfBoundary(i, 1));

	}
	Mesh initialScaffold;
	MeshAlgorithms::buildScaffoldWithFreeOuterBoundary(mInitialUVsOfCones, false, initialScaffold, (mNumScaffoldDOF - mNumDOF), 10);

	CGAL_Borders borders1(initialScaffold);



	Eigen::SparseMatrix<double, Eigen::RowMajor> scaffoldKKtEigen1, scaffoldKKtEigen2;

	getScaffoldBordersMapAndSetMetaVertices(borders1, initialScaffold.size_of_vertices());
	UpdateIndexOfHEinScaffoldSystem(initialScaffold);
	TransferScaffoldBorderFacesAndVerticesToMatlab(initialScaffold);


	res = constructScaffoldHarmonicBasisAndSendToMATLAB();
	if (!res) {
		cout << "Failed to construct Harmonic Basis for the scaffold" << endl;
		return false;
	}
	int numVerticesScaffold = initialScaffold.size_of_vertices();
	GMMDenseColMatrix scaffoldBoundaryVertices(2 * numVerticesScaffold, 1);
	for (int i = 0; i < numVerticesScaffold; i++)
	{
		scaffoldBoundaryVertices(i, 0) = mScaffoldConesAndMetaMap[i]->point().x();
		scaffoldBoundaryVertices(i + numVerticesScaffold, 0) = mScaffoldConesAndMetaMap[i]->point().y();
	}
	MatlabGMMDataExchange::SetEngineDenseMatrix("scaffoldBoundaryVertices", scaffoldBoundaryVertices);
	MatlabInterface::GetEngine().EvalToCout("[ GIF.ScaffoldUVonCones, GIF.FixedIndices, GIF.FixedValues ] = fixFirstCone( scaffoldBoundaryVertices );");

	t1.stop();
	cout << "Subspace construction took " << t1.time() << endl;

	bool NewtonSucsess;
	CGAL::Timer t2;
	t2.start();
	NewtonSucsess = runNewtonWithRemeshedScaffold(conesConstraintsStartRow, RHS);
	t2.stop();

	GMMDenseColMatrix UVs(RHS.nrows(), RHS.ncols());
	mUVs.clear();
	mUVs.resize(RHS.nrows(), RHS.ncols());
	bool res1 = mPardisoSolver.solve(RHS, mUVs);
	cout << "Newton and back substitution took " << t2.time() << endl;

	return true;
}



void GIF::constructKKTmatrix(CGAL_Borders& borders, int& conesConstraintsStartRow)
{
	std::vector<Eigen::Triplet<double>> tripletListValues;

	tripletListValues.reserve(6 * mCgalMesh.size_of_vertices() + mNumDOF);

	FillLaplacianInKKT(tripletListValues);

	conesConstraintsStartRow = mSizeOfSystemVar;
	mHarmonicBasisInAllV.resize(mSizeOfSystemVar, mNumDOF);
	FillMetaVerticesConstraintsInKKT(borders, tripletListValues, conesConstraintsStartRow);

	mSizeOfMatrix = conesConstraintsStartRow + mNumDOF;

	FillConesConstraintsInKKT(tripletListValues, conesConstraintsStartRow);
	SetElementsForPARDISO(tripletListValues, conesConstraintsStartRow);

	mKKtEigen.resize(mSizeOfMatrix, mSizeOfMatrix);
	mKKtEigen.setFromTriplets(tripletListValues.begin(), tripletListValues.end());
	mKKtEigen.makeCompressed();
}


void GIF::FillLaplacianInKKT(std::vector<Eigen::Triplet<double>>& tripletListValues)
{
	//fill the matrix by going on the triangles one by one
	for_each_facet(f, mCgalMesh)
	{
		//cout <<"f index: " << f->index() << endl;
		Halfedge_handle h1 = f->halfedge();
		Halfedge_handle h2 = h1->next();
		Halfedge_handle h3 = h2->next();

		double term1 = -0.5 * h1->cot(true);
		double term2 = -0.5 * h2->cot(true);
		double term3 = -0.5 * h3->cot(true);


		//v1
		updateTermInLaplacianByHalfEdge(h3, term3, tripletListValues); //v1->v2 and v2->v1

		//v2
		updateTermInLaplacianByHalfEdge(h1, term1, tripletListValues); //v2->v3 and v3->v2

		//v3
		updateTermInLaplacianByHalfEdge(h2, term2, tripletListValues); //v3->v1 and v1->v3
	}
}


void GIF::updateTermInLaplacianByHalfEdge(Halfedge_handle h, const double& term, std::vector<Eigen::Triplet<double>>& tripletListValues)
{
	const int i = mIndexOfHEinSystem[h->prev()->index()];
	const int j = mIndexOfHEinSystem[h->index()];

	if (term != 0) {
		updateTermInLaplacianByIndices(i, j, term, tripletListValues);
	}
}


void GIF::updateTermInLaplacianByIndices(int i, int j, const double& term, std::vector<Eigen::Triplet<double>>& tripletListValues)
{
	tripletListValues.push_back(Eigen::Triplet<double>(i, i, -term));
	tripletListValues.push_back(Eigen::Triplet<double>(j, j, -term));
	// only store upper triangular part
	if (j >= i) { //v1->v2
		tripletListValues.push_back(Eigen::Triplet<double>(i, j, term));
	}
	else {//v2->v1
		tripletListValues.push_back(Eigen::Triplet<double>(j, i, term));
	}
}


void GIF::FillMetaVerticesConstraintsInKKT(CGAL_Borders& borders, std::vector<Eigen::Triplet<double>>& tripletListValues, int& rowInKKT)
{
	//calculate weights to meta vertices
	std::vector<Halfedge_handle>& currBorder = mMetaVerticesInBorderByHalfEdge[0];
	for (int i = 0; i < currBorder.size() - 1; i++) {
		Halfedge_handle metaVertexHD = currBorder[i];
		Halfedge_handle nextMetaVertexHD = currBorder[i + 1];

		//find meta Edge Length
		Halfedge_handle h = metaVertexHD;
		double metaEdgeLength = 0.0;
		while (h != nextMetaVertexHD) {
			metaEdgeLength += h->length();
			h = h->prev();
		}

		//update meta vertices and t
		h = metaVertexHD->prev();
		double lengthFromFirstMeta = 0.0;
		while (h != nextMetaVertexHD) {
			const int firstMetaVertex = mIndexOfHEinSystem[metaVertexHD->index()];
			const int secondMetaVertex = mIndexOfHEinSystem[nextMetaVertexHD->next()->opposite()->index()];//if the vertex is on cut then this gives the correct halfedge, and if not that it is the same.
			const int curVertex = mIndexOfHEinSystem[h->index()];
			lengthFromFirstMeta += h->next()->length();
			const double t = (1 - lengthFromFirstMeta / metaEdgeLength);

			// if we use Eigen, then we only fill upper triangular part, then only A^t is relevant
			tripletListValues.push_back(Eigen::Triplet<double>(firstMetaVertex, rowInKKT, t));
			tripletListValues.push_back(Eigen::Triplet<double>(secondMetaVertex, rowInKKT, 1 - t));
			tripletListValues.push_back(Eigen::Triplet<double>(curVertex, rowInKKT, -1));
			if (mIsNearMeta[curVertex])
			{
				mHarmonicBasisInAllV(curVertex, mMetaIndicesVec[mIndexOfVertexByIndexOfHEinSystem[firstMetaVertex]]) = t;
				mHarmonicBasisInAllV(curVertex, mMetaIndicesVec[mIndexOfVertexByIndexOfHEinSystem[secondMetaVertex]]) = 1 - t;
			}


			++rowInKKT;
			h = h->prev();
		}
	}
}


void GIF::FillConesConstraintsInKKT(std::vector<Eigen::Triplet<double>>& tripletListValues, int conesConstraintsStartRow)
{
	for (int i = 0; i < mNumDOF; i++) {
		int rowInKKT = conesConstraintsStartRow + i;
		int indexOfConeInSystem = mIndexOfHEinSystem[mConesAndMetaMap[i]->halfedge()->index()];
		tripletListValues.push_back(Eigen::Triplet<double>(indexOfConeInSystem, rowInKKT, 1));
	}
}


void GIF::SetElementsForPARDISO(std::vector<Eigen::Triplet<double>>& tripletListValues, int conesConstraintsStartRow)
{
	//set diagonal elements
	for (int i = 0; i < mSizeOfMatrix; i++)
		tripletListValues.push_back(Eigen::Triplet<double>(i, i, 0));
#ifndef BCP_PARDISO_COMPAT
	// Panua's selected inversion needs these entries in the sparsity pattern.
	for (int row : mNonBoundaryVerticesNearConesVerticesVec)
		for (int j = 0; j < mNumDOF; ++j)
			tripletListValues.emplace_back(row, j + conesConstraintsStartRow, 0.0);
#endif
}


bool GIF::constructHarmonicBasisAndSendToMATLAB(int conesConstraintsStartRow)
{
	//**********calculate harmonic basis in PARDISO************
	bool res;
	res = calculateHarmonicBasisInPARDISO(conesConstraintsStartRow);
	if (!res) return false;
	MatlabInterface::GetEngine().EvalToCout("createJmatrixForOriginalMeshWithScaffold");
	return true;
}

bool GIF::calculateHarmonicBasisInPARDISO(int conesConstraintsStartRow)
{
	//**********calculate harmonic basis in PARDISO************

	int	mtype = -2;//real and symmetric indefinite
	mPardisoSolver = (PardisoLinearSolver(mtype));
	bool res = mPardisoSolver.init(true);
	if (!res) return false;
	res = mPardisoSolver.createPardisoFormatMatrix(mKKtEigen);
	if (!res) return false;
	CGAL::Timer pardiso_timer;
	pardiso_timer.start();
	res = mPardisoSolver.preprocess();
	pardiso_timer.stop();
	std::cout << "Symbolic and Numerical factorization: " << pardiso_timer.time() << "\n";
	pardiso_timer.reset();
	pardiso_timer.start();
	if (!res) return false;
#ifndef BCP_PARDISO_COMPAT
	// Preserve the original selected-inversion path for Panua/legacy builds.
	GMMCompressed1RowMatrix inverse(mKKtEigen.rows(), mKKtEigen.cols());
	if (!mPardisoSolver.selectiveInverse() || !mPardisoSolver.getMatrixInGMMformat(inverse)) return false;
	for (int row : mNonBoundaryVerticesNearConesVerticesVec)
		for (int j = 0; j < mNumDOF; ++j)
			mHarmonicBasisInAllV(row, j) = inverse(row, j + conesConstraintsStartRow);
	pardiso_timer.stop();
	std::cout << "Selected inverse: " << pardiso_timer.time() << "\n";
#else
	// We only need inverse columns associated with cone constraints. Solving
	// these unit right-hand sides is equivalent to PARDISO's selected inverse,
	// without forming a dense inverse or solving for every matrix column.
	// Batch small groups for MKL's multi-RHS solve without allocating a dense
	// inverse. Bound each dense buffer to ~16 MiB (or at least one column).
	const size_t columnsInBudget = (16u * 1024u * 1024u) /
		(std::max(size_t(mSizeOfMatrix), size_t(1)) * sizeof(double));
	const int batchSize = std::max(1, std::min(mNumDOF, int(std::min(size_t(16), columnsInBudget))));
	GMMDenseColMatrix rhs, solution;
	for (int first = 0; first < mNumDOF; first += batchSize) {
		const int count = std::min(batchSize, mNumDOF - first);
		rhs.resize(mSizeOfMatrix, count);
		solution.resize(mSizeOfMatrix, count);
		std::fill(rhs.begin(), rhs.end(), 0.0);
		for (int j = 0; j < count; ++j)
			rhs(conesConstraintsStartRow + first + j, j) = 1.0;
		if (!mPardisoSolver.solve(rhs, solution)) return false;
		for (int row : mNonBoundaryVerticesNearConesVerticesVec)
			for (int j = 0; j < count; ++j)
				mHarmonicBasisInAllV(row, first + j) = solution(row, j);
	}
	pardiso_timer.stop();
	std::cout << "Harmonic basis inverse-column solves: " << pardiso_timer.time() << "\n";
#endif
	for (int i = 0; i < mNumDOF; i++) {
		int rowInKKT = i;
		int indexOfConeInSystem = mIndexOfHEinSystem[mConesAndMetaMap[i]->halfedge()->index()];
		mHarmonicBasisInAllV(indexOfConeInSystem, rowInKKT) = 1.0;
	}
	MatlabGMMDataExchange::SetEngineSparseMatrix("HarmonicBasis", mHarmonicBasisInAllV);
	return true;
}


bool GIF::constructScaffoldHarmonicBasisAndSendToMATLAB()
{
	//**********calculate harmonic basis in PARDISO************
	bool res = calculateScaffoldHarmonicBasis();
	if (!res) return false;

	//******* Calculate frames and initialize matrices in MATLAB **********
	MatlabInterface::GetEngine().EvalToCout("createJmatrixScaffold");
	return true;
}

bool GIF::calculateScaffoldHarmonicBasis()
{

	GMMSparseComplexRowMatrix harmonicBasisInAllScaffoldV;
	harmonicBasisInAllScaffoldV.resize(mSizeOfScaffoldSystemVar, mNumScaffoldDOF);
	for (int i = 0; i < mNumScaffoldDOF; i++) {
		int rowInKKT = i;
		int indexOfConeInSystem = mIndexOfHEinScaffoldSystem[mScaffoldConesAndMetaMap[i]->halfedge()->index()];
		harmonicBasisInAllScaffoldV(indexOfConeInSystem, rowInKKT) = 1.0;
	}
	MatlabGMMDataExchange::SetEngineSparseMatrix("ScaffoldHarmonicBasis", harmonicBasisInAllScaffoldV);
	return true;
}


bool GIF::getTutteInitialValue(CGAL_Borders& borders)
{
	GMMDenseColMatrix initialValue(mNumDOF * 2, 1);
	mInitialUVsOfCones.clear();
	mInitialUVsOfCones.resize(mNumDOF);

	int mainBorderIndex = 0;
	Vertex_handle mainBorderVertex = borders.vertex(0, 0);
	int mainBorderVertexIndex = mainBorderVertex->index();

	std::vector<Halfedge_handle> pathMainBorder;
	borders.getBorderHalfEdges(mainBorderVertex, pathMainBorder);
	int min_vertex_index = 0;
	for (int j = 0; j < pathMainBorder.size(); j++)
		if (pathMainBorder[j]->vertex()->index() < pathMainBorder[min_vertex_index]->vertex()->index())
			min_vertex_index = j;
	std::rotate(pathMainBorder.begin(), pathMainBorder.begin() + (min_vertex_index + 1), pathMainBorder.end());
	int numVerticesMainBorder = pathMainBorder.size();

	double totalBorderLength = 0.0;
	for (int i = 0; i < numVerticesMainBorder; i++)
	{
		double edgeLength = pathMainBorder[i]->length();
		totalBorderLength += edgeLength;
	}
	assert(totalBorderLength > 0.0);
	mRadiusOfInitialTutte = 1.0;
	double totalAngle = 0.0;
	double angle = 0.0;
	for (int i = numVerticesMainBorder - 1; i >= 0; --i)
	{
		totalAngle += angle;
		if (mIsMeta[pathMainBorder[i]->vertex()->index()]) {
			double x = cos(totalAngle);
			double y = sin(totalAngle);
			int indexInUV = mIndicesOfMetaVerticesInUVbyGeneralIndex[pathMainBorder[i]->vertex()->index()];
			initialValue(indexInUV, 0) = x;
			initialValue(indexInUV + mNumDOF, 0) = y;
			mInitialUVsOfCones[indexInUV] = Complex(x, y);
		}
		double edgeLength = pathMainBorder[i]->length();
		angle = 2.0 * M_PI * edgeLength / totalBorderLength;
	}


	MatlabGMMDataExchange::SetEngineDenseMatrix("GIF.UVonCones", initialValue);
	MatlabInterface::GetEngine().EvalToCout("[ GIF.UVonCones, GIF.FixedIndices, GIF.FixedValues ] = fixFirstCone( GIF.UVonCones );");

	return true;
}

bool GIF::runNewtonWithRemeshedScaffold(int conesConstraintsStartRow, GMMDenseColMatrix& RHS)
{
	CGAL::Timer t1;
	t1.start();
	MatlabInterface::GetEngine().EvalToCout("newtonWithScaffoldPreps;");
	double NewtonSucsess = true;
	GMMDenseColMatrix UVonCones(mNumScaffoldDOF, 2);
	MatlabGMMDataExchange::GetEngineDenseMatrix("GIF.UVonCones2", UVonCones);
	GMMDenseColMatrix UVonCones1(mNumScaffoldDOF, 2);
	MatlabGMMDataExchange::GetEngineDenseMatrix("GIF.UVonCones1", UVonCones1);
	int scaffoldConesConstraintsStartRow2;
	int maxRemeshingTimes = 100;
	double curE = 0.0, prevE = 0.0;
	int convex_iteration = 0;
	cout << t1.time() << endl;
	for (int iter1 = 0; iter1 < maxRemeshingTimes; iter1++)
	{
		cout << "Iter number  " << iter1 << " took " << t1.time() << endl;
		MatlabInterface::GetEngine().EvalToCout("iterationOfNewtonWithScaffold;");
		cout << "Iter number  " << iter1 << " took " << t1.time() << endl;
		GMMDenseColMatrix UVonCones2(mNumScaffoldDOF, 2);
		MatlabGMMDataExchange::GetEngineDenseMatrix("GIF.UVonCones1", UVonCones2);
		std::vector<std::complex<double>> internalBorderOfScaffold;
		for (int i = 0; i < mNumDOF; i++)
		{
			internalBorderOfScaffold.push_back(Complex(UVonCones2(mNumScaffoldDOF - i - 1, 0), UVonCones2(mNumScaffoldDOF - i - 1, 1)));
		}
		bool ret = CompilationAccelerator::isSimplePolygon(internalBorderOfScaffold, false);
		if (ret)
		{
			MatlabGMMDataExchange::GetEngineDenseMatrix("GIF.UVonCones2", UVonCones);
			MatlabGMMDataExchange::GetEngineDenseMatrix("GIF.UVonCones1", UVonCones1);
		}
		GMMDenseColMatrix arr2(1, 1);
		MatlabGMMDataExchange::GetEngineDenseMatrix("arr1", arr2);

		curE = arr2(0, 0);
		if (((iter1 > 0) && (abs(prevE - curE) / prevE < mOuterRateTerminationCondition)) || (iter1 == maxRemeshingTimes - 1) || !ret || curE <= mEnergyRelatedTerminationCondition)
			break;

		prevE = curE;
		if (iter1 == maxRemeshingTimes - 1)
			break;
		std::vector<std::complex<double>> allScaffoldPoints;
		for (int i = 0; i < mNumScaffoldDOF; i++)
		{
			allScaffoldPoints.push_back(Complex(UVonCones1(i, 0), UVonCones1(i, 1)));
		}

		Mesh curScaffold;
		double outerRadius = 0.0;
		cout << "Iter number  " << iter1 << " took " << t1.time() << endl;
		MeshAlgorithms::buildScaffoldWithFixedOuterBoundary(allScaffoldPoints, false, curScaffold, mNumDOF);
		cout << "Iter number  " << iter1 << " took " << t1.time() << endl;

		CGAL_Borders borders1(curScaffold);

		Eigen::SparseMatrix<double, Eigen::RowMajor> scaffoldKKtEigen1;

		getScaffoldBordersMapAndSetMetaVertices(borders1, curScaffold.size_of_vertices());
		UpdateIndexOfHEinScaffoldSystem(curScaffold);
		cout << "Iter number  " << iter1 << " took " << t1.time() << endl;
		TransferScaffoldBorderFacesAndVerticesToMatlab(curScaffold);
		cout << "Iter number  " << iter1 << " took " << t1.time() << endl;

		bool res = constructScaffoldHarmonicBasisAndSendToMATLAB();
		if (!res) {
			cout << "Failed to construct Harmonic Basis for the scaffold" << endl;
			return false;
		}
		int numVerticesScaffold = curScaffold.size_of_vertices();
		GMMDenseColMatrix scaffoldBoundaryVertices(2 * numVerticesScaffold, 1);
		for (int i = 0; i < numVerticesScaffold; i++)
		{
			scaffoldBoundaryVertices(i, 0) = mScaffoldConesAndMetaMap[i]->point().x();
			scaffoldBoundaryVertices(i + numVerticesScaffold, 0) = mScaffoldConesAndMetaMap[i]->point().y();
		}
		cout << "Iter number  " << iter1 << " took " << t1.time() << endl;
		MatlabGMMDataExchange::SetEngineDenseMatrix("scaffoldBoundaryVertices", scaffoldBoundaryVertices);
		MatlabInterface::GetEngine().EvalToCout("[ GIF.ScaffoldUVonCones, GIF.FixedIndices, GIF.FixedValues ] = fixFirstCone( scaffoldBoundaryVertices );");

		cout << "Iter number  " << iter1 << " took " << t1.time() << endl;

	}
	
	gmm::clear(RHS);
	gmm::resize(RHS, mSizeOfMatrix, 2);
	for (int i = 0; i < mNumDOF; i++)
	{
		int row = conesConstraintsStartRow + i;
		RHS(row, 0) = UVonCones(mNumScaffoldDOF + i, 0);
		RHS(row, 1) = UVonCones(mNumScaffoldDOF + i, 1);
	}

	return (bool)NewtonSucsess;

}


bool GIF::getAllUVsByRHS(GMMDenseColMatrix& RHS)
{
	GMMDenseColMatrix UVs(RHS.nrows(), RHS.ncols());
	bool res1 = mPardisoSolver.solve(RHS, UVs);
	MatlabGMMDataExchange::SetEngineDenseMatrix("array1", UVs);
	MatlabGMMDataExchange::GetEngineDenseMatrix("array1", mUVs);

	return true;
}


//Update the UV's of the halfedges, the input array UVs is ordered such there is a vector for y values and vector for x values ([x , y])
void GIF::updatHalfedgeUVs()
{
	mAllBoundaryUVs.clear();
	mAllBoundaryUVs.resize(mCgalMesh.size_of_border_edges());
	Facet_iterator faceIt = mCgalMesh.facets_begin();
	while (faceIt != mCgalMesh.facets_end())
	{
		Halfedge_handle h = faceIt->halfedge();
		//h->uv() = Point_3(mUVs(mIndexOfHEinSystem[h->index()], 0), mUVs(mIndexOfHEinSystem[h->index()], 1), 0);
		mCgalMesh.vertex(h->vertex()->index())->uvC(Complex(mUVs(mIndexOfHEinSystem[h->index()], 0), mUVs(mIndexOfHEinSystem[h->index()], 1)));
		if (h->opposite()->is_border())
		{
			mAllBoundaryUVs[mIndicesOfAllBorderVerticesInUVbyGeneralIndex[h->vertex()->index()]] = Complex(mUVs(mIndexOfHEinSystem[h->index()], 0), mUVs(mIndexOfHEinSystem[h->index()], 1));
		}
		h = h->next();
		//h->uv() = Point_3(mUVs(mIndexOfHEinSystem[h->index()], 0), mUVs(mIndexOfHEinSystem[h->index()], 1), 0);
		mCgalMesh.vertex(h->vertex()->index())->uvC(Complex(mUVs(mIndexOfHEinSystem[h->index()], 0), mUVs(mIndexOfHEinSystem[h->index()], 1)));
		if (h->opposite()->is_border())
		{
			mAllBoundaryUVs[mIndicesOfAllBorderVerticesInUVbyGeneralIndex[h->vertex()->index()]] = Complex(mUVs(mIndexOfHEinSystem[h->index()], 0), mUVs(mIndexOfHEinSystem[h->index()], 1));
		}
		h = h->next();
		//h->uv() = Point_3(mUVs(mIndexOfHEinSystem[h->index()], 0), mUVs(mIndexOfHEinSystem[h->index()], 1), 0);
		if (h->opposite()->is_border())
		{
			mAllBoundaryUVs[mIndicesOfAllBorderVerticesInUVbyGeneralIndex[h->vertex()->index()]] = Complex(mUVs(mIndexOfHEinSystem[h->index()], 0), mUVs(mIndexOfHEinSystem[h->index()], 1));
		}
		mCgalMesh.vertex(h->vertex()->index())->uvC(Complex(mUVs(mIndexOfHEinSystem[h->index()], 0), mUVs(mIndexOfHEinSystem[h->index()], 1)));
		faceIt++;
	}
}

void GIF::getResult()
{
	int numFoldovers, numFoldsNearCones, numFoldsNearBorder, numWrongAngles, numWrongConeAngles;
	std::vector<Facet_handle> flippedTriangles;
	checkForFoldovers(flippedTriangles);
	mFlipsData.clear();
	mFlipsData.resize(9, 1);
	mFlipsData(0, 0) = flippedTriangles.size();
	if (flippedTriangles.size() == 0)
	{
		cout << "No foldovers! No need to apply the foldovers fixing method!" << endl;
		mParameters_matrix(13, 0) = 0;
		return;
	}
	// If we are here then we have foldovers

	if (!mAllowMVCFix && !mAllowLocallFix)
	{
		mParameters_matrix(21, 0) = flippedTriangles.size();
		cout << mFlipsData(0, 0) << " flips! not applying fixing method anyway because of user's requirement!" << endl;
		mParameters_matrix(13, 0) = flippedTriangles.size();
		return;
	}
	

	if (mAllowLocallFix && flippedTriangles.size() < 1000) // We dont apply the KP fix for more than 1,000 flips
	{
		cout << "Applying the foldovers fixing method to fix " << flippedTriangles.size() << " foldovers..." << endl;
		fixCotFoldoversWithKernelPositioning(flippedTriangles, false);
		checkForFoldovers(flippedTriangles);
		mFlipsData(1, 0) = flippedTriangles.size();
		mFlipsData(5, 0) = 1;
		if (flippedTriangles.size() > 0)
		{
			fixCotFoldoversWithKernelPositioning(flippedTriangles, true);
			checkForFoldovers(flippedTriangles);
			mFlipsData(2, 0) = flippedTriangles.size();
			mFlipsData(6, 0) = 1;
		}
		checkForFoldovers(flippedTriangles);
		if (flippedTriangles.size() < mMaxFoldoversForIsoTLC) // If more than 20 flips remained, it's better to skip the isoTLC fix and apply the MVC fix
		{
			CGAL::Timer fixingTime1;
			fixingTime1.start();
			fixCotFoldoversWithIsoTLC(flippedTriangles);
			checkForFoldovers(flippedTriangles);
			mFlipsData(3, 0) = flippedTriangles.size();
			mFlipsData(7, 0) = 1;
			fixingTime1.stop();
			cout << "Fix with IsoTLC took" << fixingTime1.time() << endl;
		}
		if (flippedTriangles.size() == 0)
		{
			cout << "All foldovers were fixed succesfully!" << endl;
			return;
		}
		else
			cout << flippedTriangles.size() << " foldovers remained after the foldovers fixing method was applied" << endl; //this means that the foldovers fixing method didn't work. This is not supposed to happen, because our method has theoretical guarantees
	}

	if (mAllowMVCFix)
	{
		cout << "Applying the foldovers fixing method to fix " << flippedTriangles.size() << " foldovers..." << endl;
		fixCotFoldoversWithMeanValue(flippedTriangles);
		checkForFoldovers(flippedTriangles);
		mFlipsData(4, 0) = flippedTriangles.size();
		mFlipsData(8, 0) = 1;
		if (flippedTriangles.size() == 0)
			cout << "All foldovers were fixed succesfully!" << endl;
		else
			cout << flippedTriangles.size() << " foldovers remained after the foldovers fixing method was applied" << endl; //this means that the foldovers fixing method didn't work. This is not supposed to happen, because our method has theoretical guarantees
		mParameters_matrix(13, 0) = 1;
	}

}

bool isTriangleFlipped(Facet_handle& f)
{
	Point_2 a, b, c;
	Halfedge_around_facet_const_circulator h_f = f->facet_begin();
	a = Point_2(h_f->vertex()->uvC().real(), h_f->vertex()->uvC().imag());
	h_f++;
	b = Point_2(h_f->vertex()->uvC().real(), h_f->vertex()->uvC().imag());
	h_f++;
	c = Point_2(h_f->vertex()->uvC().real(), h_f->vertex()->uvC().imag());
	Kernel::Triangle_2 t(a, b, c);
	if (t.orientation() <= 0)
		cout << a << ", " << b << ", " << c << endl;
	return t.orientation() <= 0;
}

void GIF::checkForFoldovers(std::vector<Facet_handle>& flippedTriangles)
{
	flippedTriangles.clear();

	Point_2 a, b, c;
	Facet_iterator faceIt = mCgalMesh.facets_begin();
	while (faceIt != mCgalMesh.facets_end())
	{
		/*Halfedge_handle h = faceIt->halfedge();
		a = Point_2(h->uv().x(), h->uv().y());
		h = h->next();
		b = Point_2(h->uv().x(), h->uv().y());
		h = h->next();
		c = Point_2(h->uv().x(), h->uv().y());
		Kernel::Triangle_2 t(a, b, c);
		if (t.orientation() <= 0) {
			flippedTriangles.push_back(faceIt);
		}*/

		if(isTriangleFlipped(faceIt))
			flippedTriangles.push_back(faceIt);
		faceIt++;
	}
}

void GIF::openReportWindow()
{
	MatlabGMMDataExchange::GetEngineDenseMatrix("array1", mUVs);
	updatHalfedgeUVs();
	std::vector<Facet_handle> flippedTriangles;
	checkForFoldovers(flippedTriangles);
	mParameters_matrix(12, 0) = flippedTriangles.size();

	bool isFoldoversFree = (flippedTriangles.size() > 0);
	bool isSimple = CompilationAccelerator::isSimplePolygon(mAllBoundaryUVs, false);
	if (isFoldoversFree == 0 && isSimple) {
		cout << "The final result is globally injective!" << endl;
		mParameters_matrix(11, 0) = 1;
	}
	else
	{
		mParameters_matrix(11, 0) = 0;
	}

	MatlabGMMDataExchange::SetEngineDenseMatrix("GIF.parametersMatrix", mParameters_matrix);
	MatlabInterface::GetEngine().EvalToCout("compute_final_energy;");
	MatlabInterface::GetEngine().EvalToCout("GIF_report_window(GIF);");
}

bool returnKernelCenter(Vertex_const_handle v, Complex& center)
{
	Halfedge_around_vertex_const_circulator h_v = v->vertex_begin();
	const Mesh::Halfedge_around_vertex_const_circulator h_v_end = h_v;
	//cout << "Center index: " << v->index() << endl;
	Polygon P;
	CGAL_For_all(h_v, h_v_end)
	{
		Complex c1 = h_v->opposite()->vertex()->uvC();
		P.push_back(Point2(c1.real(), c1.imag()));
		//cout << "Neighbor index: " << h_v->opposite()->vertex()->index() << endl;
	}
	P.reverse_orientation();
	if (!P.is_simple()) {
		return false;
	}
	else
	{
		Polygon K = kernel_of_polygon(P);

		if (K.is_empty()) {
			return false;
		}

		std::cout << std::fixed << std::setprecision(10);
		Point2 C;
		if (area_centroid(K, C)) {
			center = Complex(CGAL::to_double(C.x()), CGAL::to_double(C.y()));
			return true;
		}
		else {
			std::cout << "Centroid undefined (degenerate kernel).\n";
			return false;
		}
	}
}

void GIF::fixCotFoldoversWithKernelPositioning(std::vector<Facet_handle> flippedTriangles, bool allowSecondOrderFix)
{
	int solvedFirstIter = 0;
	int solvedSecondIter = 0;
	for (Facet_handle fi : flippedTriangles)
	{
		if (!isTriangleFlipped(fi)) // The flips was fixed already by fixing another flip
			continue;
		Halfedge_around_facet_const_circulator h_f = fi->facet_begin();
		const Halfedge_around_facet_const_circulator hEnd = h_f;

		int numVerticesInCurrentFace = 0;
		bool secondLevelFixSolved = false;
		CGAL_For_all(h_f, hEnd)
		{
			Complex center;
			bool res = returnKernelCenter(h_f->vertex(), center);
			if (res)
			{
				mCgalMesh.vertex(h_f->vertex()->index())->uvC(center);
				solvedFirstIter++;
				break;
			}
			else
			{
				if (allowSecondOrderFix)
				{
					Halfedge_around_vertex_const_circulator h_v = h_f->vertex()->vertex_begin();
					const Mesh::Halfedge_around_vertex_const_circulator h_v_end = h_v;
					//cout << "Center index: " << h_f->vertex()->index() << endl;
					Polygon P;
					CGAL_For_all(h_v, h_v_end)
					{
						Complex neighbotCenter;
						bool res = returnKernelCenter(h_v->opposite()->vertex(), neighbotCenter);
						if (!res)
							continue;
						else
						{
							const Complex neighborOldUVs = h_v->opposite()->vertex()->uvC();
							mCgalMesh.vertex(h_v->opposite()->vertex()->index())->uvC(neighbotCenter);
							bool res = returnKernelCenter(h_f->vertex(), center);
							if (res)
							{
								mCgalMesh.vertex(h_f->vertex()->index())->uvC(center);
								secondLevelFixSolved = true;
								solvedSecondIter++;
								break;
							}
							else
							{
								mCgalMesh.vertex(h_v->opposite()->vertex()->index())->uvC(neighborOldUVs);
								continue;
							}
						}
					}
					if (secondLevelFixSolved)
						break;
				}

				continue;
			}
		}
	}
	/*auto heIt = mCgalMesh.halfedges_begin();
	while (heIt != mCgalMesh.halfedges_end())
	{
		heIt->uv() = Point_3(heIt->vertex()->uvC().real(), heIt->vertex()->uvC().imag(), 0);
		mUVs(mIndexOfHEinSystem[heIt->index()], 0) = heIt->vertex()->uvC().real();
		mUVs(mIndexOfHEinSystem[heIt->index()], 1) = heIt->vertex()->uvC().imag();
		heIt++;
	}*/
	cout << "Kernel positioning got " << flippedTriangles.size() << " flips" << endl;
	cout << "Kernel positioning fixed " << solvedFirstIter << " flips in the first iteration" << endl;
	cout << "Kernel positioning fixed " << solvedSecondIter << " flips in the second iteration" << endl;

}

void GIF::fixCotFoldoversWithIsoTLC(std::vector<Facet_handle> flippedTriangles)
{
	int solvedFirstIter = 0;
	for (Facet_handle fi: flippedTriangles)
	{
		if (!isTriangleFlipped(fi)) // The flips was fixed already by fixing another flip
			continue;
		Facet_const_handle fc = fi;
		if (fixFlipWithIsoTLC(fc))
			solvedFirstIter++;
	}

	/*auto heIt = mCgalMesh.halfedges_begin();
	while (heIt != mCgalMesh.halfedges_end())
	{
		heIt->uv() = Point_3(heIt->vertex()->uvC().real(), heIt->vertex()->uvC().imag(), 0);
		mUVs(mIndexOfHEinSystem[heIt->index()], 0) = heIt->vertex()->uvC().real();
		mUVs(mIndexOfHEinSystem[heIt->index()], 1) = heIt->vertex()->uvC().imag();
		heIt++;
	}*/

	cout << "IsoTLC got " << flippedTriangles.size() << " flips" << endl;
	cout << "IsoTLC fixed " << solvedFirstIter << " flips" << endl;
}

void GIF::fixCotFoldoversWithMeanValue(std::vector<Facet_handle> flippedTriangles)
{
	CGAL::Timer fixingTime1;
	fixingTime1.start();
	GMMDenseColMatrix UVonCones1(mNumScaffoldDOF, 2);
	MatlabGMMDataExchange::GetEngineDenseMatrix("GIF.UVonCones1", UVonCones1);
	std::vector<std::complex<double>> internalPolygonPoints1;
	std::vector<int> indicesInGluedMeshByIndicesInScaffold;
	double outerRadiusOfScaffold = 0.0;
	for (int i = 0; i < (mNumScaffoldDOF - mNumDOF); i++)
	{
		Complex c1 = Complex(UVonCones1(i, 0), UVonCones1(i, 1));
		indicesInGluedMeshByIndicesInScaffold.push_back(mCgalMesh.size_of_vertices() + i);
	}

	for (int i = 0; i < (mNumScaffoldDOF - mNumDOF); i++)
	{
		Complex c1 = 2.0 * Complex(UVonCones1(i, 0), UVonCones1(i, 1));
		internalPolygonPoints1.push_back(c1);
	}

	for (int i = 0; i < mCgalMesh.size_of_border_edges(); i++)
	{
		internalPolygonPoints1.push_back(mAllBoundaryUVs[i]);
		indicesInGluedMeshByIndicesInScaffold.push_back(mGeneralIndicesUVbyIndicesOfAllBorderVertices[i]);
	}

	Mesh scaffold1;
	double outerRadius = 0.0;

	MeshAlgorithms::buildScaffoldWithFixedOuterBoundary(internalPolygonPoints1, false, scaffold1, mCgalMesh.size_of_border_edges());

	std::vector<Point_3> vertices1;
	std::vector<unsigned int> faceIndices1;
	std::vector<unsigned int> vertexCount1;
	int numVertices1 = mCgalMesh.size_of_vertices();
	int numFaces1 = mCgalMesh.size_of_facets();
	int numFaces2 = scaffold1.size_of_facets();
	GMMDenseColMatrix vertices2(numVertices1, 3);
	GMMDenseColMatrix faces2(numFaces1, 3);
	GMMDenseColMatrix uvsBeforeFixing(numVertices1, 2);
	GMMDenseColMatrix uvsAfterFixing(numVertices1, 2);


	for (int i = 0; i < numVertices1; i++) //inserting the vertices of the original mesh
	{
		Vertex_handle vi = mCgalMesh.vertex(i);
		Complex c1 = vi->uvC();
		Mesh::Point_3 p(c1.real(), c1.imag(), 0.0);
		uvsBeforeFixing(i, 0) = c1.real();
		uvsBeforeFixing(i, 1) = c1.imag();
		vertices1.push_back(p);
	}
	for (int i = 0; i < (mNumScaffoldDOF - mNumDOF); i++)
	{
		Mesh::Point_3 p(scaffold1.vertex(i)->point().x(), scaffold1.vertex(i)->point().y(), 0.0);
		vertices1.push_back(p);
	}
	for (int i1 = 0; i1 < numFaces1; i1++)
	{
		Facet_const_handle f = mCgalMesh.face(i1);

		Halfedge_around_facet_const_circulator h = f->facet_begin();
		const Halfedge_around_facet_const_circulator hEnd = h;

		int numVerticesInCurrentFace = 0;
		CGAL_For_all(h, hEnd)
		{
			faceIndices1.push_back(h->vertex()->index());
			numVerticesInCurrentFace++;
		}
		vertexCount1.push_back(numVerticesInCurrentFace);
	}

	for (int i1 = 0; i1 < numFaces2; i1++)
	{
		Facet_const_handle f = scaffold1.face(i1);

		Halfedge_around_facet_const_circulator h = f->facet_begin();
		const Halfedge_around_facet_const_circulator hEnd = h;

		int numVerticesInCurrentFace = 0;

		CGAL_For_all(h, hEnd)
		{
			faceIndices1.push_back(indicesInGluedMeshByIndicesInScaffold[h->vertex()->index()]);
			numVerticesInCurrentFace++;
		}
		vertexCount1.push_back(numVerticesInCurrentFace);
	}
	Mesh gluedMesh;

	MeshAlgorithms::buildTriangleMesh(vertices1, faceIndices1, gluedMesh);
	std::vector<Eigen::Triplet<double>> tripletListValues1, tripletListValues2, tripletListValues3;
	int gluedMeshBoundaryVerticesCount = gluedMesh.size_of_border_edges();
	int gluedMeshInternalVerticesCount = gluedMesh.size_of_vertices() - gluedMeshBoundaryVerticesCount;
	tripletListValues1.reserve(7 * gluedMesh.size_of_vertices());
	tripletListValues2.reserve(3 * gluedMeshBoundaryVerticesCount);
	//fill the matrix by going on the triangles one by one
	std::vector<int> internalIndices(gluedMesh.size_of_vertices());
	std::vector<int> boundaryIndices(gluedMesh.size_of_vertices());
	GMMDenseColMatrix RHS(gluedMeshBoundaryVerticesCount, 2);
	int num1 = 0, num2 = 0;
	for_each_vertex(vi, gluedMesh)
	{
		if (!vi->is_border())
		{
			internalIndices[vi->index()] = num1;
			num1++;
		}
		if (vi->is_border())
		{
			boundaryIndices[vi->index()] = num2;
			RHS(num2, 0) = -vi->point().x();
			RHS(num2, 1) = -vi->point().y();
			num2++;
		}
	}

	int r = 0; //num free vertices
	int k = 0; //num fixed/dependent vertices
	std::vector<Eigen::Triplet<double>> tripletListValues;
	tripletListValues.reserve(6 * gluedMesh.size_of_vertices() + mNumDOF);
	int n = gluedMesh.countVertexStates(r, k);
	assert(n == r + k);


	std::vector<double> curVertexRow;
	//fill the matrix by going over vertices
	for_each_vertex(v, gluedMesh)
	{
		int i = v->index();
		if (!v->is_border())
		{

			Mesh::Halfedge_around_vertex_circulator h = v->vertex_begin();
			const Mesh::Halfedge_around_vertex_circulator hEnd = h;
			int numNeighbors = 0;
			CGAL_For_all(h, hEnd)
				numNeighbors++;
			curVertexRow.clear();
			curVertexRow.resize(numNeighbors);
			int neighborOneRingIndex = 0;
			double sumWeights = 0.0;
			CGAL_For_all(h, hEnd)
			{
				double w = h->opposite()->meanValueWeight(false);
				curVertexRow[neighborOneRingIndex] = w;
				neighborOneRingIndex++;
				sumWeights += w;
			}

			neighborOneRingIndex = 0;

			//normalize the weights
			//this is not mandatory since the system should be full ranked, but may improve numerical stability
			CGAL_For_all(h, hEnd)
			{
				int neighborVertexIndex = h->opposite()->vertex()->index();
				double w = curVertexRow[neighborOneRingIndex];
				neighborOneRingIndex++;
				if (!h->opposite()->vertex()->is_border())
					tripletListValues1.push_back(Eigen::Triplet<double>(internalIndices[i], internalIndices[neighborVertexIndex], -w / sumWeights));
				else
					tripletListValues2.push_back(Eigen::Triplet<double>(internalIndices[i], boundaryIndices[neighborVertexIndex], -w / sumWeights));
			}
			tripletListValues1.push_back(Eigen::Triplet<double>(internalIndices[i], internalIndices[i], 1.0));
		}
	}

	Eigen::SparseMatrix<double, Eigen::RowMajor> laplacian1;
	laplacian1.resize(gluedMeshInternalVerticesCount, gluedMeshInternalVerticesCount);
	laplacian1.setFromTriplets(tripletListValues1.begin(), tripletListValues1.end());
	laplacian1.makeCompressed();

	Eigen::SparseMatrix<double, Eigen::RowMajor> mat_L12;
	mat_L12.resize(gluedMeshInternalVerticesCount, gluedMeshBoundaryVerticesCount);
	mat_L12.setFromTriplets(tripletListValues2.begin(), tripletListValues2.end());
	mat_L12.makeCompressed();
	const Eigen::SparseMatrix<double> M = mat_L12;
	GMMSparseRowMatrix mat_L12_mean_value(M.rows(), M.cols());

	for (unsigned k = 0; k < (unsigned)M.outerSize(); ++k)
	{
		for (Eigen::SparseMatrix<double>::InnerIterator it(M, k); it; ++it)
		{
			mat_L12_mean_value(it.row(), it.col()) = it.value();
		}
	}
	GMMDenseColMatrix rhs_elimination(gluedMeshInternalVerticesCount, 2);
	gmm::mult(mat_L12_mean_value, RHS, rhs_elimination);
	GMMDenseColMatrix solution(gluedMeshInternalVerticesCount, 2);
	int	mtype = 11;
	PardisoLinearSolver pardisoSolver(mtype);
	bool res = pardisoSolver.init(false);
	res = pardisoSolver.createPardisoFormatMatrix(laplacian1);
	res = pardisoSolver.preprocess();
	pardisoSolver.solve(rhs_elimination, solution);
	mParameters_matrix(4, 0) = mParameters_matrix(4, 0) + fixingTime1.time();
	num1 = 0;
	for_each_vertex(vi, gluedMesh)
	{
		if (!vi->is_border())
		{
			double x = solution(num1, 0);
			double y = solution(num1, 1);
			vi->uvC(Complex(x, y));
			num1++;
		}
		if (vi->is_border())
		{
			double x = vi->point().x();
			double y = vi->point().y();
			vi->uvC(Complex(x, y));
		}
	}
	for (int i = 0; i < mCgalMesh.size_of_vertices(); i++)
	{
		mCgalMesh.vertex(i)->uvC(gluedMesh.vertex(i)->uvC());
		Complex c1 = gluedMesh.vertex(i)->uvC();
		uvsAfterFixing(i, 0) = c1.real();
		uvsAfterFixing(i, 1) = c1.imag();
	}
	/*auto heIt = mCgalMesh.halfedges_begin();
	while (heIt != mCgalMesh.halfedges_end())
	{
		heIt->uv() = Point_3(heIt->vertex()->uvC().real(), heIt->vertex()->uvC().imag(), 0);
		mUVs(mIndexOfHEinSystem[heIt->index()], 0) = heIt->vertex()->uvC().real();
		mUVs(mIndexOfHEinSystem[heIt->index()], 1) = heIt->vertex()->uvC().imag();
		heIt++;
	}
	MatlabInterface::GetEngine().EvalToCout("clear array1;");
	MatlabGMMDataExchange::SetEngineDenseMatrix("array1", mUVs);*/
}

double compute_Esd_times_area_around_vertex(Vertex_const_handle& v)
{
	Halfedge_around_vertex_const_circulator h = v->vertex_begin();
	const Mesh::Halfedge_around_vertex_const_circulator hEnd = h;
	double e_sd = 0;
	CGAL_For_all(h, hEnd)
	{
		Facet_const_handle fch = h->facet();
		double cur_e_sd = extract_symmetric_dirichlet_energy(fch);
		double cur_area = fch->area();
		double cur_e_sd_area = cur_area * cur_e_sd;
		e_sd += cur_e_sd_area;
	}
	return e_sd;
}

bool isFaceEnvironmentSimplePolygon(GMMDenseColMatrix& patchF, GMMDenseColMatrix& patchUVs)
{
	unordered_set<Facet_const_handle>::const_iterator itr;
	MatrixX3i F(patchF.nrows(), 3);
	for (int i = 0; i < patchF.nrows(); i++)
		for (int j = 0; j < 3; j++)
			F(i, j) = patchF(i, j);
	
	std::vector<size_t> boundary_vertices;
	Eigen::VectorXi bnd; Eigen::MatrixXd bnd_uv;
	igl::boundary_loop(F, bnd);
	Polygon P;
	for (int i = 0; i < bnd.rows(); i++)
		P.push_back(Point2(patchUVs(bnd[i], 0), patchUVs(bnd[i], 1)));
	return P.is_simple();
}

void GIF::addFacesNextLevel(unordered_set<Facet_const_handle>& facesEnvironment)
{
	unordered_set<Facet_const_handle>::const_iterator itr;
	unordered_set<Facet_const_handle> facesEnvironmentNextLevel;
	for (itr = facesEnvironment.begin(); itr != facesEnvironment.end(); ++itr) {
		Facet_const_handle f = (*itr);
		Halfedge_around_facet_const_circulator h_f = f->facet_begin();
		const Halfedge_around_facet_const_circulator h_f_End = h_f;
		CGAL_For_all(h_f, h_f_End) // Iterating over all hafledges around each face
		{
			Vertex_const_handle vh = h_f->vertex();
			Halfedge_around_vertex_const_circulator h_v = vh->vertex_begin();
			const Mesh::Halfedge_around_vertex_const_circulator h_v_End = h_v;
			CGAL_For_all(h_v, h_v_End)
			{
				if (!h_v->is_border())
					facesEnvironmentNextLevel.insert(h_v->face());
			}
		}
	}
	facesEnvironment.insert(facesEnvironmentNextLevel.begin(), facesEnvironmentNextLevel.end());
}

void GIF::adjustPatch(unordered_set<Facet_const_handle>& facesEnvironment, GMMDenseColMatrix& patchF, GMMDenseColMatrix& patchV, GMMDenseColMatrix& patchUVs, std::unordered_map<int, int>& patch_to_original_mesh)
{
	std::unordered_map<int, int> original_mesh_to_patch;
	unordered_set<Vertex_const_handle> verticesEnvironment;
	patchF.clear();
	patchF.resize(facesEnvironment.size(), 3);
	unordered_set<Facet_const_handle>::const_iterator itr;
	int i = 0;
	for (itr = facesEnvironment.begin(); itr != facesEnvironment.end(); ++itr) {
		i++;
		Facet_const_handle f = (*itr);
		Halfedge_around_facet_const_circulator h_f1 = f->facet_begin();
		const Halfedge_around_facet_const_circulator hEnd1 = h_f1;
		CGAL_For_all(h_f1, hEnd1)
			verticesEnvironment.insert(h_f1->vertex());
	}
	patchV.clear();
	patchV.resize(verticesEnvironment.size(), 3);
	patchUVs.clear();
	patchUVs.resize(verticesEnvironment.size(), 2);
	unordered_set<Vertex_const_handle>::const_iterator itr1;
	int patch_v_ind = 0;
	for (itr1 = verticesEnvironment.begin(); itr1 != verticesEnvironment.end(); ++itr1) {
		Vertex_const_handle v = (*itr1);
		original_mesh_to_patch[v->index()] = patch_v_ind;
		patch_to_original_mesh[patch_v_ind] = v->index();
		patchV(patch_v_ind, 0) = v->point().x();
		patchV(patch_v_ind, 1) = v->point().y();
		patchV(patch_v_ind, 2) = v->point().z();
		patchUVs(patch_v_ind, 0) = v->uvC().real();
		patchUVs(patch_v_ind, 1) = v->uvC().imag();
		patch_v_ind++;
	}
	int ind_f = 0;
	for (itr = facesEnvironment.begin(); itr != facesEnvironment.end(); ++itr) {
		Facet_const_handle f = (*itr);
		Halfedge_around_facet_const_circulator h_f1 = f->facet_begin();
		const Halfedge_around_facet_const_circulator hEnd1 = h_f1;
		int v_ind = 0;
		CGAL_For_all(h_f1, hEnd1)
		{
			patchF(ind_f, v_ind) = original_mesh_to_patch[h_f1->vertex()->index()];
			v_ind++;
		}
		ind_f++;
	}
}

bool GIF::computeFaceEnvironment(Facet_const_handle& face, GMMDenseColMatrix& patchF, GMMDenseColMatrix& patchV, GMMDenseColMatrix& patchUVs, GMMDenseColMatrix& res, std::vector<size_t> boundary_vertices, std::unordered_map<int, int>& patch_to_original_mesh, int minLevels, int maxIters)
{
	unordered_set<Facet_const_handle> facesEnvironment;
	unordered_set<Vertex_const_handle> verticesEnvironment;

	facesEnvironment.insert(face);
	int numLevelsNeeded = 0;
	for (int i = 0; i < minLevels - 1; i++)
	{
		/*cout << "Current face emvironemnt is of size " << facesEnvironment.size() << endl;
		cout << "Expanding environment to level " << numLevelsNeeded + 1 << endl;*/
		addFacesNextLevel(facesEnvironment);
		numLevelsNeeded++;
	}
	adjustPatch(facesEnvironment, patchF, patchV, patchUVs, patch_to_original_mesh);
	while (true)
	{
		if (isFaceEnvironmentSimplePolygon(patchF, patchUVs))
			if (runIsoTLC2D(patchV, patchF, patchUVs, res, boundary_vertices, maxIters))
				break;
		addFacesNextLevel(facesEnvironment);
		adjustPatch(facesEnvironment, patchF, patchV, patchUVs, patch_to_original_mesh);
		numLevelsNeeded++;
		if (numLevelsNeeded == 10)
			return false;
	}
	//cout << "Needed " << numLevelsNeeded << "Levels for face environment" << endl;
	return true;

}


bool GIF::fixFlipWithIsoTLC(Facet_const_handle& face)
{
	GMMDenseColMatrix patchF, patchV, patchUVs;
	std::unordered_map<int, int> patch_to_original_mesh;
	GMMDenseColMatrix res;
	std::vector<size_t> boundary_vertices;
	bool isoTLCSucceed = computeFaceEnvironment(face, patchF, patchV, patchUVs, res, boundary_vertices, patch_to_original_mesh, 3, 50);
	if (!isoTLCSucceed)
		return false;

	GMMDenseColMatrix boundary(boundary_vertices.size(), 1);
	for (int i = 0; i < boundary_vertices.size(); i++)
		boundary(i, 0) = boundary_vertices[i];

	std::vector<bool> isPatchVertexInternal;
	isPatchVertexInternal.resize(patchV.nrows());
	for (int i = 0; i < patchV.nrows(); i++)
		isPatchVertexInternal[i] = true;
	for (size_t i : boundary_vertices)
		isPatchVertexInternal[i] = false;

	for (int i = 0; i < patchV.nrows(); i++)
		mCgalMesh.vertex(patch_to_original_mesh[i])->uvC(Complex(res(i, 0), res(i, 1)));
	
	return true;

}

void GIF::fixOutliersInFinalResult()
{
	/*GMMDenseColMatrix e_sd_values(mCgalMesh.size_of_facets(), 1);
	GMMDenseColMatrix areas(mCgalMesh.size_of_facets(), 1);*/
	CGAL::Timer t1;
	t1.start();
	vector<double> e_sd_area_per_face;
	e_sd_area_per_face.reserve((mCgalMesh.size_of_facets()) + 10);
	double e_sd = 0;
	double total_area = 0;
	for_each_const_facet(f, mCgalMesh)
	{
		double cur_e_sd = extract_symmetric_dirichlet_energy(f);
		double cur_area = f->area();
		double cur_e_sd_area = cur_area * cur_e_sd;
		e_sd += cur_e_sd_area;
		total_area += cur_area;
		e_sd_area_per_face.push_back(cur_e_sd_area);
	}
	e_sd = e_sd / total_area;
	cout << "Esd before fix = " << e_sd << endl;
	/*if (e_sd < 10)
		return;*/

	std::vector<size_t> indices(e_sd_area_per_face.size());
	for (size_t i = 0; i < indices.size(); ++i)
		indices[i] = i;
	std::sort(indices.begin(), indices.end(),
		[&e_sd_area_per_face](size_t i1, size_t i2) { return e_sd_area_per_face[i1] > e_sd_area_per_face[i2]; });

	const int FACES_COUNT_TO_FIX = 0.0001 * mCgalMesh.size_of_facets();

	for (int i = 0; i < FACES_COUNT_TO_FIX; i++)
	{
		Facet_const_handle fc = mCgalMesh.face(indices[i]);
		fixOutlierFace(fc);
	}

	t1.stop();
	cout << "Fixing outliers took " << t1.time() << endl;

	/*auto heIt = mCgalMesh.halfedges_begin();
	while (heIt != mCgalMesh.halfedges_end())
	{
		heIt->uv() = Point_3(heIt->vertex()->uvC().real(), heIt->vertex()->uvC().imag(), 0);
		mUVs(mIndexOfHEinSystem[heIt->index()], 0) = heIt->vertex()->uvC().real();
		mUVs(mIndexOfHEinSystem[heIt->index()], 1) = heIt->vertex()->uvC().imag();
		heIt++;
	}*/
}


void GIF::fixOutlierFace(Facet_const_handle& face)
{
	GMMDenseColMatrix patchF, patchV, patchUVs;
	std::unordered_map<int, int> patch_to_original_mesh;
	GMMDenseColMatrix res;
	std::vector<size_t> boundary_vertices;
	bool isoTLCSucceed = computeFaceEnvironment(face, patchF, patchV, patchUVs, res, boundary_vertices, patch_to_original_mesh, 3, 5);
	if (!isoTLCSucceed)
		return;

	GMMDenseColMatrix boundary(boundary_vertices.size(), 1);
	for (int i = 0; i < boundary_vertices.size(); i++)
		boundary(i, 0) = boundary_vertices[i];

	std::vector<bool> isPatchVertexInternal;
	isPatchVertexInternal.resize(patchV.nrows());
	for (int i = 0; i < patchV.nrows(); i++)
		isPatchVertexInternal[i] = true;
	for (size_t i : boundary_vertices)
		isPatchVertexInternal[i] = false;

	for (int i = 0; i < patchV.nrows(); i++)
		mCgalMesh.vertex(patch_to_original_mesh[i])->uvC(Complex(res(i, 0), res(i, 1)));
}
