#include "spline.h"
#include "itkImage.h"
#include "itkImportImageFilter.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkGradientRecursiveGaussianImageFilter.h"
#include "itkGradientMagnitudeRecursiveGaussianImageFilter.h"
#include "itkCovariantVector.h"
#include "itkImageRegionConstIterator.h"
#include "itkImageRegionConstIteratorWithIndex.h"
#include "itkGDCMImageIO.h"
#include "itkGDCMSeriesFileNames.h"
#include "itkImageSeriesReader.h"
#include "itkImageFileWriter.h"
#include "itkNumericSeriesFileNames.h"

#include <iostream>
#include <cstring>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <sstream>


using namespace std;

const float OUT_OF_BOUNDS = 1e4;

// Returns index of is the matrix of size I x J x K
inline long int IND(int I, int J, int K, long int i, long int j, long int k) {
	return ((i) * J * K + j * K + k);
}
;


CRSpline::CRSpline() :
		vp(), delta_t(0) {
}

CRSpline::CRSpline(const CRSpline& s) {
	for (int i = 0; i < (int) s.vp.size(); i++)
		vp.push_back(s.vp[i]);
	delta_t = s.delta_t;
}

CRSpline::~CRSpline() {
}

// Solve the Catmull-Rom parametric equation for a given time(t) and vector quadruple (p1,p2,p3,p4)
vec3 CRSpline::Eq(float t, const vec3& p1, const vec3& p2, const vec3& p3,
		const vec3& p4) {
	float t2 = t * t;
	float t3 = t2 * t;

	float b1 = .5 * (-t3 + 2 * t2 - t);
	float b2 = .5 * (3 * t3 - 5 * t2 + 2);
	float b3 = .5 * (-3 * t3 + 4 * t2 + t);
	float b4 = .5 * (t3 - t2);

	return (p1 * b1 + p2 * b2 + p3 * b3 + p4 * b4);
}

vec3 CRSpline::d_dt__Eq(float t, const vec3& p1, const vec3& p2, const vec3& p3,
		const vec3& p4) {
	float t2 = t * t;

	float b1 = .5 * (-3 * t2 + 2 * 2 * t - 1);
	float b2 = .5 * (3 * 3 * t2 - 2 * 5 * t);
	float b3 = .5 * (-3 * 3 * t2 + 2 * 4 * t + 1);
	float b4 = .5 * (3 * t2 - 2 * t);

	return (p1 * b1 + p2 * b2 + p3 * b3 + p4 * b4);
}

void CRSpline::AddSplinePoint(const vec3& v) {
	vp.push_back(v);
	if (vp.size() > 1)
		delta_t = (float) 1 / (float) (vp.size() - 1);
	else
		delta_t = (float) 1 / (float) vp.size();
}

// Updates the N points from matrix P (N x 3)
void CRSpline::update_spline_points(int i, int j, float* P) {
	if (j != 3) {
		throw "P must have size of [N x 3] for N points.";
	}
	if (i != vp.size()) {
		throw "P must have same size of [N x 3] where N is the number of points in spline.";
	}
	vp.clear();
	for (int p = 0; p < i; p++) {
		CRSpline::AddSplinePoint(
				vec3(P[p * 3 + 0], P[p * 3 + 1], P[p * 3 + 2]));
	}
}

// Adds the N points from matrix P (N x 3)
void CRSpline::add_spline_points(int i, int j, float* P) {
	if (j != 3) {
		throw "P must have size of [N x 3] for N points.";
	}
	for (int p = 0; p < i; p++) {
		CRSpline::AddSplinePoint(
				vec3(P[p * 3 + 0], P[p * 3 + 1], P[p * 3 + 2]));
	}
}

vec3 CRSpline::GetInterpolatedSplinePoint(float t) {
	if (vp.size() == 0)
		throw "At least one control point is needed for interpolation";
	// Find out in which interval we are on the spline
	int p = (int) (t / delta_t);
	// Compute local control point indices
#define BOUNDS(pp) { if (pp < 0) pp = 0; else if (pp > (int)vp.size()-1) pp = vp.size() - 1; }
	int p0 = p - 1;
	BOUNDS(p0);
	int p1 = p;
	BOUNDS(p1);
	int p2 = p + 1;
	BOUNDS(p2);
	int p3 = p + 2;
	BOUNDS(p3);
	// Relative (local) time
	float lt = (t - delta_t * (float) p) / delta_t;
	// Interpolate
	return CRSpline::Eq(lt, vp[p0], vp[p1], vp[p2], vp[p3]);
}

vec3 CRSpline::GetInterpolatedSplineTangent(float t) {
	if (vp.size() == 0)
		throw "At least one control point is needed for interpolation";
	// Find out in which interval we are on the spline
	int p = (int) (t / delta_t);
	// Compute local control point indices
#define BOUNDS(pp) { if (pp < 0) pp = 0; else if (pp > (int)vp.size()-1) pp = vp.size() - 1; }
	int p0 = p - 1;
	BOUNDS(p0);
	int p1 = p;
	BOUNDS(p1);
	int p2 = p + 1;
	BOUNDS(p2);
	int p3 = p + 2;
	BOUNDS(p3);
	// Relative (local) time
	float lt = (t - delta_t * (float) p) / delta_t;
	// Interpolate
	vec3 n = CRSpline::d_dt__Eq(lt, vp[p0], vp[p1], vp[p2], vp[p3]);
	n.normalize();
	return n;
}

void CRSpline::get_interpolated_spline_point(float t, int* i, float** p) {
	// Compute local control point
	vec3 v = CRSpline::GetInterpolatedSplinePoint(t);

	(*i) = 3;
	(*p) = new float[3];
	(*p)[0] = v.x;
	(*p)[1] = v.y;
	(*p)[2] = v.z;
}

void CRSpline::get_interpolated_spline_tangent(float t, int* i, float** p) {
	// Compute local control point
	vec3 v = CRSpline::GetInterpolatedSplineTangent(t);

	(*i) = 3;
	(*p) = new float[3];
	(*p)[0] = v.x;
	(*p)[1] = v.y;
	(*p)[2] = v.z;
}

void CRSpline::get_interpolated_spline_point(int i, float* T, int* j, int* k,
		float** pts) {
	(*j) = i;
	(*k) = 3;
	(*pts) = new float[3 * i];
	for (int p = 0; p < i; p++) {
		// Compute local control point
		vec3 v = CRSpline::GetInterpolatedSplinePoint(T[p]);

		(*pts)[p * 3 + 0] = v.x;
		(*pts)[p * 3 + 1] = v.y;
		(*pts)[p * 3 + 2] = v.z;
	}
}

void CRSpline::get_interpolated_spline_tangent(int i, float* T, int* j, int* k,
		float** pts) {
	(*j) = i;
	(*k) = 3;
	(*pts) = new float[3 * i];
	for (int p = 0; p < i; p++) {
		// Compute local control point
		vec3 v = CRSpline::GetInterpolatedSplineTangent(T[p]);

		(*pts)[p * 3 + 0] = v.x;
		(*pts)[p * 3 + 1] = v.y;
		(*pts)[p * 3 + 2] = v.z;
	}
}

// calculate accumulative distance along points T
// T is expected to be sorted.
void CRSpline::get_interpolated_spline_distance(int i, float* T,
		int S_i, float* S,
		int* V_i, float** V) {

	if (S_i != 3)
		throw "Volume spacing vector S must be 3 vector, for spacing of X, Y, Z";

	(*V_i) = i;
	(*V) = new float[i];
	float dist = 0;
	vec3 p0 = CRSpline::GetInterpolatedSplinePoint(0);
	float dt = 0.0001;
	float t = dt;
	// calculate distance to first point
	while (t < T[0]) {
		vec3 p1 = CRSpline::GetInterpolatedSplinePoint(t);
		vec3 d = p1 - p0;
		// scale distance
		d.x *= S[0];
		d.y *= S[1];
		d.z *= S[2];

		p0 = p1;
		dist += sqrt(d * d);
		t += dt;
	}
	for (int p = 0; p < i; p++) {
		// Compute local control point
		while (t < T[p]) {
			vec3 p1 = CRSpline::GetInterpolatedSplinePoint(T[p]);
			vec3 d = p1 - p0;
			// scale distance
			d.x *= S[0];
			d.y *= S[1];
			d.z *= S[2];

			p0 = p1;
			dist += sqrt(d * d);
			t += dt;
		}
		(*V)[p] = dist;
	}
}

float CRSpline::get_interpolated_spline_distance(float T,
		int S_i, float* S) {

	if (S_i != 3)
		throw "Volume spacing vector S must be 3 vector, for spacing of X, Y, Z";

	float dist = 0;
	vec3 p0 = CRSpline::GetInterpolatedSplinePoint(0);
	float dt = 0.0001;
	float t = dt;
	// calculate distance to first point
	while (t < T) {
		vec3 p1 = CRSpline::GetInterpolatedSplinePoint(t);
		vec3 d = p1 - p0;
		// scale distance
		d.x *= S[0];
		d.y *= S[1];
		d.z *= S[2];

		p0 = p1;
		dist += sqrt(d * d);
		t += dt;
	}
	return dist;
}

// Given mm distance down spline, calculate relative distance down spline from PMJ
void CRSpline::get_relative_spline_distance(int MM_i, float* MM,
		int S_i, float* S,
		int* V_i, float** V) {

	if (S_i != 3)
		throw "Volume spacing vector S must be 3 vector, for spacing of X, Y, Z";

	(*V_i) = MM_i;
	(*V) = new float[MM_i];
	//float dist = 0;
	//vec3 p0 = CRSpline::GetInterpolatedSplinePoint(0);
	float dt = 0.0001;
	//float t = dt;
	float temp_t = 0.0;
	float temp_dist;
	for (int n = 0; n < MM_i; n++) {
		temp_dist = get_interpolated_spline_distance(temp_t, S_i, S); 		
		while (temp_dist < MM[n]) {
			temp_dist = get_interpolated_spline_distance(temp_t, S_i, S);
			temp_t += dt;
			//cout<<"target="<<MM[n]<<", temp_t="<<temp_t<<", temp_dist="<<temp_dist<<endl;
			if (temp_t > 1){
				cout<<"Error: requested distance is longer than available spline."<<endl;
				break;
			}
		}
		(*V)[n] = temp_t;
	}
	
}

// given input list of 3D points P, returns a list of t values V, the
// angle A between each point and the spline, and the distance B.
// The function basically finds the projection of a point on the spline.
void CRSpline::find_spline_proj(int T_i, float* T,  int S_i, float* S,
		int P_i, int P_j, float* P,
		int* V_i, float** V, int* A_i, float **A, int* B_i, float **B) {
	if (P_j != 3) {
		throw "P must have size of [N x 3] for N points.";
	}
	(*V_i) = P_i;
	(*V) = new float[P_i];
	(*A_i) = P_i;
	(*A) = new float[P_i];
	(*B_i) = P_i;
	(*B) = new float[P_i];

	vec3 p0 = CRSpline::GetInterpolatedSplinePoint(T[0]);
	for (int p = 0; p < P_i; p++) {
		vec3 pt(P[p * 3 + 0], P[p * 3 + 1], P[p * 3 + 2]);
		float best_t;
		float best_d = 1e10;
		for (int t_i = 0; t_i < T_i; t_i++) {
			float t = T[t_i];
			vec3 sp = CRSpline::GetInterpolatedSplinePoint(t);
			vec3 d = sp - pt;
			// scale distance
			d.x *= S[0];
			d.y *= S[1];
			d.z *= S[2];

			float dist = sqrt(d * d);
			if (dist < best_d) {
				best_d = dist;
				best_t = t;
			}
		}
		// store closest position on spine
		(*V)[p] = best_t;
		vec3 normal = pt - CRSpline::GetInterpolatedSplinePoint(best_t);
		normal.normalize();
		vec3 tangent = CRSpline::GetInterpolatedSplineTangent(best_t);
		(*A)[p] = normal * tangent;
		(*B)[p] = best_d;
	}
}

int CRSpline::GetNumPoints() {
	return vp.size();
}

vec3& CRSpline::GetNthPoint(int n) {
	return vp[n];
}

void CRSpline::get_control_points(int* j, int* k, float** pts) {
	(*j) = vp.size();
	(*k) = 3;
	(*pts) = new float[3 * vp.size()];

	for (int p = 0; p < vp.size(); p++) {
		// Compute local control point
		(*pts)[p * 3 + 0] = vp[p].x;
		(*pts)[p * 3 + 1] = vp[p].y;
		(*pts)[p * 3 + 2] = vp[p].z;
	}
}

inline void swap_f(float& a, float& b) {
	float t = a;
	a = b;
	b = t;
}

// returns the weighted sum of data in position (fx, fy, fz)
inline float sum_w(float fx, float fy, float fz, int i, int j, int k,
		float* data) {
	float sum;
	int x = fx;
	int y = fy;
	int z = fz;
	//cout<<"fx="<<fx<<", fy="<<fy<<", fz="<<fz<<endl;
	//cout<<"x="<<x<<", y="<<y<<", z="<<z<<endl;
	if ((x >= 0) && (y >= 0) && (z >= 0) && (x < (i - 1)) && (y < (j - 1)) //ensures the value is positive and within bounds of the data image
			&& (z < (k - 1))) {
		float dx = fx - x;
		float dy = fy - y;
		float dz = fz - z;
		//cout<<"dx="<<dx<<", dy="<<dy<<", dz="<<dz<<endl;
		sum = data[IND(i, j, k, x, y, z)] * dx * dy * dz
				+ data[IND(i, j, k, x, y, z + 1)] * dx * dy * (1 - dz)
				+ data[IND(i, j, k, x, y + 1, z)] * dx * (1 - dy) * dz
				+ data[IND(i, j, k, x, y + 1, z + 1)] * dx * (1 - dy) * (1 - dz)
				+ data[IND(i, j, k, x + 1, y, z)] * (1 - dx) * dy * dz
				+ data[IND(i, j, k, x + 1, y, z + 1)] * (1 - dx) * dy * (1 - dz)
				+ data[IND(i, j, k, x + 1, y + 1, z)] * (1 - dx) * (1 - dy) * dz
				+ data[IND(i, j, k, x + 1, y + 1, z + 1)] * (1 - dx) * (1 - dy)
						* (1 - dz);
		float weight = dx * dy * dz + dx * dy * (1 - dz) + dx * (1 - dy) * dz + dx * (1 - dy) * (1 - dz) + (1 - dx) * dy * dz + (1 - dx) * (1 - dy) * dz + (1 - dx) * (1 - dy) * (1 - dz); //Adam added this to see 			total weight
		//cout<<"weight="<<weight<<endl;
	} else
		sum = 0;

	return sum;
}

float CRSpline::sum_line(float fx, float fy, float fz, float fxx, float fyy,
		float fzz, int i, int j, int k, float* data) {

	// make sure direction is always positive
	if (fxx < fx)
		swap_f(fx, fxx);
	if (fyy < fy)
		swap_f(fy, fyy);
	if (fzz < fz)
		swap_f(fz, fzz);

	float sum = 0;
	float fdx = (fxx - fx);
	float fdy = (fyy - fy);
	float fdz = (fzz - fz);

	if ((fdx <= 1) && (fdy <= 1) && (fdz <= 1))
		return sum_w(fx, fy, fz, i, j, k, data);

	float mx;
	float my;
	float mz;
	float da;

	if (fdx >= fdy && fdx >= fdz) {
		mx = 1;
		my = fdy / fdx;
		mz = fdz / fdx;
		da = fdx;
	} else if (fdy > fdx && fdy > fdz) { //Height is leading the position.
		mx = fdx / fdy;
		my = 1;
		mz = fdz / fdy;
		da = fdy;
	} else if (fdz > fdx && fdz > fdy) { //Depth is leading the position.
		mx = fdx / fdz;
		my = fdy / fdz;
		mz = 1;
		da = fdz;
	}

	for (int a = 0; a <= da; a++) {
		// sum up values along the line, with linear interpolation
		sum += sum_w(fx, fy, fz, i, j, k, data);

		fx += mx;
		fy += my;
		fz += mz;
	}

	return sum;
}

// sum data along spline. N is the number of points to sample per spline
// section. Every two samples are connected with a straight line. In total
// N*(vp.size()-1) points will be sampled.
float CRSpline::sum_spline(int i, int j, int k, float* data, int N) {
	float sum = 0;
	if (vp.size() < 1)
		return sum;

	vec3 v0, v1;
	int M = N * (vp.size() - 1);
	float delta_t = 1.0 / M;

	v0 = vp[0];
	for (float t = delta_t; t <= 1.0; t += delta_t) {
		v1 = CRSpline::GetInterpolatedSplinePoint(t);
		if (((int) v0.x != (int) v1.x) || ((int) v0.y != (int) v1.y)
				|| ((int) v0.z != (int) v1.z)) {
			sum += CRSpline::sum_line(v0.x, v0.y, v0.z, v1.x, v1.y, v1.z, i, j,
					k, data);
			v0 = v1;
		}
	}

	return sum;
}

// sums along a sphere with radius r in (x, y, z)
inline float sum_sphere(float x, float y, float z, int i, int j, int k, float* data,
		int l, int m, int n, unsigned short* label_ip, float r, int method,
		long int* count) {
	float sum = 0;
	float r2 = r*r;
	long int N;

	for (float ix = -r; ix <= r; ix += 1)
		for (float iy = -r; iy <= r; iy += 1)
			for (float iz = -r; iz <= r; iz += 1) {
				float cur_r2 = (ix * ix + iy * iy + iz * iz);
				if ((cur_r2 <= r2) && (x + ix >= 0) && (y + iy >= 0)
						&& (z + iz >= 0) && (x + ix < i) && (y + iy < j)
						&& (z + iz < k)) {
					float d = 0;
//							float d = data[IND(i, j, k, x + ix, y + iy, z + iz)];
					// count every cell only once
					if (label_ip[IND(i, j, k, x + ix, y + iy, z + iz)] == 0) {
						// sum only in the fringes of the "pipe".
//						d = sum_w(x+ix, y+iy, z+iz, i, j, k, data);
						d = data[IND(i, j, k, x + ix, y + iy, z + iz)];
						sum += d;
						label_ip[IND(i, j, k, x + ix, y + iy, z + iz)] = 1;
						N++;
					}
					if (method > 1) // methods 2, 3
						if ((label_ip[IND(i, j, k, x + ix, y + iy, z + iz)] == 1)) {
							// ignore center of sphere to create a pipe-like structure
							if (cur_r2 < r2 / 4) {
								if (d > 0)
									sum += -d;
								else if (method == 3)
									sum += -2 * d;
								label_ip[IND(i, j, k, x + ix, y + iy, z + iz)] = 2;
								N++;
							}
						}
				} else if (cur_r2 <= r2) {
					// penalize for summing points outside the volume
//					sum += OUT_OF_BOUNDS;
				}
			}
	if (count != NULL)
		(*count) += N;

	return sum;
}

float CRSpline::sum_line_pipe(float fx, float fy, float fz, float fxx,
		float fyy, float fzz, int i, int j, int k, float* data, int l, int m,
		int n, unsigned short* label_ip, float r, int method, long int* count) {

	// make sure direction is always positive
	if (fxx < fx)
		swap_f(fx, fxx);
	if (fyy < fy)
		swap_f(fy, fyy);
	if (fzz < fz)
		swap_f(fz, fzz);

	float sum = 0;
	float fdx = (fxx - fx);
	float fdy = (fyy - fy);
	float fdz = (fzz - fz);

	float mx;
	float my;
	float mz;
	float da;

	if (fdx >= fdy && fdx >= fdz) {
		mx = 1;
		my = fdy / fdx;
		mz = fdz / fdx;
		da = fdx;
	} else if (fdy > fdx && fdy > fdz) { //Height is leading the position.
		mx = fdx / fdy;
		my = 1;
		mz = fdz / fdy;
		da = fdy;
	} else if (fdz > fdx && fdz > fdy) { //Depth is leading the position.
		mx = fdx / fdz;
		my = fdy / fdz;
		mz = 1;
		da = fdz;
	}

	for (int a = 0; a <= da; a++) {
		// sum up values along the line, with linear interpolation
		sum += sum_sphere(fx, fy, fz, i, j, k, data, l, m, n,
				label_ip, r, method, count);

		fx += mx;
		fy += my;
		fz += mz;
	}

	return sum;
}

// sum data along spline with radius r. N is the number of points to sample per spline
// section. Every two samples are connected with a straight line. In total
// N*(vp.size()-1) points will be sampled.
float CRSpline::sum_spline_pipe(int i, int j, int k, float* data, int l, int m,
		int n, unsigned short* label_ip, float r, int N, int method) {
	long int count = 0;
	float sum = 0;
	if (vp.size() < 1)
		return sum;

	vec3 v0, v1;
	int M = N * (vp.size() - 1);
	float delta_t = 1.0 / M;

	// clear label_ip
	memset((void*) label_ip, 0, (long int) l * m * n * sizeof(label_ip[0]));

	v0 = vp[0];
	for (float t = delta_t; t <= 1.0; t += delta_t) {
		v1 = CRSpline::GetInterpolatedSplinePoint(t);
		if (((int) v0.x != (int) v1.x) || ((int) v0.y != (int) v1.y)
				|| ((int) v0.z != (int) v1.z)) {
			sum += CRSpline::sum_line_pipe(v0.x, v0.y, v0.z, v1.x, v1.y, v1.z,
					i, j, k, data, l, m, n, label_ip, r, method, &count);
			v0 = v1;
		}
	}

	// normalize sum by number of voxels to prevent drift of spline outside of volume
//	if (count > 0)
//		sum = sum/count;

	return sum;
}

// draws a sphere with radius r in (x, y, z)
inline void draw_sphere(float x, float y, float z, int i, int j, int k,
		float* data_ip, float value, int r) {
	for (int ix = -r; ix <= r; ix++)
		for (int iy = -r; iy <= r; iy++)
			for (int iz = -r; iz <= r; iz++)
				if (((ix * ix + iy * iy + iz * iz) <= r * r) && (x + ix >= 0)
						&& (y + iy >= 0) && (z + iz >= 0) && (x + ix < i)
						&& (y + iy < j) && (z + iz < k)) {
					data_ip[IND(i, j, k, x + ix, y + iy, z + iz)] = value;
				}
}

void CRSpline::draw_line(int x, int y, int z, int xx, int yy, int zz, int i,
		int j, int k, float* data_ip, float value, int r) {

	int dx = (xx - x);
	int dy = (yy - y);
	int dz = (zz - z);

	//Direction pointer.

	int step_x = 0;
	int step_y = 0;
	int step_z = 0;

	//Moving right step +1 else -1

	if (dx >= 0)
		step_x = 1;
	else {
		step_x = -1;
		dx = -dx;
	}
	if (dy >= 0)
		step_y = 1;
	else {
		step_y = -1;
		dy = -dy;
	}
	if (dz >= 0)
		step_z = 1;
	else {
		step_z = -1;
		dz = -dz;
	}

	int dx2 = dx * 2; //delta X * 2 instead of 0.5
	int dy2 = dy * 2; //delta Y * 2 ..
	int dz2 = dz * 2; //delta Z * 2 ..

	int err_termXY = 0; //Zero it
	int err_termXZ = 0; //Zero it

	//If width is greater than height
	//we are going to adjust the height movment after the x steps.

	if (dx >= dy && dx >= dz) {
		//Set err_term to height*2 and decrement by the segment width.
		//example. 2-10 =-8

		err_termXY = dy2 - dx;
		err_termXZ = dz2 - dx;

		//Step x direction by one until the end of width.

		for (int a = 0; a <= dx; a++) {
			// sum up values along the line, with linear interpolation
			if ((x >= 0) && (y >= 0) && (z >= 0) && (x < i) && (y < j)
					&& (z < k)) {
				draw_sphere(x, y, z, i, j, k, data_ip, value, r);
			}

			//Adjust error_term
			//and step down or up by one in the y path.
			//This if it's time to do so.

			if (err_termXY >= 0) {
				err_termXY -= dx2; //err minus the width*2;

				y += step_y; //Step down or up by one.

			}
			if (err_termXZ >= 0) {
				err_termXZ -= dx2; //err minus the width*2;

				z += step_z; //Step in or out by one.

			}
			err_termXY += dy2; //Add err_term by the height * 2;

			err_termXZ += dz2; //Add err_term by the depth * 2;

			//This will happen all the time.

			x += step_x; //step right or left

		}
	} else if (dy > dx && dy > dz) //Height is leading the position.

			{
		//Set err_term to width*2 and decrement by the delta y.

		err_termXY = dx2 - dy;
		err_termXZ = dz2 - dy;

		//Step y direction by one until the end of height.

		for (int a = 0; a <= dy; a++) {
			// sum up values along the line
			if ((x >= 0) && (y >= 0) && (z >= 0) && (x < i) && (y < j)
					&& (z < k)) {
				draw_sphere(x, y, z, i, j, k, data_ip, value, r);
			}

			//Adjust error_term
			//and step left or right by one in the x path.
			//This if it's time to do so.

			if (err_termXY >= 0) {
				err_termXY -= dy2; //err minus the height*2;

				x += step_x; //Step right or left by one.

			}
			if (err_termXZ >= 0) {
				err_termXZ -= dy2; //err minus the height*2;

				z += step_z; //Step depth in or out by one.

			}
			err_termXY += dx2; //Add err_term by the width * 2;

			err_termXZ += dz2; //Add err_term by the depth * 2;

			//This will happen all the time.

			y += step_y; //step up or down.

		}
	} else if (dz > dx && dz > dy) //Depth is leading the position.

			{
		//Set err_term to width*2 and decrement by the delta z.

		err_termXY = dx2 - dz;
		err_termXZ = dy2 - dz;

		//Step z direction by one until the end of depth.

		for (int a = 0; a <= dz; a++) {
			// sum up values along the line
			if ((x >= 0) && (y >= 0) && (z >= 0) && (x < i) && (y < j)
					&& (z < k)) {
				draw_sphere(x, y, z, i, j, k, data_ip, value, r);
			}

			//Adjust error_term

			//and step up or down by one in the y path.

			//This if it's time to do so.

			if (err_termXY >= 0) {
				err_termXY -= dz2; //err minus the depth*2;

				y += step_y; //Step up or down by one.

			}
			if (err_termXZ >= 0) {
				err_termXZ -= dz2; //err minus the depth*2;

				x += step_x; //Step right or left by one.

			}
			err_termXY += dy2; //Add err_term by the height * 2;

			err_termXZ += dx2; //Add err_term by the width * 2;

			z += step_z; //step in or out.

		}
	}
}

void CRSpline::draw_spline(int i, int j, int k, float* data_ip, float value,
		int r, int N) {
	if (vp.size() < 1)
		return;

	vec3 v0, v1;
	int M = N * (vp.size() - 1);
	float delta_t = 1.0 / M;

	v1 = vp[0];
	for (float t = delta_t; t <= 1.0; t += delta_t) {
		v0 = v1;
		v1 = CRSpline::GetInterpolatedSplinePoint(t);
		CRSpline::draw_line(v0.x, v0.y, v0.z, v1.x, v1.y, v1.z, i, j, k,
				data_ip, value, r);
	}
}

// greedy optimization over the control points (excluding first and last) in
// order to minimnize the summed weight along the spline.
float CRSpline::minimize_pipe(int i, int j, int k, float* data, int l, int m,
		int n, unsigned short* label_ip, float r, int N, int method, int opt_method) {

	vector<vec3> vp0 = vp;
	int S = vp.size();

	float best_sum = 0; //sum_spline_pipe(i, j, k, data, l, m, n, label_ip, r, N);
	float prev_sum = best_sum + 1;
	vector<vec3> best_vp = vp;

	if (opt_method == 1) {
		vp.clear();

		// init spline to have two end points
		vp.push_back(vp0[0]);
		vp.push_back(vp0[S-1]);

		// first optimize point by point - find best initial position
		for (int p = 1; p < S - 1; p++) {
			vec3 next_vc = vp[p-1];
			next_vc.y = vp0[p].y;

			vp.insert(vp.begin()+p, next_vc);

			best_sum = sum_spline_pipe(i, j, k, data, l, m, n, label_ip, r, N);
			prev_sum = best_sum + 1;
			best_vp = vp;

			cout<<"Processing point: "<<p<<"    Best sum: "<<best_sum<<endl;
			while (prev_sum != best_sum) {
				prev_sum = best_sum;
				// try all posible translation per all points (without fist and last)
				for (int x = -1; x <= 1; x++)
					for (int y = -1; y <= 1; y++)
						for (int z = -1; z <= 1; z++) {
							// optimize all points but first and last
							{
								vp = best_vp;
								vp[p].x += x;
								vp[p].y += y;
								vp[p].z += z;
								float cur_sum = sum_spline_pipe(i, j, k, data, l, m, n,
										label_ip, r, N, method);
								if (cur_sum < best_sum) {
									best_vp = vp;
									best_sum = cur_sum;
								}
							}
						}
			}
		}
	}

	best_sum = sum_spline_pipe(i, j, k, data, l, m, n, label_ip, r, N);
	prev_sum = best_sum + 1;
	// next optimize for all points
	while (prev_sum != best_sum) {
		prev_sum = best_sum;
		// try all posible translation per all points (without fist and last)
		for (int x = -1; x <= 1; x++)
//			for (int y = 0; y <= 0; y++)
			for (int y = -1; y <= 1; y++)
				for (int z = -1; z <= 1; z++) {
					// optimize all points but first and last
					for (int p = 1; p < S - 1; p++) {
						vp = best_vp;
						vp[p].x += x;
						vp[p].y += y;
						vp[p].z += z;
						float cur_sum = sum_spline_pipe(i, j, k, data, l, m, n,
								label_ip, r, N, method);
						if (cur_sum < best_sum) {
							best_vp = vp;
							best_sum = cur_sum;
						}
					}
				}
	}

	vp = best_vp;
	return best_sum;
}



float CRSpline::minimize_pipe_adam(int i, int j, int k, float* data, int l, int m,
		int n, unsigned short* label_ip, float r, int N, int method, int opt_method) {

	vector<vec3> vp0 = vp;
	int S = vp.size();

	float best_sum = 0; //sum_spline_pipe(i, j, k, data, l, m, n, label_ip, r, N);
	float prev_sum = best_sum + 1;
	vector<vec3> best_vp = vp;
	vector<vec3> orig_vp = vp;
	
	if (opt_method == 1) {
		vp.clear();

		// init spline to have two end points
		vp.push_back(vp0[0]);
		vp.push_back(vp0[S-1]);

		// first optimize point by point - find best initial position
		for (int p = 1; p < S - 1; p++) {
			vec3 next_vc = vp[p-1];
			next_vc.y = vp0[p].y;

			vp.insert(vp.begin()+p, next_vc);

			best_sum = sum_spline_pipe(i, j, k, data, l, m, n, label_ip, r, N);
			prev_sum = best_sum + 1;
			best_vp = vp;
			orig_vp = vp0;

			//cout<<"Processing point: "<<p<<"    Best sum: "<<best_sum<<" p.x="<<vp[p].x<<" p.y="<<vp[p].y<<" p.z="<<vp[p].z<<endl;
			//cout<<"Processing point: "<<p<<"    Orig_vp: "<<orig_vp[p].x<<", "<<orig_vp[p].y<<", "<<orig_vp[p].z<<endl;

			float orig_x=vp[p].x;			
			float orig_y=vp[p].y;
			float orig_z=vp[p].z;
			float max_x=vp[p].x;
			float min_x=vp[p].x;
			float max_y=vp[p].y;
			float min_y=vp[p].y;
			float max_z=vp[p].z;
			float min_z=vp[p].z;

			while (prev_sum != best_sum) {
				prev_sum = best_sum;
				// try all posible translation per all points (without fist and last)
				for (int x = -1; x <= 1; x++)
					
					for (int y = -1; y <= 1; y++)
						
						for (int z = -1; z <= 1; z++) {
							// optimize all points but first and last
							{
								//vp = best_vp;
								vp = orig_vp;

								vp[p].x += x;
								vp[p].y += y;
								vp[p].z += z;
								float cur_sum = sum_spline_pipe(i, j, k, data, l, m, n,
										label_ip, r, N, method);
								//cout<<"p="<<p<<" x,y,z="<<x<<", "<<y<<", "<<z<<"    Sum: "<<cur_sum<<" p.x="<<vp[p].x<<" p.y="<<vp[p].y<<" p.z="<<vp[p].z<<endl;
								if (cur_sum < best_sum) {
									best_vp = vp;
									best_sum = cur_sum;
								//	cout<<"p="<<p<<" New Best Sum="<<best_sum<<" p.x="<<vp[p].x<<" p.y="<<vp[p].y<<" p.z="<<vp[p].z<<endl;
								}
								if (vp[p].x > max_x) {
									max_x=vp[p].x;
								}
								if (vp[p].x < min_x) {
									min_x=vp[p].x;
								}
								if (vp[p].y > max_y) {
									max_y=vp[p].y;
								}
								if (vp[p].y < min_y) {
									min_y=vp[p].y;
								}
								if (vp[p].z > max_z) {
									max_z=vp[p].z;
								}
								if (vp[p].z < min_z) {
									min_z=vp[p].z;
								}
							}
						
						}
			}
			//cout<<"p="<<p<<"   Orig="<<orig_x<<", "<<orig_y<<", "<<orig_z<<"   Min/Max_X="<<min_x<<"/"<<max_x<<"   Min/Max_Y="<<min_y<<"/"<<max_y<<"   Min/Max_Z="<<min_z<<"/"<<max_z<<endl;
		}
	}

	vp = best_vp;
	return best_sum;
}


inline float sign(float x) {
	if (x > 0)
		return 1.0;
	else if (x == 0)
		return 0.0;
	else
		return -1.0;
}

// Given start and end point, returns the distance from start point to the
// first non-zero element along the line.
float CRSpline::find_line(float fx, float fy, float fz, float fxx, float fyy,
		float fzz, int l, int m, int n, unsigned short* label_ip) {


	float fdx = (fxx - fx);
	float fdy = (fyy - fy);
	float fdz = (fzz - fz);

	float mx;
	float my;
	float mz;
	float da;

	float x0 = fx;
	float y0 = fy;
	float z0 = fz;

	if (abs(fdx) >= abs(fdy) && abs(fdx) >= abs(fdz)) {
		mx = sign(fdx);
		my = fdy / abs(fdx);
		mz = fdz / abs(fdx);
		da = abs(fdx);
	} else if (abs(fdy) > abs(fdx) && abs(fdy) > abs(fdz)) { //Height is leading the position.
		mx = fdx / abs(fdy);
		my = sign(fdy);
		mz = fdz / abs(fdy);
		da = abs(fdy);
	} else if (abs(fdz) > abs(fdx) && abs(fdz) > abs(fdy)) { //Depth is leading the position.
		mx = fdx / abs(fdz);
		my = fdy / abs(fdz);
		mz = sign(fdz);
		da = abs(fdz);
	}

	for (int a = 0; a <= da; a++) {
		// sum up values along the line, with linear interpolation
		if ((fx >= 0) && (fy >= 0) && (fz >= 0) && (fx < l) && (fy < m)
				&& (fz < n)) {
			if (label_ip[IND(l, m, n, fx, fy, fz)] != 0)
				break;
		} else
			break;

		fx += mx;
		fy += my;
		fz += mz;
	}

	float dx = fx - x0;
	float dy = fy - y0;
	float dz = fz - z0;
	return sqrt(dx * dx + dy * dy + dz * dz);
}

// Given values of theta and t, returns an array of the closest distance from
// the spline to a non-zero label.
void CRSpline::extract_spine_cylinder_coord(int i, float* T, int j,
		float* THETA, int l, int m, int n, unsigned short* label_ip, int* a,
		int* b, float **D) {

	(*D) = new float[i * j];
	(*a) = i;
	(*b) = j;

	for (int t_i = 0; t_i < i; t_i++) {
		float t = T[t_i];
		vec3 tangent = CRSpline::GetInterpolatedSplineTangent(t);
		vec3 pt = CRSpline::GetInterpolatedSplinePoint(t);
//		cout<<"pt: "<<pt.x<<" "<<pt.y<<" "<<pt.z<<endl;
//		cout<<"tangent: "<<tangent.x<<" "<<tangent.y<<" "<<tangent.z<<endl;
		// calculate direction for theta == 0
		vec3 normal = tangent.cross(vec3(1, 0, 0));
		if (normal.norm2() == 0)
			normal = tangent.cross(vec3(0, 0, 1));

		normal.normalize();
//		cout<<"normal: "<<normal.x<<" "<<normal.y<<" "<<normal.z<<"   "<<normal*tangent<<endl;

		for (int theta_i = 0; theta_i < j; theta_i++) {
			float theta = THETA[theta_i];

//			cout<<"theta: "<<theta<<endl;
			vec3 n_theta = tangent.rotatev(normal, theta);
			n_theta.normalize();
//			if (n_theta.norm2() < 0.5) {
//				cout<<"AAAAAAAAAAAAAAAA "<<n_theta.norm2()<<endl;
//			}
//			cout<<"n: "<<n_theta.x<<" "<<n_theta.y<<" "<<n_theta.z<<"   "<<n_theta*tangent<<endl;
			vec3 x0 = pt;
			vec3 x1 = pt + n_theta * (l + m + n);
			(*D)[((long int) t_i) * j + theta_i] = CRSpline::find_line(x0.x,
					x0.y, x0.z, x1.x, x1.y, x1.z, l, m, n, label_ip);
		}
	}
}

// Given start and end point, returns the distance from start point to the
// first non-zero element along the line.
float CRSpline::parse_line(float fx, float fy, float fz, float fxx, float fyy,
		float fzz, int i, int j, int k, float* data, int method) {

	float fdx = (fxx - fx);
	float fdy = (fyy - fy);
	float fdz = (fzz - fz);

	float da = sqrt(fdx*fdx+fdy*fdy+fdz*fdz);

	if (da == 0)
		return 0;

	float mx = fdx/da;
	float my = fdy/da;
	float mz = fdz/da;

	float x0 = fx;
	//cout<<"fx="<<fx<<", x0="<<x0<<endl;
	float y0 = fy;
	float z0 = fz;
	
	float value = 0;
	for (int a = 0; a <= da; a++) {
		// sum up values along the line, with linear interpolation
		//cout<<"a="<<a<<", da="<<da<<endl;		
		if ((fx >= 0) && (fy >= 0) && (fz >= 0) && (fx < i) && (fy < j)
				&& (fz < k)) {
			float cur_d = sum_w(fx, fy, fz, i, j, k, data);
			//cout<<"fx="<<fx<<", x0="<<x0<<endl;
			if (method == 1) { // sum values
				value += cur_d;
			}
			if (method == 2) { // first moment
				float r2 = (fx - x0) * (fx - x0) + (fy - y0) * (fy - y0)
						+ (fz - z0) * (fz - z0);
				float r = sqrt(r2);
				value += cur_d * (r + 1);
			}
			if (method == 3) { // first moment of 1/r
				float r2 = (fx - x0) * (fx - x0) + (fy - y0) * (fy - y0)
						+ (fz - z0) * (fz - z0);
				float r = sqrt(r2);
				value += cur_d / (r + 1);
			}
			if (method == 4) { // first moment of 1/r^2
				float r2 = (fx - x0) * (fx - x0) + (fy - y0) * (fy - y0)
						+ (fz - z0) * (fz - z0);
				float r = sqrt(r2);
				value += cur_d / (r2 + 1);
			}
		}
		fx += mx;
		fy += my;
		fz += mz;
	}
	return value;
}

// Given values of theta and t, and a data returns an array parse data.
void CRSpline::parse_spine_cylinder_coord(int T_i, float* T, int THETA_i,
		float* THETA, int i, int j, int k, float* data, float min_r,
		float max_r, int method, int* a, int* b, float **D) {

	(*D) = new float[T_i * THETA_i];
	(*a) = T_i;
	(*b) = THETA_i;

	for (int t_i = 0; t_i < T_i; t_i++) {
		float t = T[t_i];
		vec3 tangent = CRSpline::GetInterpolatedSplineTangent(t);
		vec3 pt = CRSpline::GetInterpolatedSplinePoint(t);
		// calculate direction for theta == 0
		//vec3 normal = tangent.cross(vec3(1, 0, 0)); //the functional images use this line since the left/right position is the first array location
		vec3 normal = tangent.cross(vec3(0, 0, -1)); //the anatomical images use this line since the left/right position is the first array location
		if (normal.norm2() == 0)
			normal = tangent.cross(vec3(0, 0, 1));

		normal.normalize();
		//cout<<"Point="<<pt.x<<", "<<pt.y<<", "<<pt.z<<"  Normal="<<normal.x<<", "<<normal.y<<", "<<normal.z<<endl;

		for (int theta_i = 0; theta_i < THETA_i; theta_i++) {
			float theta = THETA[theta_i];

			vec3 n_theta = tangent.rotatev(normal, theta);
			n_theta.normalize();
			//cout<<"Interpolated Point:"<<pt.x<<", "<<pt.y<<", "<<pt.z<<endl;
			vec3 x0 = pt + n_theta * min_r;
			//cout<<"x0:"<<x0.x<<", "<<x0.y<<", "<<x0.z<<endl;
			vec3 x1 = pt + n_theta * max_r;
			//cout<<"x1:"<<x1.x<<", "<<x1.y<<", "<<x1.z<<endl;
			(*D)[((long int) t_i) * THETA_i + theta_i] = CRSpline::parse_line(
					x0.x, x0.y, x0.z, x1.x, x1.y, x1.z, i, j, k, data, method);
		}
	}
}


void CRSpline::parse_spine_cylinder_coord_adam(int T_i, float* T, int THETA_i,
		float* THETA, int i, int j, int k, float* data, float min_r,
		float max_r, int method, int* a, int* b, float **D) {

	(*D) = new float[T_i * THETA_i];
	(*a) = T_i;
	(*b) = THETA_i;

	for (int t_i = 0; t_i < T_i; t_i++) {
		float t = T[t_i];
		vec3 tangent = CRSpline::GetInterpolatedSplineTangent(t);
		vec3 pt = CRSpline::GetInterpolatedSplinePoint(t);
		// calculate direction for theta == 0
		vec3 normal = tangent.cross(vec3(0, 0, -1)); //this sets the normal to be in the ventral/dorsal direction for anatomical images (front = ventral, back = dorsal)			
		normal.normalize();

		for (int theta_i = 0; theta_i < THETA_i; theta_i++) {
			float theta = THETA[theta_i];

			vec3 n_theta = tangent.rotatev(normal, theta);
			n_theta.normalize();
			//cout<<"Interpolated Point:"<<pt.x<<", "<<pt.y<<", "<<pt.z<<endl;
			vec3 x0 = pt + n_theta * min_r;
			//cout<<"x0:"<<x0.x<<", "<<x0.y<<", "<<x0.z<<endl;
			vec3 x1 = pt + n_theta * max_r;
			//cout<<"x1:"<<x1.x<<", "<<x1.y<<", "<<x1.z<<endl;
			(*D)[((long int) t_i) * THETA_i + theta_i] = CRSpline::parse_line(
					x0.x, x0.y, x0.z, x1.x, x1.y, x1.z, i, j, k, data, method);
		}
	}
}

//returns value of the normal vector at a given point to python
void CRSpline::get_spline_normal(int T_i, float* T, int* a, float **Output) {

	(*Output) = new float[3];	
	float t = T[0];
	cout<<"t="<<t<<endl;
	vec3 tangent = CRSpline::GetInterpolatedSplineTangent(t);
	vec3 pt = CRSpline::GetInterpolatedSplinePoint(t);
	// calculate direction for theta == 0
	vec3 normal = tangent.cross(vec3(0, 0, -1)); //this sets the normal to be in the ventral/dorsal direction for anatomical images (front = ventral, back = dorsal)
	normal.normalize();
	(*Output)[0] = normal.x;
	(*Output)[1] = normal.y;
	(*Output)[2] = normal.z;
	
}


//Given start and end point, searches line and identifies the zero or negative elements on the line, or finds change in pixel intensity along line, depending on method input
float CRSpline::parse_line_zero_holes(float fx, float fy, float fz, float fxx, float fyy,
		float fzz, int i, int j, int k, float* data, int l, int m, int n, 
		unsigned short* label_ip, float search_r, int method, float min_value) {

	float fdx = (fxx - fx);
	float fdy = (fyy - fy);
	float fdz = (fzz - fz);
	float fx0 = fx;
	float fy0 = fy;
	float fz0 = fz;

	float da = sqrt(fdx*fdx+fdy*fdy+fdz*fdz);

	if (da == 0)
		return 0;

	float max_radius;
	float cur_radius;

	int counter = 0;	
	float cur_gradient;
	float sum_gradient;

	float mx = fdx/da;
	float my = fdy/da;
	float mz = fdz/da;

	float x0 = fx;
	float y0 = fy;
	float z0 = fz;
	
	float value = 0;
	
	int x = fx; //need to initialize these here to do the last_value calculation before the for loop
	int y = fy; 
	int z = fz; 
	float last_value = data[IND(i, j, k, x, y, z)];
	float start_value;
	int first_run=1;

	for (int a = 0; a <= da; a++) {
		// checks for the first zero point along the line
		//cout<<"a="<<a<<", da="<<da<<endl;		
		//cout<<"beginning of for loop last_value="<<last_value<<", method ="<<method<<endl;
		if ((fx >= 0) && (fy >= 0) && (fz >= 0) && (fx < i) && (fy < j) && (fz < k)) {
			int x = fx; //rounds down
			int y = fy; 
			int z = fz; 
			cur_radius = sqrt((fx-fx0)*(fx-fx0)+(fy-fy0)*(fy-fy0)+(fz-fz0)*(fz-fz0));
			if (first_run==1){
				start_value = data[IND(i, j, k, x, y, z)];				
				first_run = 0;
			}
			
			if (method == 1){			
				if (data[IND(i, j, k, x, y, z)] > min_value) {
					label_ip[IND(i, j, k, x, y, z)] = 1;
					max_radius = cur_radius;
					//cout<<"x0="<<x0<<", y0="<<y0<<", z0="<<z0<<", x="<<x<<", y="<<y<<", z="<<z<<", data value="<<data[IND(i, j, k, x, y, z)]<<", max_radius="<<max_radius<<", IND(i, j, k, x, y, z)="<<IND(i, j, k, x, y, z)<<endl;
				}
				else {
					break;
				}
			}
			if ((method == 2) && (cur_radius <= search_r * 1.1) && (cur_radius >= search_r * 0.9)){
				if (data[IND(i, j, k, x, y, z)] > min_value) {
					//cur_gradient = (data[IND(i, j, k, x, y, z)] / last_value) - 1;
					cur_gradient = (data[IND(i, j, k, x, y, z)] / start_value);
					if (cur_gradient >= 1){
						cout<<"a="<<a<<", start value="<<start_value<<", cur_val="<<data[IND(i, j, k, x, y, z)]<<", cur_gradient="<<cur_gradient<<", cur_radius="<<cur_radius<<", x0="<<x0<<", y0="<<y0<<", z0="<<z0<<", x="<<x<<", y="<<y<<", z="<<z<<endl;
					}											
					last_value = data[IND(i, j, k, x, y, z)];
					sum_gradient += cur_gradient;
					counter++;
				}		
				else {
					break;
				}	
			}
		}
		fx += mx;
		fy += my;
		fz += mz;
	}
	sum_gradient = sum_gradient / counter;
	if (method == 1){	
		return max_radius;
	}
	if (method == 2){
		cout<<"Done Line"<<endl;		
		//cout<<"counter="<<counter<<", gradient="<<sum_gradient<<endl;
		//return sum_gradient;
		return cur_gradient;
	}
}

// similar to parse_spine_cylinder_coord, but searches for the point where the pixel value is 0 or less than one to find the boundaries of a signed distance map
void CRSpline::parse_spine_cylinder_coord_find_radius(int T_i, float* T, int THETA_i,
		float* THETA, int i, int j, int k, float* data, int l, int m, int n, 
		unsigned short* label_ip, float max_r, float search_r, int holes_i, 
		float* holes, int method, float min_value, int* a, int* b, float **D) {

	(*D) = new float[T_i * THETA_i];
	(*a) = T_i;
	(*b) = THETA_i;
	double PI = 3.141592;
	int found_hole = 0;
	int estimate_range = T_i / 100;
	for (int t_i = 0; t_i < T_i; t_i++) {
		float t = T[t_i];		
		vec3 tangent = CRSpline::GetInterpolatedSplineTangent(t);
		vec3 pt = CRSpline::GetInterpolatedSplinePoint(t);
		for (int z = 0; z < holes_i; z++){
			if (holes[z] == floor(pt.y)){
				found_hole = 1;
				cout<<"t="<<t_i<<", FOUND HOLE!"<<endl;
			}					
		}
		// calculate direction for theta == 0	
		vec3 normal = tangent.cross(vec3(0, 0, -1)); //this sets the normal to be in the ventral/dorsal direction for anatomical images (front = ventral, back = dorsal)

		normal.normalize();

		for (int theta_i = 0; theta_i < THETA_i; theta_i++) {
			float theta = THETA[theta_i];
			vec3 n_theta = tangent.rotatev(normal, theta);
			n_theta.normalize();
			//cout<<"Interpolated Point:"<<pt.x<<", "<<pt.y<<", "<<pt.z<<endl;
			vec3 x0 = pt;
			//cout<<"x0:"<<x0.x<<", "<<x0.y<<", "<<x0.z<<endl;
			vec3 x1 = pt + n_theta * max_r;
			//cout<<"x1:"<<x1.x<<", "<<x1.y<<", "<<x1.z<<endl;
			
			
			//cout<<"Theta="<<(theta * THETA_i)<<endl;
			if (found_hole == 1 && (theta_i > 90) && (theta_i < 270)) {
				float est_radius_sum = 0;
				//cout<<"t="<<t_i<<", theta="<<theta_i<<", processing hole"<<endl;				
				for (int b = 1; b < (estimate_range + 1); b++){
					if (theta_i > (179 + estimate_range)){					
						cout<<"radius_lookup="<<(*D)[((long int) (t_i - b)) * THETA_i + theta_i]<<endl;						
						est_radius_sum += (*D)[((long int) (t_i - b)) * THETA_i + theta_i] + (*D)[((long int) (t_i - b)) * THETA_i + theta_i - 180];
					}
					else {
						cout<<"radius_lookup="<<(*D)[((long int) (t_i - b)) * THETA_i + theta_i]<<endl;
						est_radius_sum += (*D)[((long int) (t_i - b)) * THETA_i + theta_i] + (*D)[((long int) (t_i - b)) * THETA_i + theta_i + 180]; 	
					}
				}
				float new_radius = (est_radius_sum / estimate_range) / 2;
				cout<<"New Radius Calculated="<<new_radius<<endl;
				(*D)[((long int) t_i) * THETA_i + theta_i] = new_radius;
				vec3 new_point = pt + n_theta * new_radius;
				label_ip[IND(i, j, k, new_point.x, new_point.y, new_point.z)] = 1;
			}
			else{
				//cout<<"t="<<t_i<<", theta="<<theta_i<<", normal"<<endl;				
				(*D)[((long int) t_i) * THETA_i + theta_i] = CRSpline::parse_line_zero_holes(
						x0.x, x0.y, x0.z, x1.x, x1.y, x1.z, i, j, k, data, l, m, n, 
						label_ip, search_r, method, min_value);
			}
		}
	}
}



//Returns a set of points that are 10 unit vectors away from the spline center in the normal direcion (set to the ventral direction)
void CRSpline::get_interpolated_spline_normal(int i, float* T, float theta, int* j, int* k,
		float** pts) {
	(*j) = i;
	(*k) = 3;
	(*pts) = new float[3 * i];	
	for (int p = 0; p < i; p++) {
		// Compute local control point
		vec3 pt = CRSpline::GetInterpolatedSplinePoint(T[p]);		
		vec3 v = CRSpline::GetInterpolatedSplineTangent(T[p]);				
		vec3 normal = v.cross(vec3(0, 0, -1)); //this sets the normal to be in the ventral/dorsal direction for anatomical images (front = ventral, back = dorsal)

		normal.normalize();

	vec3 n_theta = v.rotatev(normal, theta);
	n_theta.normalize();
	vec3 x1 = pt + n_theta * 10;		
	
		(*pts)[p * 3 + 0] = x1.x;
		(*pts)[p * 3 + 1] = x1.y;
		(*pts)[p * 3 + 2] = x1.z;
	}

		
}


//Determines a new set of control points by finding the center of the line made of the two radial lines heading from the point 180 degrees apart
void CRSpline::get_radius_control_point(int T_i, float* T, int a, int b, float* grid, int segments, int* j, int* k, float **pts) { 
	(*j) = T_i;
	(*k) = 3;
	(*pts) = new float[3 * T_i];
	double PI = 3.141592;

	cout<<"STarted"<<endl;	
	for (int p = 0; p < T_i; p++) {
		vec3 average = vec3(0, 0, 0);		
		vec3 pt = CRSpline::GetInterpolatedSplinePoint(T[p]);
		vec3 v = CRSpline::GetInterpolatedSplineTangent(T[p]);			
		vec3 normal = v.cross(vec3(0, 0, -1)); //this sets the normal to be in the ventral/dorsal direction for anatomical images (front = ventral, back = dorsal)

		normal.normalize();
		
		for (int z = 0; z < segments; z++){
			int theta = (180 / segments) * z;		
			vec3 n_theta = v.rotatev(normal, theta * (2 * PI / 360)); 
			n_theta.normalize();
			float distance = grid[p * b + theta] - (grid[p * b + theta] + grid[p * b + theta + 180]) / 2;		
			vec3 pt2 = pt + n_theta * distance; 
			if (average.x == 0 && average.y == 0 && average.z == 0){
				average = pt2;
			}
			else{
				average.x = (average.x + pt2.x) / 2;
				average.y = (average.y + pt2.y) / 2;
				average.z = (average.z + pt2.z) / 2;
			}
			//cout<<"Theta="<<theta<<", distance="<<int(distance)<<", pt.x="<<pt.x<<", pt.y="<<pt.y<<", pt.z="<<pt.z<<", pt2.x="<<pt2.x<<", pt2.y="<<pt2.y<<", pt2.z="<<pt2.z<<endl;
		}
		cout<<"Old Point: x="<<pt.x<<", y="<<pt.y<<", z="<<pt.z<<"  Change: x="<<int(average.x-pt.x)<<", y="<<int(average.y-pt.y)<<", z="<<int(average.z-pt.z)<<endl;	
		(*pts)[p * 3 + 0] = average.x;
		(*pts)[p * 3 + 1] = average.y;
		(*pts)[p * 3 + 2] = average.z;
	}
}	


//Determines a new set of control points that lie on the spine radius at a given angle from the normal
void CRSpline::get_new_angular_control_point(int T_i, float* T, int angle, int a, int b, float* grid, int holes_i, float* holes, int holes_d_i, int holes_d_j, float* holes_d, 
		int l, int m, int n, unsigned short* label_ip, int* j, int* k, float **pts) { //   
	(*j) = T_i;
	(*k) = 3;
	(*pts) = new float[3 * T_i];
	double PI = 3.141592;
	int found_hole = 0;
	vec3 pt2;
	//cout<<"Holes_d shape="<<holes_d_i<<" x "<<holes_d_j<<endl;
	for (int p = 0; p < T_i; p++) {
		//vec3 pt2 = vec3(1, 1, 1);		
		vec3 pt = CRSpline::GetInterpolatedSplinePoint(T[p]);

		for (int z = 0; z < holes_i; z++){
			if (holes[z] == floor(pt.y)){
				found_hole = 1;
				//cout<<"t="<<p<<", FOUND HOLE!"<<endl;
			}					
		}
		if (found_hole == 1){			
			
			(*pts)[p * 3 + 0] = (*pts)[(p - 1) * 3 + 0] + holes_d[p * 3 + 0];
			(*pts)[p * 3 + 1] = (*pts)[(p - 1) * 3 + 1] + holes_d[p * 3 + 1];
			(*pts)[p * 3 + 2] = (*pts)[(p - 1) * 3 + 2] + holes_d[p * 3 + 2];
			vec3 pt2 = vec3((*pts)[p * 3 + 0], (*pts)[p * 3 + 1], (*pts)[p * 3 + 2]);
			//cout<<"p="<<p<<", pt2.x="<<pt2.x<<", pt2.y="<<pt2.y<<", pt2.z="<<pt2.z<<endl;	
			CRSpline::fill_spine_label(pt.x, pt.y, pt.z, pt2.x, pt2.y, pt2.z, l, m, n, 1, label_ip);		
		}
		else{
			vec3 v = CRSpline::GetInterpolatedSplineTangent(T[p]);		
			vec3 normal = v.cross(vec3(0, 0, -1)); //this sets the normal to be in the ventral/dorsal direction for anatomical images (front = ventral, back = dorsal)
			normal.normalize();
		
			vec3 n_theta = v.rotatev(normal, angle * (2 * PI / 360)); 
			n_theta.normalize();
			float radius = grid[p * b + angle];		
			vec3 pt2 = pt + n_theta * radius; 
			//cout<<"p="<<p<<", radius="<<int(radius)<<", x="<<pt2.x<<", y="<<pt2.y<<", z="<<pt2.z<<endl;
			//float radius_check = sqrt((pt2.x - pt.x) * (pt2.x - pt.x) + (pt2.y - pt.y) * (pt2.y - pt.y) + (pt2.z - pt.z) * (pt2.z - pt.z));
			//if ((radius - radius_check) > 0.001){			
			//	cout<<"Grid Radius="<<radius<<", calced radius="<<radius_check<<endl;
			//}
			(*pts)[p * 3 + 0] = pt2.x;
			(*pts)[p * 3 + 1] = pt2.y;
			(*pts)[p * 3 + 2] = pt2.z;
			CRSpline::fill_spine_label(pt.x, pt.y, pt.z, pt2.x, pt2.y, pt2.z, l, m, n, 1, label_ip);
		}	
	}
}	

// Allows the user to input a set of points and the line between the two is filled
void CRSpline::fill_spine_label(float fx, float fy, float fz, float fxx, float fyy, float fzz, int l, int m, int n, int output_value,
		unsigned short* label_ip) {

	//cout<<"Point 2 ="<<fxx<<", "<<fyy<<", "<<fzz<<", point 1="<<fx<<", "<<fy<<", "<<fz<<endl; 	
	float fdx = (fxx - fx);
	float fdy = (fyy - fy);
	float fdz = (fzz - fz);
	float fx0 = fx;
	float fy0 = fy;
	float fz0 = fz;
	
	float da = sqrt(fdx*fdx+fdy*fdy+fdz*fdz);

	//if (da == 0)
		//cout<<"ERROR: da=0"<<endl;

	float mx = fdx/da;
	float my = fdy/da;
	float mz = fdz/da;

	float x0 = fx;
	float y0 = fy;
	float z0 = fz;

	for (int a = 0; a <= da; a++) {
		if ((fx >= 0) && (fy >= 0) && (fz >= 0) && (fx < l) && (fy < m) && (fz < n)) {
			int x = fx; //rounds down
			int y = fy; 
			int z = fz; 
			//cout<<"Filling at x="<<x<<", y="<<y<<", z="<<z<<endl;			
			label_ip[IND(l, m, n, x, y, z)] = output_value;	
		}
		fx += mx;
		fy += my;
		fz += mz;
	}
}


//Calculate Gradient of an Image
void CRSpline::gradient_calc(int i, int j, int k, float* data, int r, int s,  int t, unsigned short* output_label){

	//define the pixel and image types	
	cout<<"Starting Definitions..."<<endl;	
	typedef float PixelType;
	typedef itk::Image<PixelType, 3> ImageType;
	typedef itk::CovariantVector< double, 3 > GradientPixelType;
	typedef itk::Image< GradientPixelType, 3 > GradientImageType;
	typedef itk::GradientRecursiveGaussianImageFilter<ImageType, GradientImageType> GradientFilterType;
		
	//Initialize a new image which will read the input label	
	cout<<"Starting Initialization..."<<endl;	
	ImageType::Pointer image = ImageType::New();
	ImageType::IndexType start;
	start[0] = 0;
	start[1] = 0; 
	start[2] = 0;
	ImageType::SizeType size;
	size[0] = k;
	size[1] = j;
	size[2] = i;
	ImageType::RegionType region;
	region.SetSize( size );
	region.SetIndex( start );
	image->SetRegions( region );
	image->Allocate();
	cout<<"Initialization complete..."<<endl;	
	
	//Create an image iterator to copy the input label into the image file
	typedef itk::ImageRegionIterator< ImageType > IteratorType;
	IteratorType it( image, image->GetRequestedRegion() );
	
	cout<<"Setting Iterator..."<<endl;
	int count = 0;	
	while(!it.IsAtEnd()){	
		it.Set( data[ count ] );
		++it;
		count++;
	}
	
	cout<<"Calculating Gradient..."<<endl;	
	//Calculate the vector gradient of the image
	GradientFilterType::Pointer gradientMapFilter = GradientFilterType::New();
	gradientMapFilter->SetInput( image );
	gradientMapFilter->SetSigma( 1.0 );
	gradientMapFilter->Update();

	cout<<"Gradient Complete..."<<endl;	
	//Creates a new image and iterator of the gradient
	GradientImageType::Pointer image2 = gradientMapFilter->GetOutput();
	typedef itk::ImageRegionConstIterator< GradientImageType > IteratorType2;
	IteratorType2 it2( image2, image2->GetRequestedRegion() );
	ImageType::IndexType idx = it2.GetIndex();

	//Outputs the input image to test if the input worked correctly

	cout<<"Starting Output Iterator..."<<endl;
	long int count2 = 0;	
	while(!it2.IsAtEnd()){	
		float magnitude = sqrt(it2.Get()[0] * it2.Get()[0] + it2.Get()[1] * it2.Get()[1] + it2.Get()[2] * it2.Get()[2]);
		int index_x = it2.GetIndex()[2];
		int index_y = it2.GetIndex()[1];		
		int index_z = it2.GetIndex()[0];
		long int index = (index_x) * s * t + index_y * s + index_z;		
		output_label[ index ] = magnitude ;
		count2++;
		++it2;

	}
	
}


// given a single input point P, returns a list of t values V, the
// angle A between each point and the spline, and the distance B.

void CRSpline::find_spline_proj_point(int T_i, float* T,  int S_i, float* S,
		int P_i, int P_j,  float* P, int* V_i, float** V, int* A_i, float **A) {

	(*V_i) = P_i;
	(*V) = new float[P_i];
	(*A_i) = P_i;
	(*A) = new float[P_i];
	double PI = 3.141592;

	vec3 p0 = CRSpline::GetInterpolatedSplinePoint(T[0]);
	for (int p = 0; p < P_i; p++) {
		vec3 pt(P[p * 3 + 0], P[p * 3 + 1], P[p * 3 + 2]);
		float best_t;
		float best_d = 1e10;
		for (int t_i = 0; t_i < T_i; t_i++) {
			float t = T[t_i];
			vec3 sp = CRSpline::GetInterpolatedSplinePoint(t);
			vec3 d = sp - pt;
			// scale distance
			d.x *= S[0];
			d.y *= S[1];
			d.z *= S[2];
			float dist = sqrt(d * d);
			if (dist < best_d) {
				best_d = dist;
				best_t = t;
			}
		}
		// store closest position on spine
		(*V)[p] = best_t;
		vec3 spline_point = CRSpline::GetInterpolatedSplinePoint(best_t);	
		vec3 normal_pt = pt - spline_point;
		normal_pt.normalize();
		vec3 v = CRSpline::GetInterpolatedSplineTangent(best_t);
		vec3 normal = v.cross(vec3(0, 0, -1)); //this sets the normal to be in the ventral/dorsal direction for anatomical images (front = ventral, back = dorsal)
		normal.normalize(); 
		//cout<<"Normal from point: "<<(pt.x + normal.x * 10)<<" "<<(pt.y + normal.y * 10)<<" "<<(pt.z + normal.z * 10)<<endl;
		//cout<<"Nearest spline point: "<<spline_point.x<<" "<<spline_point.y<<" "<<spline_point.z<<endl;
		//cout<<"Point to spline: "<<(pt.x + normal_pt.x * 10)<<" "<<(pt.y + normal_pt.y * 10)<<" "<<(pt.z + normal_pt.z * 10)<<endl;		
		float cos_angle = (normal_pt * normal) / (normal.norm() * normal_pt.norm());
		float angle = acos(cos_angle) * 360 / (2 * PI);		
		if (pt.z < spline_point.z){
			angle = 360 - angle;
		}
		(*A)[p] = angle * (2 * PI) / 360; //Angle in radians

	}
}


void CRSpline::create_overlay(int T_i, float* T,  int THETA_i, float* THETA, int radial_grid_i, float* radial_grid, int S_i, float* S, float threshold,
		int i, int j,  int k, float* data, 
		int d, int e,  int f, float* out_data) { 

	float rad_incr = .1;
	float rad_cur;
	float t_start = 184;
	float t_end = 204;
	float angle_start = 0;
	float angle_end = 90;
	//cout<<"Output Image Dimensions (d/e/f)="<<d<<"/"<<e<<"/"<<f<<endl;
	//cout<<"Data Dimensions (i/j/k)="<<i<<"/"<<j<<"/"<<k<<endl;
	
	for (int t_i = 0; t_i < T_i; t_i++){		
		int within_region = 0;						
		if ((t_i >= t_start) && (t_i <= t_end)){ 
			within_region = 1;
			//cout<<"t_i = "<<t_i<<endl;
			//cout<<"Within_region. T[t_i]="<<T[t_i]<<", T[t_i + 1]="<<T[t_i + 1]<<endl;
		}
		if (within_region == 1){
			vec3 pt = CRSpline::GetInterpolatedSplinePoint(T[t_i]);
			vec3 v = CRSpline::GetInterpolatedSplineTangent(T[t_i]);			
			vec3 normal = v.cross(vec3(0, 0, -1)); //this sets the normal to be in the ventral/dorsal direction for anatomical images (front = ventral, back = dorsal)
			normal.normalize(); 
			//cout<<"T="<<T[t_i]<<"pt="<<pt.x<<" "<<pt.y<<" "<<pt.z<<endl;

			for (int theta_i = 0; theta_i < THETA_i; theta_i++) {
				int within_region_2 = 0;
				float cur_angle;
				cur_angle = theta_i * 360.0 / THETA_i;						
				if ((cur_angle >= angle_start) && (cur_angle <= angle_end)){ 
					within_region_2 = 1;
					//cout<<"t_i = "<<t_i<<", theta_i = "<<theta_i<<", cur_angle = "<<cur_angle<<endl;
				}
				if (within_region_2 == 1){
					vec3 n_theta = v.rotatev(normal, THETA[theta_i]); 
					n_theta.normalize();
					rad_cur = 0;
					float dist = 0;
					for (int r_i = 0; r_i < radial_grid_i; r_i++){						
						while (dist < radial_grid[r_i]) {
							vec3 pt2 = pt + n_theta * rad_cur;
							vec3 d = pt2 - pt;
							d.x *= S[0];
							d.y *= S[1];
							d.z *= S[2];
							dist = sqrt(d * d);
							rad_cur += rad_incr;
						}
						vec3 pt2 = pt + n_theta * rad_cur;				
						int index_x = pt2.x;
						int index_y = pt2.y;
						int index_z = pt2.z;
						long int index = index_x * e * f + index_y * f + index_z;
						if (data[ t_i * j * k + theta_i * k + r_i ] >= threshold) {
							//cout<<"Greater than threshold."<<endl;
							out_data[ index ] = 1; //data[ t_i * j * k + theta_i * k + r_i ];
						}
						else if (data[ t_i * j * k + theta_i * k + r_i ] <= -threshold) {
							//cout<<"Less than threshold."<<endl;
							out_data[ index ] = 2; //data[ t_i * j * k + theta_i * k + r_i ];
						}
					}
				}	
			}
		}
	}
	
}


// Given one location along the spline (P), calculates the area of normal within the spine given the number of radials and distance to the edge as inputs (both part of Grid). S is the volume spacing.
float CRSpline::get_slice_area(int T_i, float* T,  int THETA_i, float* THETA, int P_i, float* P, 
		int a, int b, float* grid, int S_i, float* S) {

	float area = 0;
	int t_i = 0;
	cout<<"Size of proj_t is "<<P_i<<", point is at t="<<P[1]<<endl;
	
	while ((area == 0) && (t_i < T_i)){
		int within_region = 0;						
		if ((T[t_i] <= P[1]) && (T[t_i + 1] >= P[1])){ 
			within_region = 1;
			cout<<"t_i = "<<t_i<<endl;
			cout<<"Within_region. T[t_i]="<<T[t_i]<<", T[t_i + 1]="<<T[t_i + 1]<<endl;
		}
		if (within_region == 1){
			vec3 pt = CRSpline::GetInterpolatedSplinePoint(P[1]);
			vec3 v = CRSpline::GetInterpolatedSplineTangent(P[1]);			
			vec3 normal = v.cross(vec3(0, 0, -1)); //this sets the normal to be in the ventral/dorsal direction for anatomical images (front = ventral, back = dorsal)
			normal.normalize(); 
			
			for (int theta_i = 0; theta_i < (THETA_i - 1); theta_i++) { //THETA_i - 1 since I look forward one value in the body of the loop
				vec3 n_theta_1 = v.rotatev(normal, THETA[theta_i]); 
				n_theta_1.normalize();
				vec3 n_theta_2 = v.rotatev(normal, THETA[theta_i + 1]); 
				n_theta_2.normalize();
				//cout<<"Theta 1 = "<<THETA[theta_i]<<", Theta 2 = "<<THETA[theta_i + 1]<<endl;
				//since the desired point may not be one of the exact grid points, the radius is the average of the radius of the point on either side of the desired point
				float radius_1 = ( grid[t_i * b + theta_i] + grid[(t_i + 1) * b + theta_i]) / 2; 
				float radius_2 = ( grid[t_i * b + (theta_i + 1)] + grid[(t_i + 1) * b + (theta_i + 1)]) / 2; 
				vec3 radial_1 = n_theta_1 * radius_1;
				vec3 radial_2 = n_theta_2 * radius_2;
				// scale distance
				radial_1.x *= S[0];
				radial_1.y *= S[1];
				radial_1.z *= S[2];				
				radial_2.x *= S[0];
				radial_2.y *= S[1];
				radial_2.z *= S[2];
				//cout<<"radial_1.norm()="<<radial_1.norm()<<", radial_2.norm()="<<radial_2.norm()<<endl;
				vec3 cross_prod = radial_1.cross(radial_2);
				float new_area = 0.5 * cross_prod.norm();
				area += new_area;
				//cout<<"Radius 1 = "<<radius_1<<", radius 2 = "<<radius_2<<", new_area = "<<new_area<<endl;				
			}
		}
		t_i++;
	}	
	
	return area;
}

void CRSpline::load_functional_images(int i, int j, int k, float* data, int current_volume, int slices_per_vol, int offset, int total_volumes, int r, int s,  int t, unsigned short* output_label){

	int last_index;
	//define the pixel and image types	
	typedef float PixelType;
	typedef itk::Image<PixelType, 3> ImageType;
		
	//Initialize a new image which will read the input label	
	ImageType::Pointer image = ImageType::New();
	ImageType::IndexType start;
	start[0] = 0;
	start[1] = 0; 
	start[2] = 0;
	ImageType::SizeType size;
	size[0] = k;
	size[1] = j;
	size[2] = i;
	//cout<<"Image Size="<<k<<", "<<j<<", "<<i<<endl;
	ImageType::RegionType region;
	region.SetSize( size );
	region.SetIndex( start );
	image->SetRegions( region );
	image->Allocate();

	typedef itk::ImageRegionIterator< ImageType > IteratorType;
	IteratorType it( image, image->GetRequestedRegion() );

	int count = 0;	
	while(!it.IsAtEnd()){	
		it.Set( data[ count ] );
		++it;
		count++;
	}
	
	//Creates a new image and iterator of the gradient
	typedef itk::ImageRegionConstIterator< ImageType > IteratorType2;
	IteratorType2 it2( image, image->GetRequestedRegion() );
	ImageType::IndexType idx = it2.GetIndex();

	
	//Outputs the input image
	for (int count_i = 0; count_i < slices_per_vol; count_i++) { 	
		if (count_i == (slices_per_vol - 1)){		
			idx[2] = current_volume + count_i * total_volumes - offset;
		}
		else{
			idx[2] = current_volume + count_i * total_volumes;
		}
		idx[1] = 0;
		idx[0] = 0;		
		it2.SetIndex(idx); 
		cout<<"Slice #"<<count_i<<"it2.GetIndex()="<<it2.GetIndex()[2]<<", "<<it2.GetIndex()[1]<<", "<<it2.GetIndex()[0]<<endl;
		if (count_i == (slices_per_vol - 1)){		
			last_index = (current_volume + count_i * total_volumes - offset + 1);
		}
		else{
			last_index = (current_volume + count_i * total_volumes + 1);
		}		
		while(it2.GetIndex()[2] != last_index){	
			unsigned short voxel_value = it2.Get();
			int index_x = it2.GetIndex()[2];
			int index_y = it2.GetIndex()[1];		
			int index_z = it2.GetIndex()[0];
			long int index = (count_i) * s * t + index_y * s + index_z;		
			output_label[ index ] = voxel_value;
			++it2;
		}
	}
}


//Takes an array of x,y coordinates of a polygon's vertices and returns the centroid to the array passed in as a variable. The array of vertices must be passed as [x,y,x,y...]
inline void calculate_centroid(int pts_i, float* pts, int output_i, float* output) {
	//Could aslo try to pass the output to a new variable but I'm not sure if that will work
	//(*output_i) = 2;	
	//(*output) = new float[2];
	
	output[0] = 0;
	output[1] = 0;
	double signedArea = 0.0;
	double x0 = 0.0; // Current vertex X
	double y0 = 0.0; // Current vertex Y
	double x1 = 0.0; // Next vertex X
	double y1 = 0.0; // Next vertex Y
	double a = 0.0;  // Partial signed area

	// For all vertices except last

	for (int i = 0; i < (pts_i / 2 - 2); i++) {
	        x0 = pts[i * 2];
        	y0 = pts[i * 2 + 1];
        	x1 = pts[(i + 1) * 2];
        	y1 = pts[(i + 1) * 2 + 1];
//		cout<<"X0/Y0 ="<<x0<<"/"<<y0<<endl;
        	a = x0*y1 - x1*y0;
        	signedArea += a;
        	output[0] += (x0 + x1) * a;
        	output[1] += (y0 + y1) * a;
	}

	// Do last vertex	
	x0 = pts[pts_i - 2]; //-2 because pts_i is the total length, but the first element location is 0, therefore second to last element would be at pts_i - 2
	y0 = pts[pts_i - 1];
	x1 = pts[0];
	y1 = pts[1];
	a = x0*y1 - x1*y0;
	signedArea += a;
	output[0] += (x0 + x1) * a;
	output[1] += (y0 + y1) * a;

	signedArea *= 0.5;
	output[0] /= (6 * signedArea);
	output[1] /= (6 * signedArea);
}


//Takes an image (data), a list of relative spline points (t), a list of angles from the normal (theta), an initial radius estimate (a value at all radials),
//and adjusts the radius at each theta and each slice depending on the gradient strength. It returns an array of the new radii, the weightings to be assigned 
//to each slice based on how strong their gradients are, and a list of new spline points
void CRSpline::refine_spline(int T_i, float* T, int THETA_i, float* THETA, int S_i, float* S, int iterations, float radial_search_percentage, int image_type,
		int i, int j, int k, float* data, 
		int radii_y, int radii_x, float* radii, int* weight_i, float **weight, int* pts_i, int* pts_j, float **pts) {

	float force_multiplier = 4; //AS paper said 0.8 was best - I USED 4 FOR ANAT IMAGES	
	double ee = 2.71828182;	
	double PI = 3.141592;
	//float centroid_coord[2 * THETA_i];
	//int centroid_coord_size = 2 * THETA_i;	
	
	int direction_j;
	float x_coord;
	float y_coord;
	int ptx_max = i - 1;
	int pty_max = j - 1;
	int ptz_max = k - 1;
	float temp_radial_coords[THETA_i * 3];	
	float rad_cur;
	float rad_incr = 0.1;	
	
	(*weight_i) = T_i;
	(*weight) = new float[T_i];
	(*pts_i) = T_i;
	(*pts_j) = 3;
	(*pts) = new float[T_i * 3];

	//define the pixel and image types	
	typedef float PixelType;
	typedef itk::Image<PixelType, 3> ImageType;
	typedef itk::CovariantVector< double, 3 > GradientPixelType;
	typedef itk::Image< GradientPixelType, 3 > GradientImageType;
	typedef itk::GradientRecursiveGaussianImageFilter<ImageType, GradientImageType> GradientFilterType;
		
	//Initialize a new image which will read the input label	
	ImageType::Pointer image = ImageType::New();
	ImageType::IndexType start;
	start[0] = 0;
	start[1] = 0; 
	start[2] = 0;
	ImageType::SizeType size;
	size[0] = k;
	size[1] = j;
	size[2] = i;
	//cout<<"Image Size="<<k<<", "<<j<<", "<<i<<endl;
	ImageType::RegionType region;
	region.SetSize( size );
	region.SetIndex( start );
	image->SetRegions( region );
	image->Allocate();

	//Create an image iterator to copy the input label into the image file
	typedef itk::ImageRegionIterator< ImageType > IteratorType;
	//GradientImageType::Pointer image = gradientMapFilter->GetOutput();
	IteratorType it( image, image->GetRequestedRegion() );

	int count = 0;	
	while(!it.IsAtEnd()){	
		it.Set( data[ count ] );
		++it;
		count++;
	}
	
	//Calculate the vector gradient of the image
	GradientFilterType::Pointer gradientMapFilter = GradientFilterType::New();
	gradientMapFilter->SetInput( image );
	gradientMapFilter->SetSigma( 1.0 );
	gradientMapFilter->Update();
	//cout<<"Done gradient calculation"<<endl;	

	//Creates a new image and iterator of the gradient
	GradientImageType::Pointer image2 = gradientMapFilter->GetOutput();
	typedef itk::ImageRegionConstIterator< GradientImageType > IteratorType2;
	IteratorType2 it2( image2, image2->GetRequestedRegion() );
	ImageType::IndexType idx = it2.GetIndex();

	
	//Outputs the input image to test if the input worked correctly
	/*
	if (output_gradient_image == 1){	
		//cout<<"Starting Output Iterator..."<<endl;
		long int count2 = 0;	
		while(!it2.IsAtEnd()){	
			float magnitude = sqrt(it2.Get()[0] * it2.Get()[0] + it2.Get()[1] * it2.Get()[1] + it2.Get()[2] * it2.Get()[2]);
			int index_x = it2.GetIndex()[2];
			int index_y = it2.GetIndex()[1];		
			int index_z = it2.GetIndex()[0];
			long int index = (index_x) * s * t + index_y * s + index_z;		
			output_label[ index ] = magnitude ;
			count2++;
			++it2;
		}
		//cout<<"Output Iterator finished!"<<endl;
	}
	*/

	//Calculate the max and min gradient magnitude	
	//cout<<"Starting Magnitude Iterator..."<<endl;
	float mag_min = 10000;
	float mag_max = 0;
	idx[0] = 0;
	idx[1] = 0;
	idx[2] = 0;
	it2.SetIndex(idx);
	while(!it2.IsAtEnd()){	
		float magnitude = sqrt(it2.Get()[0] * it2.Get()[0] + it2.Get()[1] * it2.Get()[1] + it2.Get()[2] * it2.Get()[2]);
		if (magnitude > mag_max){		
			mag_max = magnitude;
		}
		if (magnitude < mag_min){		
			mag_min = magnitude;
		}
		++it2;		
	}
	//cout<<"Magnitude Iterator finished!"<<endl;
	//cout<<"Max Gradient Magnitude = "<<mag_max<<endl;
	//cout<<"Min Gradient Magnitude = "<<mag_min<<endl;
	
	for (int iter_i = 0; iter_i < iterations; iter_i++) {
		//Loop to calculate the radii at each vertice
		vec3 normal;		
		for (int t_i = 0; t_i < T_i; t_i++) {
			//cout<<"t="<<t_i<<endl;
			vec3 pt = CRSpline::GetInterpolatedSplinePoint(T[t_i]);
			vec3 v = CRSpline::GetInterpolatedSplineTangent(T[t_i]);			
			vec3 normal;		
			if (image_type == 1) { //checks if the image is functional or anatomical since they are captured differently
				normal = v.cross(vec3(-1, 0, 0)); //this sets the normal to be in the ventral/dorsal direction for functional images (front = ventral, back = dorsal)
			}			
			else {	
				normal = v.cross(vec3(0, 0, -1)); //this sets the normal to be in the ventral/dorsal direction for anatomical images (front = ventral, back = dorsal)
			}
			normal.normalize();			 
			float r;	

			float weight_sum = 0;
			//cout<<"Check 1: t_i="<<t_i<<endl;
			for (int theta_i = 0; theta_i < THETA_i; theta_i++) { 
				//cout<<"Check 1: t_i="<<t_i<<", theta_i="<<theta_i<<endl;
				//Look up the current radius for given theta				
				r = radii[t_i * THETA_i + theta_i];
				
				//if ((t_i == 30) && (theta_i == 90)){			
				//	cout<<"T = "<<t_i<<", theta = "<<theta_i<<", radius = "<<r<<endl;
				//}	
				if (r < 0){ // find_radius_from_gradient function was setting some radii to negative values so I put this in as a temporary fix
					r = 1;
					cout<<"Previous radius determination was less than 0 @ t="<<t_i<<", theta_i="<<theta_i<<endl;
				}
				
				//if (t_i == 0 && theta_i == 95){			
				//	cout<<"T = "<<t_i<<", theta = "<<theta_i<<", pt.x="<<pt.x<<", pt.y="<<pt.y<<", pt.z="<<pt.z<<", radius = "<<r<<endl;
				//}	
				float force = 0;
				float new_force = 0;
				float rot_angle;				
				if (image_type == 1){
					rot_angle = -THETA[theta_i]; //functional images need to be rotated in the opposite direction to the anatomical images
				}
				else{
					rot_angle = THETA[theta_i]; 
				}
				vec3 n_theta = v.rotatev(normal, rot_angle); //Points n_theta vector in the proper direction
				n_theta.normalize();
				float target_rad;

				//NEW SEGMENT HERE
				rad_cur = 0;
				float dist = 0;
				float target_dist = 0;
				while (dist < r) {
					vec3 pt2 = pt + n_theta * rad_cur;
					vec3 d = pt2 - pt;
					d.x *= S[0];
					d.y *= S[1];
					d.z *= S[2];
					dist = sqrt(d * d);
					rad_cur += rad_incr;
				}

				target_rad = rad_cur - rad_incr; //subtract rad_incr to take into account the last addition which occured before the while loop check
				//if ((t_i == 30) && (theta_i == 90)){					
				//	cout<<"target_rad="<<target_rad<<", dist="<<dist<<endl;
				//}

				//END OF NEW SEGMENT


				for (int j_count = -2; j_count < 3; j_count++){ //For loop for 5 locations along the radius
					//NEW CODE STARTS HERE
					target_dist = dist;
					//target_rad += target_rad * radial_search_percentage * (1 - iter_i / iterations) * j_count; //Finds length of radial at each of 5 locations
					target_dist += target_dist * radial_search_percentage * j_count; //Finds length of radial at each of 5 locations
					//Given the desired radius in mm, search along the radial line for that position
					rad_cur = 0;
					float dist_2 = 0;
					while (dist_2 < target_dist) {
						vec3 pt2 = pt + n_theta * rad_cur;
						vec3 d = pt2 - pt;
						d.x *= S[0];
						d.y *= S[1];
						d.z *= S[2];
						dist_2 = sqrt(d * d);
						rad_cur += rad_incr;
					}
					target_rad = rad_cur - rad_incr; //subtract rad_incr to take into account the last addition which occured before the while loop check
					vec3 pt2 = pt + n_theta * target_rad;
					//Check to see if the current point is out of the bounds of the image
					if (pt2.x > ptx_max){
						pt2.x = ptx_max;
					}
					else if (pt2.x < 0){
						pt2.x = 0;
					}

					if (pt2.y > pty_max){
						pt2.y = pty_max;
					}
					else if (pt2.y < 0){
						pt2.y = 0;
					}

					if (pt2.z > ptz_max){
						pt2.z = ptz_max;
					}
					else if (pt2.z < 0){
						pt2.z = 0;
					}
					
					//cout<<"pt2 location="<<pt2.x<<", "<<pt2.y<<", "<<pt2.z<<endl;
					idx[0] = pt2.z;
					idx[1] = pt2.y;
					idx[2] = pt2.x;
					it2.SetIndex(idx);
					vec3 gradient_vec = vec3(it2.Get()[2], it2.Get()[1], it2.Get()[0]);
					//cout<<it2.GetIndex()<<", grad.x="<<gradient_vec.x<<", y="<<gradient_vec.y<<", z="<<gradient_vec.z<<", curr_rad="<<curr_rad<<endl;										
					gradient_vec = gradient_vec / (mag_max - mag_min);		
					float dot_prod = gradient_vec * n_theta;
					//cout<<"Check 2"<<endl;				
					if (j_count < 0){
						direction_j = -1;
					}	
					else if (j_count == 0){
						direction_j = 0;
					}	
					else{
						direction_j = 1;
					}	

					new_force = dot_prod * direction_j;
					force += new_force;
					//if ((t_i == 30) && (theta_i = 90)){					
					//	cout<<"Force="<<force<<endl;
					//}
					
				}
				//if (t_i == 50 && (theta_i == 90 || theta_i == 0)){
				//	cout<<"X Force="<<x_force<<endl;
				//}
				//cout<<"Done j_count for loop."<<endl;
				//Update the radius in the grid			
				float new_radius = dist + force * force_multiplier; //float new_radius = target_rad + force * force_multiplier;
				
				if (new_radius < 0){
					new_radius = 1;
					cout<<"Error: New radius determined to be less than 0 @ t="<<t_i<<", theta_i="<<theta_i<<endl;
				}
				//if ((t_i == 30) && (theta_i == 90)){					
				//		cout<<"iter_i="<<iter_i<<", New Radius="<<new_radius<<endl;
				//}
				radii[t_i * THETA_i + theta_i] = new_radius;	

								

				//Determine gradient at new radius for the weighting calculation
				vec3 pt2 = pt + n_theta * new_radius; //THIS SHOULDN'T BE NEW_RADIUS ANYMORE
				
				if (pt2.x > ptx_max){
					pt2.x = ptx_max;
				}
				else if (pt2.x < 0){
					pt2.x = 0;
				}

				if (pt2.y > pty_max){
					pt2.y = pty_max;
				}
				else if (pt2.y < 0){
					pt2.y = 0;
				}

				if (pt2.z > ptz_max){
					pt2.z = ptz_max;
				}
				else if (pt2.z < 0){
					pt2.z = 0;
				}
				
			
				idx[0] = pt2.z;
				idx[1] = pt2.y;
				idx[2] = pt2.x;
				it2.SetIndex(idx);
				
				vec3 gradient_vec = vec3(it2.Get()[2], it2.Get()[1], it2.Get()[0]);
				gradient_vec.normalize();
				
				float dot_prod_weight = gradient_vec * n_theta;
				weight_sum += dot_prod_weight;
			}
			if (iter_i == (iterations - 1)) { //only do on the last iteration			
				//Calculate the weight placed on each slice. Weight depends on the direction of the radii versus gradient direction at each of the vertices.		
				(*weight)[t_i] = weight_sum; 
				//cout<<"t_i ="<<t_i<<", Old Center: x="<<pt.x<<", y="<<pt.y<<", z="<<pt.z<<", New Center: x="<<avg_x<<", y="<<avg_y<<", z="<<avg_z<<", weight="<<weight_sum<<endl;
				//cout<<"Weight="<<(*weight)[t_i]<<endl;
			}
		}
	}
}


//Takes an image (data), a list of relative spline points (t), a list of angles from the normal (theta), 
//and takes an initial guess of the radius at each theta and radial depending on the gradient strength. It returns an array of the new radii.
void CRSpline::find_radius_from_gradient(int T_i, float* T, int THETA_i, float* THETA, int S_i, float* S, int image_type, int i, int j, int k, float* data, int radii_y, int radii_x, float* radii, float max_radius, 
					int r, int s,  int t, unsigned short* output_label) {

	double PI = 3.141592;
	float delta_rad = 0.1;
	float rad_cur;
	//define the pixel and image types	
	typedef float PixelType;
	typedef itk::Image<PixelType, 3> ImageType;
	typedef itk::CovariantVector< double, 3 > GradientPixelType;
	typedef itk::Image< GradientPixelType, 3 > GradientImageType;
	typedef itk::GradientRecursiveGaussianImageFilter<ImageType, GradientImageType> GradientFilterType;
		
	//Initialize a new image which will read the input label	
	ImageType::Pointer image = ImageType::New();
	ImageType::IndexType start;
	start[0] = 0;
	start[1] = 0; 
	start[2] = 0;
	ImageType::SizeType size;
	size[0] = k;
	size[1] = j;
	size[2] = i;
	ImageType::RegionType region;
	region.SetSize( size );
	region.SetIndex( start );
	image->SetRegions( region );
	image->Allocate();

	//Create an image iterator to copy the input label into the image file
	typedef itk::ImageRegionIterator< ImageType > IteratorType;
	//GradientImageType::Pointer image = gradientMapFilter->GetOutput();
	IteratorType it( image, image->GetRequestedRegion() );

	int count = 0;	
	while(!it.IsAtEnd()){	
		it.Set( data[ count ] );
		++it;
		count++;
	}
	
	//Calculate the vector gradient of the image
	GradientFilterType::Pointer gradientMapFilter = GradientFilterType::New();
	gradientMapFilter->SetInput( image );
	gradientMapFilter->SetSigma( 1.0 );
	gradientMapFilter->Update();
	//cout<<"Done gradient calculation"<<endl;	

	//Creates a new image and iterator of the gradient
	GradientImageType::Pointer image2 = gradientMapFilter->GetOutput();
	typedef itk::ImageRegionConstIterator< GradientImageType > IteratorType2;
	IteratorType2 it2( image2, image2->GetRequestedRegion() );
	ImageType::IndexType idx = it2.GetIndex();

	//Outputs the input image to test if the input worked correctly

	//cout<<"Starting Output Iterator..."<<endl;
	long int count2 = 0;	
	while(!it2.IsAtEnd()){	
		float magnitude = sqrt(it2.Get()[0] * it2.Get()[0] + it2.Get()[1] * it2.Get()[1] + it2.Get()[2] * it2.Get()[2]);
		int index_x = it2.GetIndex()[2];
		int index_y = it2.GetIndex()[1];		
		int index_z = it2.GetIndex()[0];
		long int index = (index_x) * s * t + index_y * s + index_z;		
		output_label[ index ] = magnitude ;
		count2++;
		++it2;

	}
	//cout<<"Output Iterator finished!"<<endl;


	//Calculate the max and min gradient magnitude	
	//cout<<"Starting Magnitude Iterator..."<<endl;
	float mag_min = 10000;
	float mag_max = 0;
	idx[0] = 0;
	idx[1] = 0;
	idx[2] = 0;
	it2.SetIndex(idx);
	while(!it2.IsAtEnd()){	
		float magnitude = sqrt(it2.Get()[0] * it2.Get()[0] + it2.Get()[1] * it2.Get()[1] + it2.Get()[2] * it2.Get()[2]);
		if (magnitude > mag_max){		
			mag_max = magnitude;
		}
		if (magnitude < mag_min){		
			mag_min = magnitude;
		}
		++it2;		
	}
	//cout<<"Magnitude Iterator finished!"<<endl;
	
	vec3 normal;

	for (int t_i = 0; t_i < T_i; t_i++) {
		vec3 pt = CRSpline::GetInterpolatedSplinePoint(T[t_i]);
		//cout<<"pt.x="<<pt.x<<", pt.y="<<pt.y<<", pt.z="<<pt.z<<endl;
		vec3 v;
		

		if ((t_i > 0) && (t_i < (T_i - 2))){ //This is meant to take the average tangent of 3 points surrounding the desired location 
			//vec3 normal;		
			vec3 v2 = CRSpline::GetInterpolatedSplineTangent(T[t_i]);
			vec3 v1 = CRSpline::GetInterpolatedSplineTangent(T[t_i - 1]);
			vec3 v3 = CRSpline::GetInterpolatedSplineTangent(T[t_i + 1]);
									
			vec3 normal2;		
			vec3 normal1;		
			vec3 normal3;		
			if (image_type == 1) { //checks if the image is functional or anatomical since they are captured differently
				normal2 = v2.cross(vec3(-1, 0, 0)); //this sets the normal to be in the ventral/dorsal direction for functional images (front = ventral, back = dorsal)
				normal1 = v1.cross(vec3(-1, 0, 0)); //this sets the normal to be in the ventral/dorsal direction for functional images (front = ventral, back = dorsal)
				normal3 = v3.cross(vec3(-1, 0, 0)); //this sets the normal to be in the ventral/dorsal direction for functional images (front = ventral, back = dorsal)
			}			
			else {	
				normal2 = v2.cross(vec3(0, 0, -1)); //this sets the normal to be in the ventral/dorsal direction for anatomical images (front = ventral, back = dorsal)
				normal1 = v1.cross(vec3(0, 0, -1)); //this sets the normal to be in the ventral/dorsal direction for anatomical images (front = ventral, back = dorsal)
				normal3 = v3.cross(vec3(0, 0, -1)); //this sets the normal to be in the ventral/dorsal direction for anatomical images (front = ventral, back = dorsal)
			}
			normal2.normalize(); 
			normal1.normalize();
			normal3.normalize();			

			v.x = (v1.x + v2.x + v3.x) / 3;
			v.y = (v1.y + v2.y + v3.y) / 3;
			v.z = (v1.z + v2.z + v3.z) / 3;

			normal.x = (normal1.x + normal2.x + normal3.x) / 3;
			normal.y = (normal1.y + normal2.y + normal3.y) / 3;
			normal.z = (normal1.z + normal2.z + normal3.z) / 3;
		}
		else {
			v = CRSpline::GetInterpolatedSplineTangent(T[t_i]);						
			//vec3 normal;		
			if (image_type == 1) { //checks if the image is functional or anatomical since they are captured differently
				normal = v.cross(vec3(-1, 0, 0)); //this sets the normal to be in the ventral/dorsal direction for functional images (front = ventral, back = dorsal)
			}			
			else {	
				normal = v.cross(vec3(0, 0, -1)); //this sets the normal to be in the ventral/dorsal direction for anatomical images (front = ventral, back = dorsal)
			}			
			normal.normalize(); 
			//cout<<"t="<<t_i<<", normal="<<normal.x<<" "<<normal.y<<" "<<normal.z<<endl;			
		}
		float r;
		
		vec3 max_point = pt;	
		for (int theta_i = 0; theta_i < THETA_i; theta_i++) { 
			//cout<<"T_i="<<t_i<<", theta_i="<<theta_i<<endl;
			float max_grad = 0;			
			float target_rad = 0.5; 
			float rot_angle;				
			if (image_type == 1){
				rot_angle = -THETA[theta_i]; //functional images need to be rotated in the opposite direction to the anatomical images
			}
			else{
				rot_angle = THETA[theta_i]; 
			}
			//cout<<"t="<<t_i<<", normal="<<normal.x<<" "<<normal.y<<" "<<normal.z<<endl;			
			vec3 n_theta = v.rotatev(normal, rot_angle);
			n_theta.normalize();
			//if (theta_i == 0){			
			//cout<<"t="<<t_i<<", pt="<<pt.x<<" "<<pt.y<<" "<<pt.z<<", n_theta="<<n_theta.x<<" "<<n_theta.y<<" "<<n_theta.z<<endl;
			//}
		
			//NEW CODE
			rad_cur = 0;
			float dist = 0;
			//float temp_grad[int(max_radius / delta_rad)];
			//float temp_rad[int(max_radius / delta_rad)];
			//int counter = 0;
			while (dist < max_radius) {
				vec3 pt2 = pt + n_theta * rad_cur;
				//cout<<"dist="<<dist<<endl;
				//if ((theta_i == 135) && (t_i == 20)) {
				//cout<<"pt ="<<pt.x<<" "<<pt.y<<" "<<pt.z<<", pt2 ="<<pt2.x<<" "<<pt2.y<<" "<<pt2.z<<endl;
				//}
				vec3 d = pt2 - pt;
				d.x *= S[0];
				d.y *= S[1];
				d.z *= S[2];
				dist = sqrt(d * d);
				
				idx[0] = pt2.z;
				idx[1] = pt2.y;
				idx[2] = pt2.x;
				it2.SetIndex(idx);	
				vec3 gradient_vec = vec3(it2.Get()[2], it2.Get()[1], it2.Get()[0]);
				float grad_mag = gradient_vec.norm();	
				
				//temp_grad[counter] = grad_mag;
				//temp_rad[counter] = dist;

				//can delete if statement below  
				if (image_type == 1){				
					if ((grad_mag < 60 && max_grad > 100) || (grad_mag > 100 )){
						break; //breaks from the loop if a maximum has been reached and then the grad decreases before another accension (happens in areas where CSF volume is low)
					}					
				}
				else{
					if ((grad_mag < 300 && max_grad > 400) || (grad_mag < 700 && max_grad > 1000) || (grad_mag < 1000 && max_grad > 2000) || (grad_mag < 200 && max_grad > 300) || (grad_mag < 500 && max_grad > 700) || (grad_mag > 300)){
						break; //breaks from the loop if a maximum has been reached and then the grad decreases before another accension (happens in areas where CSF volume is low)
					}
				}

				if (grad_mag > max_grad){
					max_grad = grad_mag;
					max_point = pt2;
				}				


				rad_cur += delta_rad;
				
			}
	
		
			radii[t_i * THETA_i + theta_i] = dist; //THIS LINE REPLACED THE ONE BELOW WHEN THE NEW CODE WAS ADDED
			//radii[t_i * THETA_i + theta_i] = target_rad - delta_rad;
		}
	}
	//cout<<"Initial Radius Calculation Finished"<<endl;
}

///Returns the components of the gradient vectors at the outer radius of the spline
void CRSpline::get_vector_information(int T_i, float* T, int THETA_i, float* THETA, int radii_y, int radii_x, float* radii, int i, int j, int k, float* data,  
					int* U_i, int* U_j, int* U_k, float **U, int* V_i, int* V_j, int* V_k, float **V, int* W_i, int* W_j, int* W_k, float **W) {

	(*U_i) = k;
	(*U_j) = j;
	(*U_k) = i;
	(*U) = new float[k * j * i];	
	(*V_i) = k;
	(*V_j) = j;
	(*V_k) = i;
	(*V) = new float[k * j * i];
	(*W_i) = k;
	(*W_j) = j;
	(*W_k) = i;
	(*W) = new float[k * j * i];
	
	//define the pixel and image types	
	typedef float PixelType;
	typedef itk::Image<PixelType, 3> ImageType;
	typedef itk::CovariantVector< double, 3 > GradientPixelType;
	typedef itk::Image< GradientPixelType, 3 > GradientImageType;
	typedef itk::GradientRecursiveGaussianImageFilter<ImageType, GradientImageType> GradientFilterType;
		
	//Initialize a new image which will read the input label	
	ImageType::Pointer image = ImageType::New();
	ImageType::IndexType start;
	start[0] = 0;
	start[1] = 0; 
	start[2] = 0;
	ImageType::SizeType size;
	size[0] = k;
	size[1] = j;
	size[2] = i;
	ImageType::RegionType region;
	region.SetSize( size );
	region.SetIndex( start );
	image->SetRegions( region );
	image->Allocate();
	cout<<"Image Size="<<k<<", "<<j<<", "<<i<<endl;

	//Create an image iterator to copy the input label into the image file
	typedef itk::ImageRegionIterator< ImageType > IteratorType;
	//GradientImageType::Pointer image = gradientMapFilter->GetOutput();
	IteratorType it( image, image->GetRequestedRegion() );

	int count = 0;	
	while(!it.IsAtEnd()){	
		it.Set( data[ count ] );
		++it;
		count++;
	}
	
	//Calculate the vector gradient of the image
	GradientFilterType::Pointer gradientMapFilter = GradientFilterType::New();
	gradientMapFilter->SetInput( image );
	gradientMapFilter->SetSigma( 1.0 );
	gradientMapFilter->Update();
	//cout<<"Done gradient calculation"<<endl;	

	//Creates a new image and iterator of the gradient
	GradientImageType::Pointer image2 = gradientMapFilter->GetOutput();
	typedef itk::ImageRegionConstIterator< GradientImageType > IteratorType2;
	IteratorType2 it2( image2, image2->GetRequestedRegion() );
	ImageType::IndexType idx = it2.GetIndex();
	
	for (int t_i = 0; t_i < T_i; t_i++) {
		vec3 pt = CRSpline::GetInterpolatedSplinePoint(T[t_i]);
		vec3 v = CRSpline::GetInterpolatedSplineTangent(T[t_i]);			
		vec3 normal = v.cross(vec3(0, 0, -1)); //this sets the normal to be in the ventral/dorsal direction for anatomical images (front = ventral, back = dorsal)
		normal.normalize(); 
		float r;
		
		vec3 max_point = pt;	
		for (int theta_i = 0; theta_i < THETA_i; theta_i++) { 
			vec3 n_theta = v.rotatev(normal, THETA[theta_i]); //Points n_theta vector in the proper direction
			n_theta.normalize();
			r = radii[t_i * THETA_i + theta_i];
			float target_rad = r;			
			float actual_rad = 0;										
			while (abs(actual_rad - target_rad)  > 0.1){						
				if (actual_rad < target_rad){
					target_rad += 0.1;
				}
				else {
					target_rad -= 0.1;
				}
				vec3 pt2_temp = pt + n_theta * target_rad;
				vec3 radial_line = pt2_temp - pt;
				actual_rad = radial_line.norm();
				//cout<<"target_rad="<<target_rad<<", r="<<r<<", actual_rad="<<actual_rad<<endl;
			}
			//cout<<"target_rad after adj.="<<target_rad<<endl;
			vec3 pt2 = pt + n_theta * target_rad;
					
			//cout<<"pt2 location="<<pt2.x<<", "<<pt2.y<<", "<<pt2.z<<endl;
			idx[0] = pt2.z;
			idx[1] = pt2.y;
			idx[2] = pt2.x;
			it2.SetIndex(idx);
			vec3 gradient_vec = vec3(it2.Get()[2], it2.Get()[1], it2.Get()[0]);
			gradient_vec.normalize();
			(*U)[IND(i, j, k, pt.x, pt.y, pt.z)] = gradient_vec.x;
			(*V)[IND(i, j, k, pt.x, pt.y, pt.z)] = gradient_vec.y;
			(*W)[IND(i, j, k, pt.x, pt.y, pt.z)] = gradient_vec.z;
			
		}
	}

}

void CRSpline::create_image_subsegment(int i, int j, int k, float* data, int P_i, int P_j, float* P, int l, int m, int n, unsigned short* label_ip) {

	//define the pixel and image types	
	typedef float PixelType;
	typedef itk::Image<PixelType, 3> ImageType;
		
	//Initialize a new image which will read the input label	
	ImageType::Pointer image = ImageType::New();
	ImageType::IndexType start;
	start[0] = 0;
	start[1] = 0; 
	start[2] = 0;
	ImageType::SizeType size;
	size[0] = k;
	size[1] = j;
	size[2] = i;
	cout<<"Input Image Size="<<k<<", "<<j<<", "<<i<<endl;
	cout<<"Output Image Size="<<n<<", "<<m<<", "<<l<<endl;
	ImageType::RegionType region;
	region.SetSize( size );
	region.SetIndex( start );
	image->SetRegions( region );
	image->Allocate();

	//Create an image iterator to copy the input label into the image file
	typedef itk::ImageRegionIterator< ImageType > IteratorType;
	IteratorType it( image, image->GetRequestedRegion() );

	int count = 0;	
	while(!it.IsAtEnd()){	
		it.Set( data[ count ] );
		++it;
		count++;
	}
	
	long int count2 = 0;	
	int a_count = 0;
	int b_count = 0;
	int c_count = 0;
	ImageType::IndexType idx = it.GetIndex();


	for(int a = P[4]; a < P[5]; a++) {
		for (int b = P[2]; b < P[3]; b++) {
 			for (int c = P[0]; c < P[1]; c++) {
				long int index = a_count * n * m + b_count * m + c_count;
				//cout<<"index="<<index<<endl;
				idx[0] = c;
				idx[1] = b;
				idx[2] = a;
				it.SetIndex(idx);
				//cout<<"idx="<<idx<<endl;
				float value = it.Get();
				label_ip[ index ]  = value;	
				c_count++;
				//cout<<"c="<<c<<", b="<<b<<", a="<<a<<", value="<<value<<endl;
			}
			c_count = 0;
			b_count++;
		}
		b_count = 0;
		a_count++;
	}

}

//Given an input 3D image, calculates the gaussian of the x/y/z components of the gradient, outputs the magnitude of the x-components
//and outputs the magnitude of all three components together.
void CRSpline::gaussian_of_gradient(int i, int j, int k, float* data, int r, int s,  int t, unsigned short* output_label, int a, int b,  int c, unsigned short* output_label_2) {

	//define the pixel and image types	
	typedef float PixelType;
	typedef itk::Image<PixelType, 3> ImageType;
	typedef itk::CovariantVector< double, 3 > GradientPixelType;
	typedef itk::Image< GradientPixelType, 3 > GradientImageType;
	typedef itk::GradientRecursiveGaussianImageFilter<ImageType, GradientImageType> GradientFilterType;
		
	//Initialize a new image which will read the input label	
	ImageType::Pointer image = ImageType::New();
	ImageType::IndexType start;
	start[0] = 0;
	start[1] = 0; 
	start[2] = 0;
	ImageType::SizeType size;
	size[0] = k;
	size[1] = j;
	size[2] = i;
	//cout<<"Image Size="<<k<<", "<<j<<", "<<i<<endl;
	ImageType::RegionType region;
	region.SetSize( size );
	region.SetIndex( start );
	image->SetRegions( region );
	image->Allocate();

	//Create an image iterator to copy the input label into the image file
	typedef itk::ImageRegionIterator< ImageType > IteratorType;
	//GradientImageType::Pointer image = gradientMapFilter->GetOutput();
	IteratorType it( image, image->GetRequestedRegion() );

	int count = 0;	
	while(!it.IsAtEnd()){	
		it.Set( data[ count ] );
		++it;
		count++;
	}
	
	//Calculate the vector gradient of the image
	GradientFilterType::Pointer gradientMapFilter = GradientFilterType::New();
	gradientMapFilter->SetInput( image );
	gradientMapFilter->SetSigma( 1.0 );
	gradientMapFilter->Update();

	//Creates a new image and iterator of the gradient
	GradientImageType::Pointer image2 = gradientMapFilter->GetOutput();
	typedef itk::ImageRegionConstIterator< GradientImageType > IteratorType2;
	IteratorType2 it2( image2, image2->GetRequestedRegion() );
	ImageType::IndexType idx = it2.GetIndex();

	/*
	//Outputs the input image to test if the input worked correctly
	long int count2 = 0;	
	while(!it2.IsAtEnd()){	
		float magnitude = sqrt(it2.Get()[0] * it2.Get()[0] + it2.Get()[1] * it2.Get()[1] + it2.Get()[2] * it2.Get()[2]);
		int index_x = it2.GetIndex()[2];
		int index_y = it2.GetIndex()[1];		
		int index_z = it2.GetIndex()[0];
		long int index = (index_x) * s * t + index_y * s + index_z;		
		output_label[ index ] = magnitude ;
		count2++;
		++it2;
	}
	*/


	//NEW PART
	//Creates three new images, one for each of the gradient components
	ImageType::Pointer image_x = ImageType::New();
	image_x->SetRegions( region );
	image_x->Allocate();

	ImageType::Pointer image_y = ImageType::New();
	image_y->SetRegions( region );
	image_y->Allocate();

	ImageType::Pointer image_z = ImageType::New();
	image_z->SetRegions( region );
	image_z->Allocate();

	IteratorType it_x( image_x, image_x->GetRequestedRegion() );
	IteratorType it_y( image_y, image_y->GetRequestedRegion() );
	IteratorType it_z( image_z, image_z->GetRequestedRegion() );

	idx[2] = 0;
	idx[1] = 0;
	idx[0] = 0;		
	it2.SetIndex(idx); 
	//int count = 0;	
	while(!it_x.IsAtEnd()){	
		it_x.Set( it2.Get()[0] );
		int index_x = it_x.GetIndex()[2];
		int index_y = it_x.GetIndex()[1];		
		int index_z = it_x.GetIndex()[0];
		long int index = (index_x) * s * t + index_y * s + index_z;		
		output_label[ index ] = it_x.Get();
		++it_x;
		++it2;
		//count++;
	}
	
	typedef itk::RecursiveGaussianImageFilter<ImageType, ImageType> GaussianFilterType;
	
	//Calculate the gaussian of the image
	GaussianFilterType::Pointer gaussianFilter = GaussianFilterType::New();

	gaussianFilter->SetInput( image_x );
	gaussianFilter->SetSigma( 1.0 );
	gaussianFilter->Update();
	ImageType::Pointer image_x_gauss = gaussianFilter->GetOutput();

	typedef itk::ImageRegionConstIterator< ImageType > IteratorType3;
	IteratorType3 it_x_gauss( image_x_gauss, image_x_gauss->GetRequestedRegion() );
	ImageType::IndexType idx_x_gauss = it_x_gauss.GetIndex();

	GaussianFilterType::Pointer gaussianFilter2 = GaussianFilterType::New();
	gaussianFilter2->SetInput( image_y );
	gaussianFilter2->SetSigma( 1.0 );
	gaussianFilter2->Update();
	ImageType::Pointer image_y_gauss = gaussianFilter2->GetOutput();

	typedef itk::ImageRegionConstIterator< ImageType > IteratorType4;
	IteratorType4 it_y_gauss( image_y_gauss, image_y_gauss->GetRequestedRegion() );
	ImageType::IndexType idx_y_gauss = it_y_gauss.GetIndex();

	GaussianFilterType::Pointer gaussianFilter3 = GaussianFilterType::New();
	gaussianFilter3->SetInput( image_z );
	gaussianFilter3->SetSigma( 1.0 );
	gaussianFilter3->Update();
	ImageType::Pointer image_z_gauss = gaussianFilter3->GetOutput();

	typedef itk::ImageRegionConstIterator< ImageType > IteratorType5;
	IteratorType5 it_z_gauss( image_z_gauss, image_z_gauss->GetRequestedRegion() );
	ImageType::IndexType idx_z_gauss = it_z_gauss.GetIndex();

	//Outputs the input image to test if the input worked correctly
	it_x_gauss.SetIndex(idx);
	it_y_gauss.SetIndex(idx);
	it_z_gauss.SetIndex(idx);
	int count2 = 0;	
	while(!it_x_gauss.IsAtEnd()){	
		float magnitude = sqrt(it_x_gauss.Get() * it_x_gauss.Get() + it_y_gauss.Get() * it_y_gauss.Get() + it_z_gauss.Get() * it_z_gauss.Get());
		int index_x = it_x_gauss.GetIndex()[2];
		int index_y = it_x_gauss.GetIndex()[1];		
		int index_z = it_x_gauss.GetIndex()[0];
		long int index = (index_x) * b * c + index_y * b + index_z;		
		output_label_2[ index ] = magnitude;
		count2++;
		++it_x_gauss;
		++it_y_gauss;
		++it_z_gauss;
	}

}



//Given an input 3D image, calculates the gradient of the x/y/z components of the gradient(i.e., the second gradient), 
//outputs the magnitude of the x-components and outputs the magnitude of all three components together.
void CRSpline::gradient_of_gradient(int i, int j, int k, float* data, int r, int s,  int t, unsigned short* output_label, int a, int b,  int c, unsigned short* output_label_2) {

	//define the pixel and image types	
	typedef float PixelType;
	typedef itk::Image<PixelType, 3> ImageType;
	typedef itk::CovariantVector< double, 3 > GradientPixelType;
	typedef itk::Image< GradientPixelType, 3 > GradientImageType;
	typedef itk::GradientRecursiveGaussianImageFilter<ImageType, GradientImageType> GradientFilterType;
		
	//Initialize a new image which will read the input label	
	ImageType::Pointer image = ImageType::New();
	ImageType::IndexType start;
	start[0] = 0;
	start[1] = 0; 
	start[2] = 0;
	ImageType::SizeType size;
	size[0] = k;
	size[1] = j;
	size[2] = i;
	//cout<<"Image Size="<<k<<", "<<j<<", "<<i<<endl;
	ImageType::RegionType region;
	region.SetSize( size );
	region.SetIndex( start );
	image->SetRegions( region );
	image->Allocate();

	//Create an image iterator to copy the input label into the image file
	typedef itk::ImageRegionIterator< ImageType > IteratorType;
	//GradientImageType::Pointer image = gradientMapFilter->GetOutput();
	IteratorType it( image, image->GetRequestedRegion() );

	int count = 0;	
	while(!it.IsAtEnd()){	
		it.Set( data[ count ] );
		++it;
		count++;
	}
	
	//Calculate the vector gradient of the image
	GradientFilterType::Pointer gradientMapFilter = GradientFilterType::New();
	gradientMapFilter->SetInput( image );
	gradientMapFilter->SetSigma( 1.0 );
	gradientMapFilter->Update();

	//Creates a new image and iterator of the gradient
	GradientImageType::Pointer image2 = gradientMapFilter->GetOutput();
	typedef itk::ImageRegionConstIterator< GradientImageType > IteratorType2;
	IteratorType2 it2( image2, image2->GetRequestedRegion() );
	ImageType::IndexType idx = it2.GetIndex();


	//NEW PART
	//Creates three new images, one for each of the gradient components
	ImageType::Pointer image_x = ImageType::New();
	image_x->SetRegions( region );
	image_x->Allocate();

	ImageType::Pointer image_y = ImageType::New();
	image_y->SetRegions( region );
	image_y->Allocate();

	ImageType::Pointer image_z = ImageType::New();
	image_z->SetRegions( region );
	image_z->Allocate();

	IteratorType it_x( image_x, image_x->GetRequestedRegion() );
	IteratorType it_y( image_y, image_y->GetRequestedRegion() );
	IteratorType it_z( image_z, image_z->GetRequestedRegion() );

	idx[2] = 0;
	idx[1] = 0;
	idx[0] = 0;		
	it2.SetIndex(idx); 
	//int count = 0;	
	while(!it_x.IsAtEnd()){	
		it_x.Set( it2.Get()[0] );
		int index_x = it_x.GetIndex()[2];
		int index_y = it_x.GetIndex()[1];		
		int index_z = it_x.GetIndex()[0];
		long int index = (index_x) * s * t + index_y * s + index_z;		
		output_label[ index ] = it_x.Get();
		++it_x;
		++it2;
		//count++;
	}
	
	typedef itk::GradientMagnitudeRecursiveGaussianImageFilter<ImageType, ImageType> GradientMagnitudeFilterType;
	
	//Calculate the gaussian of the image
	GradientMagnitudeFilterType::Pointer gradMagFilter = GradientMagnitudeFilterType::New();

	gradMagFilter->SetInput( image_x );
	gradMagFilter->SetSigma( 1.0 );
	gradMagFilter->Update();
	ImageType::Pointer image_x_grad = gradMagFilter->GetOutput();

	typedef itk::ImageRegionConstIterator< ImageType > IteratorType3;
	IteratorType3 it_x_grad( image_x_grad, image_x_grad->GetRequestedRegion() );
	ImageType::IndexType idx_x_grad = it_x_grad.GetIndex();

	GradientMagnitudeFilterType::Pointer gradMagFilter2 = GradientMagnitudeFilterType::New();
	gradMagFilter2->SetInput( image_y );
	gradMagFilter2->SetSigma( 1.0 );
	gradMagFilter2->Update();
	ImageType::Pointer image_y_grad = gradMagFilter2->GetOutput();

	typedef itk::ImageRegionConstIterator< ImageType > IteratorType4;
	IteratorType4 it_y_grad( image_y_grad, image_y_grad->GetRequestedRegion() );
	ImageType::IndexType idx_y_grad = it_y_grad.GetIndex();

	GradientMagnitudeFilterType::Pointer gradMagFilter3 = GradientMagnitudeFilterType::New();
	gradMagFilter3->SetInput( image_z );
	gradMagFilter3->SetSigma( 1.0 );
	gradMagFilter3->Update();
	ImageType::Pointer image_z_grad = gradMagFilter3->GetOutput();

	typedef itk::ImageRegionConstIterator< ImageType > IteratorType5;
	IteratorType5 it_z_grad( image_z_grad, image_z_grad->GetRequestedRegion() );
	ImageType::IndexType idx_z_grad = it_z_grad.GetIndex();

	//Outputs the input image to test if the input worked correctly
	it_x_grad.SetIndex(idx);
	it_y_grad.SetIndex(idx);
	it_z_grad.SetIndex(idx);
	int count2 = 0;	
	while(!it_x_grad.IsAtEnd()){	
		float magnitude = sqrt(it_x_grad.Get() * it_x_grad.Get() + it_y_grad.Get() * it_y_grad.Get() + it_z_grad.Get() * it_z_grad.Get());
		int index_x = it_x_grad.GetIndex()[2];
		int index_y = it_x_grad.GetIndex()[1];		
		int index_z = it_x_grad.GetIndex()[0];
		long int index = (index_x) * b * c + index_y * b + index_z;		
		output_label_2[ index ] = magnitude;
		count2++;
		++it_x_grad;
		++it_y_grad;
		++it_z_grad;
	}
}


//Given a set of radii points, calculates the center of each slice
void CRSpline::calculate_new_center_points(int T_i, float* T, int THETA_i, float* THETA, int S_i, float* S,  
		int radii_y, int radii_x, float* radii, int* pts_i, int* pts_j, float **pts) {

	float rad_incr = 0.1;
	float x_coord;
	float y_coord;
	float z_coord;

	(*pts_i) = T_i;
	(*pts_j) = 3;
	(*pts) = new float[T_i * 3];

	vec3 normal;		
	for (int t_i = 0; t_i < T_i; t_i++) {
		x_coord = 0;
		y_coord = 0;
		z_coord = 0;
		vec3 pt = CRSpline::GetInterpolatedSplinePoint(T[t_i]);
		vec3 v = CRSpline::GetInterpolatedSplineTangent(T[t_i]);			
		vec3 normal = v.cross(vec3(0, 0, -1)); //this sets the normal to be in the ventral/dorsal direction for anatomical images (front = ventral, back = dorsal)
		normal.normalize();			 
		for (int theta_i = 0; theta_i < THETA_i; theta_i++) {
			vec3 n_theta = v.rotatev(normal, THETA[theta_i]); 
			n_theta.normalize();
			float rad_cur = 0;
			float dist = 0;
			while (dist < radii[t_i * THETA_i + theta_i]) {
				vec3 pt2 = pt + n_theta * rad_cur;
				vec3 d = pt2 - pt;
				d.x *= S[0];
				d.y *= S[1];
				d.z *= S[2];
				dist = sqrt(d * d);
				rad_cur += rad_incr;
			}
			vec3 pt2 = pt + n_theta * rad_cur;
			x_coord += pt2.x;
			y_coord += pt2.y;
			z_coord += pt2.z;
		}			

		float avg_x = x_coord / (THETA_i);
		float avg_y = y_coord / (THETA_i);
		float avg_z = z_coord / (THETA_i);

		(*pts)[t_i * 3 + 0] = avg_x;
		(*pts)[t_i * 3 + 1] = avg_y;
		(*pts)[t_i * 3 + 2] = avg_z;
	}
}

//Given a set of radii points, returns the set of co-ordinates at a given angle for all values of t. Used for determining hole locations
void CRSpline::get_radial_coordinates(int T_i, float* T, int THETA_i, float* THETA, int theta_targ, int S_i, float* S,  
		int radii_y, int radii_x, float* radii, int* pts_i, int* pts_j, float **pts) {

	float rad_incr = 0.1;

	(*pts_i) = T_i;
	(*pts_j) = 3;
	(*pts) = new float[T_i * 3];

	vec3 normal;		
	for (int t_i = 0; t_i < T_i; t_i++) {
		vec3 pt = CRSpline::GetInterpolatedSplinePoint(T[t_i]);
		vec3 v = CRSpline::GetInterpolatedSplineTangent(T[t_i]);			
		vec3 normal = v.cross(vec3(0, 0, -1)); //this sets the normal to be in the ventral/dorsal direction for anatomical images (front = ventral, back = dorsal)
		normal.normalize();			 
		vec3 n_theta = v.rotatev(normal, THETA[theta_targ]); 
		n_theta.normalize();
		float rad_cur = 0;
		float dist = 0;
		while (dist < radii[t_i * THETA_i + theta_targ]) {
			vec3 pt2 = pt + n_theta * rad_cur;
			vec3 d = pt2 - pt;
			d.x *= S[0];
			d.y *= S[1];
			d.z *= S[2];
			dist = sqrt(d * d);
			rad_cur += rad_incr;
		}
		vec3 pt2 = pt + n_theta * rad_cur;
		(*pts)[t_i * 3 + 0] = pt2.x;
		(*pts)[t_i * 3 + 1] = pt2.y;
		(*pts)[t_i * 3 + 2] = pt2.z;
	}
}

void CRSpline::get_grad_values_line(float T, float THETA, int search_points, int S_i, float* S, int i, int j, int k, float* data, int image_type,
		int* pts_i, float **pts) {

	float rad_incr = 0.1;
	cout<<"i="<<i<<", j="<<j<<", k="<<k<<endl;
	(*pts_i) = search_points;
	(*pts) = new float[search_points];

	vec3 pt = CRSpline::GetInterpolatedSplinePoint(T);
	vec3 v = CRSpline::GetInterpolatedSplineTangent(T);			
	vec3 normal = v.cross(vec3(0, 0, -1)); //this sets the normal to be in the ventral/dorsal direction for anatomical images (front = ventral, back = dorsal)
	normal.normalize();			 
	vec3 n_theta = v.rotatev(normal, THETA); 
	n_theta.normalize();
	float rad_cur = 0;
	float dist = 0;
	float new_dist = 0;
	int counter = 0;
	while (counter < search_points) {
		while (dist == new_dist) {	
			vec3 pt2 = pt + n_theta * rad_cur;
			vec3 d = pt2 - pt;
			d.x *= S[0];
			d.y *= S[1];
			d.z *= S[2];
			new_dist = sqrt(d * d);
			rad_cur += rad_incr;
			//cout<<"new_dist = "<<new_dist<<endl;
		}
		dist = new_dist;
		counter += 1;
		vec3 pt2 = pt + n_theta * (rad_cur - rad_incr);
		float grad_val = data[IND(i, j, k, pt2.x, pt2.y, pt2.z)];
		cout<<"counter="<<counter<<",grad_val = "<<grad_val<<", pt2.x = "<<pt2.x<<", pt2.y = "<<pt2.y<<", pt2.z = "<<pt2.z<<endl;
		//if (grad_val < 10){		
		//	(*pts)[counter - 1] = 0;
		//}
		//else {
		(*pts)[counter - 1] = grad_val;
		//}
		//cout<<"counter="<<counter<<", pt2.x = "<<pt2.x<<", pt2.y = "<<pt2.y<<", pt2.z = "<<pt2.z<<endl;
	}
}

void CRSpline::create_templates(int T_i, float* T,  int THETA_i, float* THETA, int axial, float start_t, int theta_start, int theta_end, int i, int j, int k, float* data, int d, int e,  int f, unsigned short* input_label, 
	int points, int S_i, float* S, int image_type, int* a, int* b, float **D, int* A_i, float **A, int* B_i, float **B, int* pts_i, int* pts_j, float **pts) {

	(*a) = T_i * THETA_i;
	(*b) = points;
	(*D) = new float[T_i * THETA_i * points]; //this is the template output
	(*A_i) = T_i * THETA_i;
	(*A) = new float[T_i * THETA_i]; //this is the index value
	(*B_i) = T_i * THETA_i;
	(*B) = new float[T_i * THETA_i]; //this is the distance value in mm
	(*pts_i) = T_i * THETA_i;
	(*pts_j) = 3;
	(*pts) = new float[T_i * THETA_i * 3];
	int cur_pt = 0;
	long int last_index;
	float delta_rad = 0.1;
	float rad_cur;
	int ptx_max = i - 1;
	int pty_max = j - 1;
	int ptz_max = k - 1;
	int edge_hit = 0;

	cout<<"Theta Start: "<<theta_start<<"; Theta End: "<<theta_end<<endl;
	
	//Initialize all of the output arrays
	for (int a = 0; a < (T_i * THETA_i * points); a++) {   
		(*D)[a] = 0;
	}
	
	for (int a = 0; a < (T_i * THETA_i); a++) {   
		(*A)[a] = 0;
		(*B)[a] = 0;
	}
	
	
	for (int t_i = start_t * T_i; t_i < T_i; t_i++) {
		vec3 pt = CRSpline::GetInterpolatedSplinePoint(T[t_i]);
		long int last_index = pt.x * j * k + pt.y * k + pt.z;
		//cout<<"pt.x="<<pt.x<<", pt.y="<<pt.y<<", pt.z="<<pt.z<<endl;
		//cout<<"t_i="<<t_i<<endl;	
		vec3 v = vec3(0,1,0);
		if (axial != 1) {
			vec3 v = CRSpline::GetInterpolatedSplineTangent(T[t_i]);			
		}
								
		vec3 normal;		

		if (image_type == 1) { //checks if the image is functional or anatomical since they are captured differently
			normal = v.cross(vec3(-1, 0, 0)); //this sets the normal to be in the ventral/dorsal direction for functional images (front = ventral, back = dorsal)
		}			
		else {	
			normal = v.cross(vec3(0, 0, -1)); //this sets the normal to be in the ventral/dorsal direction for anatomical images (front = ventral, back = dorsal)
		}
		normal.normalize(); 

		float r;
		
		vec3 max_point = pt;	
		
		vec3 pt_last = vec3(0,0,0);
		for (int theta_i = 0; theta_i < THETA_i; theta_i++) { 
			
			float rot_angle;				
			if (image_type == 1){
				rot_angle = -THETA[theta_i]; //functional images need to be rotated in the opposite direction to the anatomical images
			}
			else{
				rot_angle = THETA[theta_i]; 
			}
			//cout<<"t="<<t_i<<", normal="<<normal.x<<" "<<normal.y<<" "<<normal.z<<endl;			
			vec3 n_theta = v.rotatev(normal, rot_angle);
			n_theta.normalize();

			rad_cur = 0;
			cur_pt = 0;
			float dist = 0;
			//cout<<"pt.x="<<pt.x<<", pt.y="<<pt.y<<", pt.z="<<pt.z<<endl;
			//cout<<"pt_last.x="<<pt_last.x<<", pt_last.y="<<pt_last.y<<", pt_last.z="<<pt_last.z<<endl;
			while (cur_pt < points) {
				vec3 pt2 = pt + n_theta * rad_cur;
			
				if (pt2.x > ptx_max){
					pt2.x = ptx_max;
					edge_hit = 1;
					//cout<<"X edge hit at pt2.x="<<pt2.x<<", pt2.y="<<pt2.y<<", pt2.z="<<pt2.z<<endl;		
				}
				else if (pt2.x < 0){
					pt2.x = 0;
					edge_hit = 1;
					//cout<<"X edge hit at pt2.x="<<pt2.x<<", pt2.y="<<pt2.y<<", pt2.z="<<pt2.z<<endl;		
				}

				if (pt2.y > pty_max){
					pt2.y = pty_max;
					edge_hit = 1;
					//cout<<"Y edge hit at pt2.x="<<pt2.x<<", pt2.y="<<pt2.y<<", pt2.z="<<pt2.z<<endl;		
				}
				else if (pt2.y < 0){
					pt2.y = 0;
					edge_hit = 1;
					//cout<<"Y edge hit at pt2.x="<<pt2.x<<", pt2.y="<<pt2.y<<", pt2.z="<<pt2.z<<endl;		
				}

				if (pt2.z > ptz_max){
					pt2.z = ptz_max;
					edge_hit = 1;
					//cout<<"Z edge hit at pt2.x="<<pt2.x<<", pt2.y="<<pt2.y<<", pt2.z="<<pt2.z<<endl;		
				}
				else if (pt2.z < 0){
					pt2.z = 0;
					edge_hit = 1;
					//cout<<"Z edge hit at pt2.x="<<pt2.x<<", pt2.y="<<pt2.y<<", pt2.z="<<pt2.z<<endl;		
				}			

				vec3 d = pt2 - pt;
				d.x *= S[0];
				d.y *= S[1];
				d.z *= S[2];
				dist = sqrt(d * d);
			
				int index_x = pt2.x;
				int index_y = pt2.y;
				int index_z = pt2.z;
				long int index_data = index_x * j * k + index_y * k + index_z;
				

				if (((floor(pt2.x) != floor(pt_last.x)) || (floor(pt2.y) != floor(pt_last.y)) || (floor(pt2.z) != floor(pt_last.z))) && edge_hit == 0  && (theta_i > theta_start) && (theta_i < theta_end)){ //check is the point location has changed, but the edge hasn't been hit
					//cout<<"Okay"<<endl;		
					if ((THETA_i * t_i + theta_i) == 74893) {
						cout<<"Template Value: "<<data[index_data]<<"; Label Value: "<<input_label[index_data]<<"; edge hit: "<<edge_hit<<endl;
					}
					(*D)[THETA_i * t_i * points + theta_i * points + cur_pt] = data[ index_data ]; //updates the output template
				

					if ((input_label[ index_data ] == 0) && (input_label[ last_index ] != 0)) { //finds the edge of the manually filled image	
												
						if ((*A)[THETA_i * t_i + theta_i] == 0) { //checks to make sure the value hasn't already been set. This matters when you keep running into an edge value					
							//cout<<"Edge Hit. t_i: "<<t_i<<"; theta_i: "<<theta_i<<"; pt_last.x="<<pt_last.x<<"; pt_last.y="<<pt_last.y<<"; pt_last.z="<<pt_last.z<<"; pt.x="<<pt.x<<"; pt.y="<<pt.y<<"; pt.z="<<pt.z<<endl;
							(*A)[THETA_i * t_i + theta_i] = cur_pt; //updates the index output
							(*B)[THETA_i * t_i + theta_i] = dist; //updates the edge distance output
							(*pts)[(THETA_i * t_i + theta_i) * 3 + 0] = pt2.x;
							(*pts)[(THETA_i * t_i + theta_i) * 3 + 1] = pt2.y;
							(*pts)[(THETA_i * t_i + theta_i) * 3 + 2] = pt2.z;
						}
					}
					cur_pt++;
					last_index = index_data;
				}
				else if (edge_hit == 1 && input_label[ index_data ] != 0 && (theta_i > theta_start) && (theta_i < theta_end)) {
					//cout<<"Okay"<<endl;
					if ((THETA_i * t_i + theta_i) == 74893) {
						cout<<"Template Value: "<<data[index_data]<<"; Label Value: "<<input_label[index_data]<<"; edge hit: "<<edge_hit<<endl;
					}
					if ((*A)[THETA_i * t_i + theta_i] == 0) { //checks to make sure the value hasn't already been set. This matters when you keep running into an edge value				
						//if (t_i < 2 && theta_i > 75 && theta_i < 105) {						
						//	cout<<"Hit edge at N="<<t_i<<", M="<<theta_i<<", pt_last.x="<<pt_last.x<<", pt_last.y="<<pt_last.y<<", pt_last.z="<<pt_last.z<<", pt.x="<<pt.x<<", pt.y="<<pt.y<<", pt.z="<<pt.z<<" Index="<<cur_pt<<endl;			
						//}
						//cout<<"Image Edge Hit. t_i: "<<t_i<<"; theta_i: "<<theta_i<<"; pt_last.x="<<pt_last.x<<"; pt_last.y="<<pt_last.y<<"; pt_last.z="<<pt_last.z<<"; pt.x="<<pt.x<<"; pt.y="<<pt.y<<"; pt.z="<<pt.z<<endl;
						(*D)[THETA_i * t_i * points + theta_i * points + cur_pt] = 1000; 
						(*A)[THETA_i * t_i + theta_i] = cur_pt; //updates the index output
						(*B)[THETA_i * t_i + theta_i] = dist; //updates the edge distance output
						(*pts)[(THETA_i * t_i + theta_i) * 3 + 0] = pt2.x;
						(*pts)[(THETA_i * t_i + theta_i) * 3 + 1] = pt2.y;
						(*pts)[(THETA_i * t_i + theta_i) * 3 + 2] = pt2.z;
					}
				
				}			
				pt_last = pt2;
				rad_cur += delta_rad;
				edge_hit = 0;	

				if (rad_cur > delta_rad * 1000) { //If the radial is extending out of the bounds of the image this allows the program to exit.
					cur_pt = points;
				}
				//cout<<"cur_pt="<<cur_pt<<endl;	
			}
			
		}
	}
}

void CRSpline::get_full_radial_values(int T_i, float* T,  int THETA_i, float* THETA, int axial, int i, int j, int k, float* data,  
	int points, int S_i, float* S, int image_type, int* a, int* b, float **D) {

	(*a) = T_i * THETA_i;
	(*b) = points;
	(*D) = new float[T_i * THETA_i * points]; //this is the template output
	
	int total_points = T_i * THETA_i * points;
	int cur_pt = 0;
	long int last_index;
	float delta_rad = 0.1;
	float rad_cur;
	int ptx_max = i - 1;
	int pty_max = j - 1;
	int ptz_max = k - 1;
	int edge_hit = 0;
	int edge_found = 0;
	
	//Initialize the array template to some value
	for (int z = 0; z < total_points; z++){
		(*D)[z] = 0;
	}

	for (int t_i = 0; t_i < T_i; t_i++) {
		vec3 pt = CRSpline::GetInterpolatedSplinePoint(T[t_i]);
		long int last_index = pt.x * j * k + pt.y * k + pt.z;
		vec3 v = vec3(0,1,0);
		if (axial != 1) {
			vec3 v = CRSpline::GetInterpolatedSplineTangent(T[t_i]);			
		}
		vec3 normal;		

		if (image_type == 1) { //checks if the image is functional or anatomical since they are captured differently
			normal = v.cross(vec3(-1, 0, 0)); //this sets the normal to be in the ventral/dorsal direction for functional images (front = ventral, back = dorsal)
		}			
		else {	
			normal = v.cross(vec3(0, 0, -1)); //this sets the normal to be in the ventral/dorsal direction for anatomical images (front = ventral, back = dorsal)
		}					
		normal.normalize(); 
		float r;
		vec3 max_point = pt;	
		vec3 pt_last = vec3(0,0,0);
		for (int theta_i = 0; theta_i < THETA_i; theta_i++) { 
			float rot_angle;				
			if (image_type == 1){
				rot_angle = -THETA[theta_i]; //functional images need to be rotated in the opposite direction to the anatomical images
			}
			else{
				rot_angle = THETA[theta_i]; 
			}
			vec3 n_theta = v.rotatev(normal, rot_angle);
			n_theta.normalize();
			rad_cur = 0;
			cur_pt = 0;
			float dist = 0;

			while (cur_pt < points) {
				vec3 pt2 = pt + n_theta * rad_cur;
				
				if (pt2.x > ptx_max){
					pt2.x = ptx_max;
					edge_hit = 1;
				}
				else if (pt2.x < 0){
					pt2.x = 0;
					edge_hit = 1;
				}

				if (pt2.y > pty_max){
					pt2.y = pty_max;
					edge_hit = 1;
				}
				else if (pt2.y < 0){
					pt2.y = 0;
					edge_hit = 1;
				}

				if (pt2.z > ptz_max){
					pt2.z = ptz_max;
					edge_hit = 1;
				}
				else if (pt2.z < 0){
					pt2.z = 0;
					edge_hit = 1;
				}			

				vec3 d = pt2 - pt;
				d.x *= S[0];
				d.y *= S[1];
				d.z *= S[2];
				dist = sqrt(d * d);
				
				int index_x = pt2.x;
				int index_y = pt2.y;
				int index_z = pt2.z;
				long int index_data = index_x * j * k + index_y * k + index_z;
				if (((floor(pt2.x) != floor(pt_last.x)) || (floor(pt2.y) != floor(pt_last.y)) || (floor(pt2.z) != floor(pt_last.z))) && edge_hit == 0){
					(*D)[THETA_i * t_i * points + theta_i * points + cur_pt] = data[ index_data ]; //updates the output template
					//cout<<"Data Point:"<<data[ index_data ]<<endl;
					cur_pt++;
					last_index = index_data;
				}	
				else if (edge_hit == 1 && edge_found == 0) {
					(*D)[THETA_i * t_i * points + theta_i * points + cur_pt] = 1000; 
					edge_found == 1;
				}
				edge_hit = 0;		
				pt_last = pt2;
				rad_cur += delta_rad;		
				if (rad_cur > delta_rad * 1000) { //If the radial is extending out of the bounds of the image this allows the program to exit.
					cur_pt = points;
				}
			}
			edge_found = 0;
		}
	}
}

void CRSpline::create_new_edge_overlay_from_index_values(int T_i, float* T, int THETA_i, float* THETA, int S_i, float* S, int image_type, int axial, int newEdge_i, int newEdge_j, int* newEdge, 
	int l, int m, int n, unsigned short* label_ip) {
	

	float r;
	float rad_cur;
	float rad_incr = 0.1;
	vec3 pt2;
	int ptx_max = l - 1;
	int pty_max = m - 1;
	int ptz_max = n - 1;
	int cur_pt;
	int indexValue = 0;
	float index;

	for (int t_i = 0; t_i < T_i; t_i++) {		
		//cout<<"New Edge Overlay, t_i="<<t_i<<endl;		
		vec3 pt = CRSpline::GetInterpolatedSplinePoint(T[t_i]);
		vec3 v = vec3(0,1,0);
		if (axial != 1) {
			vec3 v = CRSpline::GetInterpolatedSplineTangent(T[t_i]);			
		}
		vec3 normal;		
		if (image_type == 1) { //checks if the image is functional or anatomical since they are captured differently
			normal = v.cross(vec3(-1, 0, 0)); //this sets the normal to be in the ventral/dorsal direction for functional images (front = ventral, back = dorsal)
		}			
		else {	
			normal = v.cross(vec3(0, 0, -1)); //this sets the normal to be in the ventral/dorsal direction for anatomical images (front = ventral, back = dorsal)
		}			
		normal.normalize();
		vec3 pt_last = vec3(0,0,0);		
		for (int theta_i = 0; theta_i < THETA_i; theta_i++) { 
			float rot_angle;				

			if (image_type == 1){
				rot_angle = -THETA[theta_i]; //functional images need to be rotated in the opposite direction to the anatomical images
			}
			else{
				rot_angle = THETA[theta_i]; 
			}
			vec3 n_theta = v.rotatev(normal, rot_angle); //Points n_theta vector in the proper direction
			n_theta.normalize();
			//Look up the edgeIndex for given t and theta
			indexValue = t_i * THETA_i + theta_i;
			if (indexValue > (newEdge_i * newEdge_j - 1)){
				indexValue = 0;
			}
			index = newEdge[indexValue];
			
			//cout<<"theta_i="<<theta_i<<", index="<<index<<endl;
			
			
			rad_cur = 0;
			cur_pt = 0;
			float dist = 0;
			int infinite_loop_counter = 0;
			//cout<<"pt_last.x="<<pt_last.x<<", pt_last.y="<<pt_last.y<<", pt_last.z="<<pt_last.z<<endl;
			while (cur_pt < (index + 1)) {
				vec3 pt2 = pt + n_theta * rad_cur;
				
				if (pt2.x > ptx_max){
					pt2.x = ptx_max;
				}
				else if (pt2.x < 0){
					pt2.x = 0;
				}

				if (pt2.y > pty_max){
					pt2.y = pty_max;
				}
				else if (pt2.y < 0){
					pt2.y = 0;
				}

				if (pt2.z > ptz_max){
					pt2.z = ptz_max;
				}
				else if (pt2.z < 0){
					pt2.z = 0;
				}			

				//vec3 d = pt2 - pt;
				//d.x *= S[0];
				//d.y *= S[1];
				//d.z *= S[2];
				//dist = sqrt(d * d);
				
				int index_x = pt2.x;
				int index_y = pt2.y;
				int index_z = pt2.z;

				if ((floor(pt2.x) != floor(pt_last.x)) || (floor(pt2.y) != floor(pt_last.y)) || (floor(pt2.z) != floor(pt_last.z))){
					cur_pt++;
				}			
				rad_cur += rad_incr;		
				pt_last = pt2;
				//cout<<"cur_pt="<<cur_pt<<endl;
				if (index > 0){			
					//cout<<"indexValue="<<indexValue<<", index="<<index<<", pt.x="<<pt.x<<", pt.y="<<pt.y<<", pt.z="<<pt.z<<", pt2.x="<<pt2.x<<", pt2.y="<<pt2.y<<", pt2.z="<<pt2.z<<endl;
				}
				infinite_loop_counter += 1;
				if (infinite_loop_counter > 1000){ //occurs at the edge of the screen, in which case a line will be drawn to the edge of the screen
					cout<<"Point hit at edge of screen. Drawing line to edge. pt2.x="<<pt2.x<<", pt2.y="<<pt2.y<<", pt2.z="<<pt2.z<<endl;
					break;
				}
			}
			//cout<<"t_i="<<t_i<<", theta_i="<<theta_i<<endl;
			if (index > 0){			
				CRSpline::fill_spine_label(pt.x, pt.y, pt.z, pt_last.x, pt_last.y, pt_last.z, l, m, n, 1, label_ip);			
			}
		}
	}
}	


//Given a list of relative distances (t), angles (theta), and radii from the spline, create a filled image of the spine
void CRSpline::create_new_edge_overlay_from_distance_values(int T_i, float* T, int THETA_i, float* THETA, int S_i, float* S, int image_type, int axial, int a, int b, float* grid, 
		int l, int m, int n, unsigned short* label_ip) {    
	
	double PI = 3.141592;
	float r;
	float rad_cur;
	float rad_incr = 0.1;
	vec3 pt2;

	for (int t_i = 0; t_i < T_i; t_i++) {
		//vec3 pt2 = vec3(1, 1, 1);		
		vec3 pt = CRSpline::GetInterpolatedSplinePoint(T[t_i]);
		vec3 v = vec3(0,1,0);
		if (axial != 1) {
			vec3 v = CRSpline::GetInterpolatedSplineTangent(T[t_i]);			
		}	
		vec3 normal;		
		if (image_type == 1) { //checks if the image is functional or anatomical since they are captured differently
			normal = v.cross(vec3(-1, 0, 0)); //this sets the normal to be in the ventral/dorsal direction for functional images (front = ventral, back = dorsal)
		}			
		else {	
			normal = v.cross(vec3(0, 0, -1)); //this sets the normal to be in the ventral/dorsal direction for anatomical images (front = ventral, back = dorsal)
		}					
		normal.normalize();
		for (int theta_i = 0; theta_i < THETA_i; theta_i++) { 
			float rot_angle;				
			if (image_type == 1){
				rot_angle = -THETA[theta_i]; //functional images need to be rotated in the opposite direction to the anatomical images
			}
			else{
				rot_angle = THETA[theta_i]; 
			}
			vec3 n_theta = v.rotatev(normal, rot_angle); //Points n_theta vector in the proper direction
			n_theta.normalize();
			//Look up the current radius for given theta			
			r = grid[t_i * THETA_i + theta_i];
			//NEW CODE STARTS HERE			
			rad_cur = 0;
			float dist = 0;
			while (dist < r) {
				vec3 pt2 = pt + n_theta * rad_cur;
				vec3 d = pt2 - pt;
				d.x *= S[0];
				d.y *= S[1];
				d.z *= S[2];
				dist = sqrt(d * d);
				rad_cur += rad_incr;
			}
			float target_rad = rad_cur - rad_incr; //subtract rad_incr to take into account the last addition which occured before the while loop check
			vec3 pt2 = pt + n_theta * target_rad;
			//END OF NEW CODE

			//vec3 pt2 = pt + n_theta * r; //This line was replaced by the one above
			
			CRSpline::fill_spine_label(pt.x, pt.y, pt.z, pt2.x, pt2.y, pt2.z, l, m, n, 1, label_ip);		
		}
	}
}	

void CRSpline::draw_normals(int T_i, float* T, int image_type, int divider, int l, int m, int n, unsigned short* label_ip) {    
	
	int norm_multiplier = 10;

	for (int t_i = 0; t_i < T_i; t_i++) {		
		vec3 pt = CRSpline::GetInterpolatedSplinePoint(T[t_i]);
		vec3 v = CRSpline::GetInterpolatedSplineTangent(T[t_i]);			
		vec3 normal;		
		if (image_type == 1) { //checks if the image is functional or anatomical since they are captured differently
			normal = v.cross(vec3(-1, 0, 0)); //this sets the normal to be in the ventral/dorsal direction for functional images (front = ventral, back = dorsal)
		}			
		else {	
			normal = v.cross(vec3(0, 0, -1)); //this sets the normal to be in the ventral/dorsal direction for anatomical images (front = ventral, back = dorsal)
		}			
		normal.normalize();
		
		vec3 pt2 = pt + normal * norm_multiplier;

		if ((t_i % divider) == 0) {
			//cout<<"t_i="<<t_i<<", pt.x="<<pt.x<<", pt.y="<<pt.y<<", pt.z="<<pt.z<<", pt2.x="<<pt2.x<<", pt2.y="<<pt2.y<<", pt2.z="<<pt2.z<<endl;
			CRSpline::fill_spine_label(pt.x, pt.y, pt.z, pt2.x, pt2.y, pt2.z, l, m, n, 1, label_ip);			
		}
	}
}

void CRSpline::calculate_center_from_filled(int T_i, float* T, int THETA_i, float* THETA, int image_type, int axial, int l, int m, int n, unsigned short* label_ip, int* pts_i, int* pts_j, float **pts) {
	//Calculates the new center point from a set of radials. If a single radial in one slice is not present, it will return 0,0,0 as the new center point for that slice.
	//Calculates the point as the average of all points in the plane of each slice
	float rad_incr = 0.1;
	float x_coord;
	float y_coord;
	float z_coord;
	int count = 0;
	float avg_x;
	float avg_y;
	float avg_z;

	(*pts_i) = T_i;
	(*pts_j) = 3;
	(*pts) = new float[T_i * 3];
	
	vec3 normal;		
	for (int t_i = 0; t_i < T_i; t_i++) {
		x_coord = 0;
		y_coord = 0;
		z_coord = 0;
		vec3 pt = CRSpline::GetInterpolatedSplinePoint(T[t_i]);
		vec3 v = vec3(0,1,0);
		if (axial != 1) {
			vec3 v = CRSpline::GetInterpolatedSplineTangent(T[t_i]);			
		}
		if (image_type == 1) { //checks if the image is functional or anatomical since they are captured differently
			normal = v.cross(vec3(-1, 0, 0)); //this sets the normal to be in the ventral/dorsal direction for functional images (front = ventral, back = dorsal)
		}			
		else {	
			normal = v.cross(vec3(0, 0, -1)); //this sets the normal to be in the ventral/dorsal direction for anatomical images (front = ventral, back = dorsal)
		}			
		normal.normalize();	 
		count = 0;
		float rad_cur = 0;
		for (int theta_i = 0; theta_i < THETA_i; theta_i++) {
			vec3 n_theta = v.rotatev(normal, THETA[theta_i]); 
			n_theta.normalize();
			rad_cur = 0;
			vec3 pt2 = pt + n_theta * rad_cur;
			int value = label_ip[IND(l, m, n, pt2.x, pt2.y, pt2.z)];
			while (value != 0) {
				vec3 pt2 = pt + n_theta * rad_cur;
				rad_cur += rad_incr;
				pt2 = pt + n_theta * rad_cur;
				value = label_ip[IND(l, m, n, pt2.x, pt2.y, pt2.z)];
				//cout<<"pt.x="<<pt.x<<", pt.y="<<pt.y<<", pt.z="<<pt.z<<", pt2.x="<<pt2.x<<", pt2.y="<<pt2.y<<", pt2.z="<<pt2.z<<", value="<<value<<endl;
			}

			x_coord += pt2.x;
			y_coord += pt2.y;
			z_coord += pt2.z;
			
			if (rad_cur > 5){			
				count++;
			}
		}			
		
		//cout<<"Count = "<<count<<endl;
		if (count == (THETA_i)){		
			avg_x = x_coord / (THETA_i);
			avg_y = y_coord / (THETA_i);
			avg_z = z_coord / (THETA_i);
			//cout<<"t_i="<<t_i<<", x="<<avg_x<<", y="<<avg_y<<", z="<<avg_z<<endl;
		}
		else{
			avg_x = 0;
			avg_y = 0;
			avg_z = 0;
		}

		(*pts)[t_i * 3 + 0] = avg_x;
		(*pts)[t_i * 3 + 1] = avg_y;
		(*pts)[t_i * 3 + 2] = avg_z;
	}
}

void CRSpline::get_points_at_edge_of_surface(int T_i, float* T, int THETA_i, float* THETA, int image_type, int l, int m, int n, unsigned short* label_ip, int* pts_i, int* pts_j, float **pts) {

	float rad_cur;
	float rad_incr = 0.1;
	vec3 pt2;
	int ptx_max = l - 1;
	int pty_max = m - 1;
	int ptz_max = n - 1;
	int last_x = 0;
	int last_y = 0;
	int last_z = 0;
	int value = 0;

	(*pts_i) = T_i * THETA_i;
	(*pts_j) = 3;
	(*pts) = new float[T_i * THETA_i * 3];

	for (int t_i = 0; t_i < T_i; t_i++) {		
		vec3 pt = CRSpline::GetInterpolatedSplinePoint(T[t_i]);
		vec3 v = CRSpline::GetInterpolatedSplineTangent(T[t_i]);			
		vec3 normal;		
		if (image_type == 1) { //checks if the image is functional or anatomical since they are captured differently
			normal = v.cross(vec3(-1, 0, 0)); //this sets the normal to be in the ventral/dorsal direction for functional images (front = ventral, back = dorsal)
		}			
		else {	
			normal = v.cross(vec3(0, 0, -1)); //this sets the normal to be in the ventral/dorsal direction for anatomical images (front = ventral, back = dorsal)
		}			
		normal.normalize();

		for (int theta_i = 0; theta_i < THETA_i; theta_i++) { 
			float rot_angle;				

			if (image_type == 1){
				rot_angle = -THETA[theta_i]; //functional images need to be rotated in the opposite direction to the anatomical images
			}
			else{
				rot_angle = THETA[theta_i]; 
			}
			vec3 n_theta = v.rotatev(normal, rot_angle); //Points n_theta vector in the proper direction
			n_theta.normalize();

			
			rad_cur = 0;
			//cout<<"pt_last.x="<<pt_last.x<<", pt_last.y="<<pt_last.y<<", pt_last.z="<<pt_last.z<<endl;
			vec3 pt2 = pt + n_theta * rad_cur;
			int index_x = pt2.x;
			int index_y = pt2.y;
			int index_z = pt2.z;
			long int index_data = index_x * m * n + index_y * n + index_z;
			value = label_ip[ index_data ]; 
			while (value > 0) {
				
				vec3 pt2 = pt + n_theta * rad_cur;

				if (pt2.x > ptx_max){
					pt2.x = ptx_max;
				}
				else if (pt2.x < 0){
					pt2.x = 0;
				}

				if (pt2.y > pty_max){
					pt2.y = pty_max;
				}
				else if (pt2.y < 0){
					pt2.y = 0;
				}

				if (pt2.z > ptz_max){
					pt2.z = ptz_max;
				}
				else if (pt2.z < 0){
					pt2.z = 0;
				}			
				
				int index_x = pt2.x;
				int index_y = pt2.y;
				int index_z = pt2.z;
				long int index_data = index_x * m * n + index_y * n + index_z;
				value = label_ip[ index_data ]; 

				last_x = index_x;
				last_y = index_y;	
				last_z = index_z;
				//cout<<"t_i="<<t_i<<", theta_i="<<theta_i<<", pt.x="<<pt.x<<", pt.y="<<pt.y<<", pt.z="<<pt.z<<", pt2.x="<<pt2.x<<", pt2.y="<<pt2.y<<", pt2.z="<<pt2.z<<endl;
				//cout<<"value="<<value<<", rad_cur="<<rad_cur<<endl;				
				rad_cur += rad_incr;		
			}
			
			if (rad_cur > 1){
				(*pts)[(t_i * THETA_i + theta_i) * 3 + 0] = last_x;
				(*pts)[(t_i * THETA_i + theta_i) * 3 + 1] = last_y;
				(*pts)[(t_i * THETA_i + theta_i) * 3 + 2] = last_z;
			}
			else {
				(*pts)[(t_i * THETA_i + theta_i) * 3 + 0] = 0;
				(*pts)[(t_i * THETA_i + theta_i) * 3 + 1] = 0;
				(*pts)[(t_i * THETA_i + theta_i) * 3 + 2] = 0;
			}
		}
	}
}	


void CRSpline::get_radii_from_edgeindex(int T_i, float* T, int THETA_i, float* THETA, int S_i, float* S, int image_type, int axial, int newEdge_i, int newEdge_j, int* newEdge, int* a, int* b, float **D) {

	//converts a set of index points for an edge into a set of radial distance values (mm) from the center point
	
	(*a) = T_i;
	(*b) = THETA_i;
	(*D) = new float[T_i * THETA_i];

	float r;
	float rad_cur;
	float rad_incr = 0.1;
	vec3 pt2;
	int cur_pt;
	int indexValue = 0;
	float index;

	for (int t_i = 0; t_i < T_i; t_i++) {		
		vec3 pt = CRSpline::GetInterpolatedSplinePoint(T[t_i]);
		vec3 v = vec3(0,1,0);
		if (axial != 1) {
			vec3 v = CRSpline::GetInterpolatedSplineTangent(T[t_i]);			
		}
		vec3 normal;		
		if (image_type == 1) { //checks if the image is functional or anatomical since they are captured differently
			normal = v.cross(vec3(-1, 0, 0)); //this sets the normal to be in the ventral/dorsal direction for functional images (front = ventral, back = dorsal)
		}			
		else {	
			normal = v.cross(vec3(0, 0, -1)); //this sets the normal to be in the ventral/dorsal direction for anatomical images (front = ventral, back = dorsal)
		}			
		normal.normalize();
		vec3 pt_last = vec3(0,0,0);		
		for (int theta_i = 0; theta_i < THETA_i; theta_i++) { 
			float rot_angle;				

			if (image_type == 1){
				rot_angle = -THETA[theta_i]; //functional images need to be rotated in the opposite direction to the anatomical images
			}
			else{
				rot_angle = THETA[theta_i]; 
			}
			vec3 n_theta = v.rotatev(normal, rot_angle); //Points n_theta vector in the proper direction
			n_theta.normalize();
			//Look up the edgeIndex for given t and theta
			indexValue = t_i * THETA_i + theta_i;
			if (indexValue > (newEdge_i * newEdge_j - 1)){
				indexValue = 0;
			}
			index = newEdge[indexValue];
			
			rad_cur = 0;
			cur_pt = 0;
			float dist = 0;
			//cout<<"pt_last.x="<<pt_last.x<<", pt_last.y="<<pt_last.y<<", pt_last.z="<<pt_last.z<<endl;
			while (cur_pt < (index)) {
				vec3 pt2 = pt + n_theta * rad_cur;
				vec3 d = pt2 - pt;
				d.x *= S[0];
				d.y *= S[1];
				d.z *= S[2];
				dist = sqrt(d * d);
				
				int index_x = pt2.x;
				int index_y = pt2.y;
				int index_z = pt2.z;

				if ((floor(pt2.x) != floor(pt_last.x)) || (floor(pt2.y) != floor(pt_last.y)) || (floor(pt2.z) != floor(pt_last.z))){
					cur_pt++;
				}			
				rad_cur += rad_incr;		
				pt_last = pt2;
			}
			(*D)[THETA_i * t_i + theta_i] = dist;
		}
	}
}
			

void CRSpline::calculate_center_from_origin_vectors(int T_i, float* T, int originVector_i, int originVector_j, float* originVector, int S_i, float* S, int image_type, int* a, int* b, float **D){

//Given a set of r and thetha as input for each slice of the spine, calculate the new center point

	(*a) = T_i;
	(*b) = 3;
	(*D) = new float[T_i * 3];

	float rad_cur;
	float rad_incr = 0.1;
	vec3 pt2;

	for (int t_i = 0; t_i < T_i; t_i++) {		
		vec3 pt = CRSpline::GetInterpolatedSplinePoint(T[t_i]);
		vec3 v = CRSpline::GetInterpolatedSplineTangent(T[t_i]);			
		vec3 normal;		
		if (image_type == 1) { //checks if the image is functional or anatomical since they are captured differently
			normal = v.cross(vec3(-1, 0, 0)); //this sets the normal to be in the ventral/dorsal direction for functional images (front = ventral, back = dorsal)
		}			
		else {	
			normal = v.cross(vec3(0, 0, -1)); //this sets the normal to be in the ventral/dorsal direction for anatomical images (front = ventral, back = dorsal)
		}			
		normal.normalize();
		vec3 pt_last = pt;		

		float rot_angle;				
		if (image_type == 1){
			rot_angle = -originVector[t_i * 2 + 1]; //functional images need to be rotated in the opposite direction to the anatomical images
		}
		else{
			rot_angle = originVector[t_i * 2 + 1]; 
		}
		vec3 n_theta = v.rotatev(normal, rot_angle); //Points n_theta vector in the proper direction
		n_theta.normalize();
		
		float targetRadius = originVector[t_i * 2 + 0];
		rad_cur = 0;
		float dist = 0;
		cout<<"t_i="<<t_i<<", targetRad="<<targetRadius<<endl;
	
		while (dist < targetRadius) {
			vec3 pt2 = pt + n_theta * rad_cur;
			if (targetRadius > 15){
				targetRadius = 0.1; //Allows the loop to exit early
			}
			vec3 d = pt2 - pt;
			d.x *= S[0];
			d.y *= S[1];
			d.z *= S[2];
			dist = sqrt(d * d);
			
			int index_x = pt2.x;
			int index_y = pt2.y;
			int index_z = pt2.z;
	
			rad_cur += rad_incr;		
			pt_last = pt2;
			//cout<<"RadCurr="<<rad_cur<<endl;
		}
		//cout<<"pt2.x="<<pt2.x<<", pt2.y="<<pt2.y<<", pt2.z="<<pt2.z<<endl;
		cout<<"pt_last.x="<<pt_last.x<<", pt_last.y="<<pt_last.y<<", pt_last.z="<<pt_last.z<<endl;
		(*D)[t_i * 3 + 0] = pt_last.x;
		(*D)[t_i * 3 + 1] = pt_last.y;
		(*D)[t_i * 3 + 2] = pt_last.z;
	}
}


void CRSpline::calculate_center_points_from_edge_segmentation(int T_i, float* T, int THETA_i, float* THETA, int S_i, float* S, int image_type, int axial, int newEdge_i, int newEdge_j, int* newEdge, int matchPercent_i, int matchPercent_j, float* matchPercent, int l, int m, int n, unsigned short* label_ip, int* a, int* b, float **D) {
	
	//Calculates a new set of center points from the edge segmentation output, using the matchPercent output
	
	(*a) = T_i;
	(*b) = 3;
	(*D) = new float[T_i * 3];

	float r;
	float rad_cur;
	float rad_incr = 0.1;
	vec3 pt2;
	int cur_pt;
	int indexValue = 0;
	float index;
	int calcNewCenter = 1;
	float maxX = 0;
	float minX = 1000;
	float maxY = 0;
	float minY = 1000;
	float maxZ = 0;
	float minZ = 1000;
	int counter = 0;

	for (int t_i = 0; t_i < T_i; t_i++) {		
		vec3 pt = CRSpline::GetInterpolatedSplinePoint(T[t_i]);
		vec3 v = vec3(0,1,0);
		if (axial != 1) {
			vec3 v = CRSpline::GetInterpolatedSplineTangent(T[t_i]);			
		}
		vec3 normal;		
		if (image_type == 1) { //checks if the image is functional or anatomical since they are captured differently
			normal = v.cross(vec3(-1, 0, 0)); //this sets the normal to be in the ventral/dorsal direction for functional images (front = ventral, back = dorsal)
		}			
		else {	
			normal = v.cross(vec3(0, 0, -1)); //this sets the normal to be in the ventral/dorsal direction for anatomical images (front = ventral, back = dorsal)
		}			
		normal.normalize();
		vec3 pt_last = vec3(0,0,0);		
		
		for (int theta_i = 0; theta_i < THETA_i; theta_i++) {
			if (matchPercent[t_i * THETA_i + theta_i] == 0) {
				calcNewCenter = 0;
				//cout<<"Calc new center = 0"<<endl;
			}
			else {
				counter += 1;
			}
		}
		//cout<<"T_i: "<<t_i<<", non-zero radials: "<<counter<<endl;
		
		if (calcNewCenter == 1) { //Will not run if even one of the radials doesn't exist
			//cout<<"t_i="<<t_i<<"; pt.y="<<pt.y<<endl;
			for (int theta_i = 0; theta_i < THETA_i; theta_i++) {			
				float rot_angle;				

				if (image_type == 1){
					rot_angle = -THETA[theta_i]; //functional images need to be rotated in the opposite direction to the anatomical images
				}
				else{
					rot_angle = THETA[theta_i]; 
				}
				vec3 n_theta = v.rotatev(normal, rot_angle); //Points n_theta vector in the proper direction
				n_theta.normalize();
				indexValue = t_i * THETA_i + theta_i;
				if (indexValue > (newEdge_i * newEdge_j - 1)){
					indexValue = 0;
				}
				index = newEdge[indexValue];
			
				rad_cur = 0;
				cur_pt = 0;
				float dist = 0;
				//cout<<"pt_last.x="<<pt_last.x<<", pt_last.y="<<pt_last.y<<", pt_last.z="<<pt_last.z<<endl;
				while (cur_pt < (index)) {
					vec3 pt2 = pt + n_theta * rad_cur;
					//vec3 d = pt2 - pt;
					//d.x *= S[0];
					//d.y *= S[1];
					//d.z *= S[2];
					//dist = sqrt(d * d);
				
					int index_x = pt2.x;
					int index_y = pt2.y;
					int index_z = pt2.z;

					if ((floor(pt2.x) != floor(pt_last.x)) || (floor(pt2.y) != floor(pt_last.y)) || (floor(pt2.z) != floor(pt_last.z))){
						cur_pt++;
					}			
					rad_cur += rad_incr;		
					pt_last = pt2;
				}
				if (pt_last.x > maxX) {maxX = pt_last.x;}
				if (pt_last.y > maxY) {maxY = pt_last.y;}
				if (pt_last.z > maxZ) {maxZ = pt_last.z;}
				if (pt_last.x < minX) {minX = pt_last.x;}				
				if (pt_last.y < minY) {minY = pt_last.y;}				
				if (pt_last.z < minZ) {minZ = pt_last.z;}								
						
				//cout<<pt_last.z<<endl;
				
				CRSpline::fill_spine_label(pt_last.x, pt_last.y, pt_last.z, pt_last.x, pt_last.y, pt_last.z, l, m, n, 1, label_ip);
//				CRSpline::fill_spine_label(pt_last.x-1, pt_last.y-1, pt_last.z-1, pt_last.x, pt_last.y, pt_last.z, l, m, n, 1, label_ip);						
			}
			(*D)[t_i * 3 + 0] = (maxX + minX) / 2;
			(*D)[t_i * 3 + 1] = (maxY + minY) / 2;
			(*D)[t_i * 3 + 2] = (maxZ + minZ) / 2;
		}
		
		else {
			(*D)[t_i * 3 + 0] = 0;
			(*D)[t_i * 3 + 1] = 0;
			(*D)[t_i * 3 + 2] = 0;
		}

		maxX = 0;
		minX = 1000;
		maxY = 0;
		minY = 1000;
		maxZ = 0;
		minZ = 1000;
		counter = 0;
		calcNewCenter = 1;

	}
}



void CRSpline::create_flattened_spine(int T_i, float* T, int THETA_i, float* THETA, int S_i, float* S, int image_type, int radial_points, int sag_view, int i, int j, int k, float* data, int* a, int* b, float **D) {

	(*a) = T_i;
	(*b) = radial_points * 2;
	(*D) = new float[T_i * radial_points * 2];

	int axial = 1;
	int offset;
	float rad_cur;
	float rad_incr = 0.1;
	vec3 pt2;
	int cur_pt;
	int output_index;
	
	//initialize output array
	for (int a = 0; a < T_i; a++){
		for (int b = 0; b < (radial_points * 2); b++){
			(*D)[a * radial_points * 2 + b] = 0;
		}
	}
	
	for (int t_i = 0; t_i < T_i; t_i++) {		
		vec3 pt = CRSpline::GetInterpolatedSplinePoint(T[t_i]);
		vec3 v = vec3(0,1,0);
		if (axial != 1) {
			vec3 v = CRSpline::GetInterpolatedSplineTangent(T[t_i]);			
		}
		vec3 normal;		
		if (image_type == 1) { //checks if the image is functional or anatomical since they are captured differently
			normal = v.cross(vec3(-1, 0, 0)); //this sets the normal to be in the ventral/dorsal direction for functional images (front = ventral, back = dorsal)
		}			
		else {	
			normal = v.cross(vec3(0, 0, -1)); //this sets the normal to be in the ventral/dorsal direction for anatomical images (front = ventral, back = dorsal)
		}			
		normal.normalize();
		vec3 pt_last = vec3(0,0,0);		
		cout<<"t_i="<<t_i<<endl;			
		for (int theta_i = 0; theta_i < THETA_i; theta_i++) {		
			if (theta_i < 1	|| theta_i > (THETA_i - 1) || (theta_i > (THETA_i / 2 - 1) && theta_i < (THETA_i / 2 + 1))){
				//cout<<"t_i="<<t_i<<", theta_i="<<theta_i<<endl;			
				float rot_angle;				

				if (image_type == 1){
					rot_angle = -THETA[theta_i]; //functional images need to be rotated in the opposite direction to the anatomical images
				}
				else{
					rot_angle = THETA[theta_i]; 
				}
				vec3 n_theta = v.rotatev(normal, rot_angle); //Points n_theta vector in the proper direction
				n_theta.normalize();
		
				rad_cur = 0;
				cur_pt = 0;
				float dist = 0;
				//cout<<"pt_last.x="<<pt_last.x<<", pt_last.y="<<pt_last.y<<", pt_last.z="<<pt_last.z<<endl;
				while (cur_pt < (radial_points - 1)) {
					//cout<<"Cur pt="<<cur_pt<<endl;				
					vec3 pt2 = pt + n_theta * rad_cur;
					
					int index_x = pt2.x;
					int index_y = pt2.y;
					int index_z = pt2.z;

					if ((floor(pt2.x) != floor(pt_last.x)) || (floor(pt2.y) != floor(pt_last.y)) || (floor(pt2.z) != floor(pt_last.z))){
						cout<<"t_i="<<t_i<<", theta_i="<<theta_i<<", index_x="<<index_x<<", index_y="<<index_y<<", index_z="<<index_z<<endl;
						if (sag_view == 1) {
							offset = radial_points + floor(pt2.x) - floor(pt.x);
						}
						else {
							offset = radial_points + floor(pt2.z) - floor(pt.z);
						}
						long int index_data = index_x * j * k + index_y * k + index_z;
						output_index = t_i * radial_points * 2 + offset;
						(*D)[output_index] += data[ index_data ];
						cur_pt++;
					}			
					rad_cur += rad_incr;		
					pt_last = pt2;
				}
			}
		}
	}
}

// Given two relative locations down the spline, calculates the number of non-zero voxels in the region
void CRSpline::get_segment_voxels(int T_i, float* T,  int THETA_i, float* THETA, int axial, int image_type, int t_start, int t_end, int S_i, float* S, int d, int e,  int f, unsigned short* input_label, int r, int s, int t, unsigned short* output_label, int* voxels_i, float** voxels) {

	(*voxels_i) = 1;
	(*voxels) = new float[1];	
	int ptx_max = d - 1;
	int pty_max = e - 1;
	int ptz_max = f - 1;
	float rad_incr = .1;
	float rad_cur;
	
	for (int t_i = 0; t_i < T_i; t_i++){		
		int within_region = 0;						
		if ((t_i >= t_start) && (t_i <= t_end)){
			within_region = 1;
			//cout<<"Within_region"<<endl;
		}
		if (within_region == 1){
			vec3 pt = CRSpline::GetInterpolatedSplinePoint(T[t_i]);
			vec3 v;
			if (axial == 0) {v = CRSpline::GetInterpolatedSplineTangent(T[t_i]);}
			else {v = vec3(0,1,0);}

			vec3 normal;		
			if (image_type == 1) { //checks if the image is functional or anatomical since they are captured differently
				normal = v.cross(vec3(-1, 0, 0)); //this sets the normal to be in the ventral/dorsal direction for functional images (front = ventral, back = dorsal)
			}			
			else {	
				normal = v.cross(vec3(0, 0, -1)); //this sets the normal to be in the ventral/dorsal direction for anatomical images (front = ventral, back = dorsal)
			}
			normal.normalize(); 
			//cout<<"T="<<T[t_i]<<"pt="<<pt.x<<" "<<pt.y<<" "<<pt.z<<endl;
			vec3 pt_last = vec3(0,0,0);		
			for (int theta_i = 0; theta_i < THETA_i; theta_i++) {
				float rot_angle;				
				if (image_type == 1){
					rot_angle = -THETA[theta_i]; //functional images need to be rotated in the opposite direction to the anatomical images
				}
				else{
					rot_angle = THETA[theta_i]; 
				}
				vec3 n_theta = v.rotatev(normal, rot_angle); 
				n_theta.normalize();
				rad_cur = 0;
				float dist = 0;
				while (rad_cur < 100) {
					vec3 pt2 = pt + n_theta * rad_cur;
					int index_x = pt2.x;
					int index_y = pt2.y;
					int index_z = pt2.z;	

					//Checks to see if it's a new point
					if ((floor(pt2.x) != floor(pt_last.x)) || (floor(pt2.y) != floor(pt_last.y)) || (floor(pt2.z) != floor(pt_last.z))){
						int value = input_label[IND(d, e, f, pt2.x, pt2.y, pt2.z)];
						int curr_value = output_label[IND(r, s, t, pt2.x, pt2.y, pt2.z)];
						if (value == 1 && curr_value == 0) {
							output_label[IND(r, s, t, pt2.x, pt2.y, pt2.z)] = 1;
							(*voxels)[0] += 1;
						}
					}

					if (pt2.x > ptx_max){break;}
					else if (pt2.x < 0){break;}
					if (pt2.y > pty_max){break;}
					else if (pt2.y < 0){break;}
					if (pt2.z > ptz_max){break;}
					else if (pt2.z < 0){break;}						

					rad_cur += rad_incr;
				}
			}
		}
	}
}


void CRSpline::create_coordinate_system_slices(int T_i, float* T, int THETA_i, float* THETA, int S_i, float* S, int image_type, int radius, int slice, 
	int l, int m, int n, unsigned short* label_ip) {
	

	float r;
	float rad_cur;
	float rad_incr = 0.1;
	double PI = 3.141592;
	vec3 pt2;
	int ptx_max = l - 1;
	int pty_max = m - 1;
	int ptz_max = n - 1;
	int cur_pt;
	int indexValue = 0;
	int slice_spacing = int(T_i * 1.0 / 5);
	float index;
	int within_range = 0;
	int within_extra_slice_range = 0;

	for (int t_i = 0; t_i < T_i; t_i++) {		
		if (t_i == 0 || t_i == (slice + slice_spacing * 0) || t_i == (slice + slice_spacing * 1) || t_i == (slice + slice_spacing * 2) || t_i == (slice + slice_spacing * 3) || t_i == (0 + 1) || t_i == (slice + slice_spacing * 0 + 1) || t_i == (slice + slice_spacing * 1 + 1) || t_i == (slice + slice_spacing * 2 + 1) || t_i == (slice + slice_spacing * 3 + 1)){
			within_range = 1;
		}
		if (t_i == (slice + slice_spacing * 0) || t_i == (slice + slice_spacing * 1) || t_i == (slice + slice_spacing * 2) || t_i == (slice + slice_spacing * 3) || t_i == (slice + slice_spacing * 0 + 1) || t_i == (slice + slice_spacing * 1 + 1) || t_i == (slice + slice_spacing * 2 + 1) || t_i == (slice + slice_spacing * 3 + 1)){
			within_extra_slice_range = 1;
		}
		
		if (within_range == 1){
			//cout<<"New Edge Overlay, t_i="<<t_i<<endl;		
			vec3 pt = CRSpline::GetInterpolatedSplinePoint(T[t_i]);
			vec3 v = CRSpline::GetInterpolatedSplineTangent(T[t_i]);			
			vec3 normal;		
			if (image_type == 1) { //checks if the image is functional or anatomical since they are captured differently
				normal = v.cross(vec3(-1, 0, 0)); //this sets the normal to be in the ventral/dorsal direction for functional images (front = ventral, back = dorsal)
			}			
			else {	
				normal = v.cross(vec3(0, 0, -1)); //this sets the normal to be in the ventral/dorsal direction for anatomical images (front = ventral, back = dorsal)
			}			
			normal.normalize();
			vec3 pt_last = vec3(0,0,0);		
			for (int theta_i = 0; theta_i < THETA_i; theta_i++) { 
				float rot_angle;				

				if (image_type == 1){
					rot_angle = -THETA[theta_i]; //functional images need to be rotated in the opposite direction to the anatomical images
				}
				else{
					rot_angle = THETA[theta_i]; 
				}
				vec3 n_theta = v.rotatev(normal, rot_angle); //Points n_theta vector in the proper direction
				n_theta.normalize();
				//Look up the edgeIndex for given t and theta
			
				//cout<<"theta_i="<<theta_i<<", index="<<index<<endl;
			
				rad_cur = 0;
				cur_pt = 0;
				float dist = 0;
				int infinite_loop_counter = 0;

				while (dist < radius) {
					vec3 pt2 = pt + n_theta * rad_cur;
				
					if (pt2.x > ptx_max){
						pt2.x = ptx_max;
					}
					else if (pt2.x < 0){
						pt2.x = 0;
					}

					if (pt2.y > pty_max){
						pt2.y = pty_max;
					}
					else if (pt2.y < 0){
						pt2.y = 0;
					}

					if (pt2.z > ptz_max){
						pt2.z = ptz_max;
					}
					else if (pt2.z < 0){
						pt2.z = 0;
					}			

					vec3 d = pt2 - pt;
					d.x *= S[0];
					d.y *= S[1];
					d.z *= S[2];
					dist = sqrt(d * d);
				
					int index_x = pt2.x;
					int index_y = pt2.y;
					int index_z = pt2.z;

					if ((floor(pt2.x) != floor(pt_last.x)) || (floor(pt2.y) != floor(pt_last.y)) || (floor(pt2.z) != floor(pt_last.z))){
						cur_pt++;
					}			
					rad_cur += rad_incr;		
					pt_last = pt2;
					//cout<<"cur_pt="<<cur_pt<<endl;
					if (index > 0){			
						//cout<<"indexValue="<<indexValue<<", index="<<index<<", pt.x="<<pt.x<<", pt.y="<<pt.y<<", pt.z="<<pt.z<<", pt2.x="<<pt2.x<<", pt2.y="<<pt2.y<<", pt2.z="<<pt2.z<<endl;
					}
					infinite_loop_counter += 1;
					if (infinite_loop_counter > 1000){ //occurs at the edge of the screen, in which case a line will be drawn to the edge of the screen
						cout<<"Point hit at edge of screen. Drawing line to edge. pt2.x="<<pt2.x<<", pt2.y="<<pt2.y<<", pt2.z="<<pt2.z<<endl;
						break;
					}
				}
				//cout<<"t_i="<<t_i<<", theta_i="<<theta_i<<endl;
				if (t_i == 0 || t_i == 1){
					CRSpline::fill_spine_label(pt.x, pt.y, pt.z, pt_last.x, pt_last.y, pt_last.z, l, m, n, 2, label_ip);			
				}

				if (within_extra_slice_range == 1 && theta_i >= 0 && theta_i < int(1 * THETA_i / 4)){
					CRSpline::fill_spine_label(pt.x, pt.y, pt.z, pt_last.x, pt_last.y, pt_last.z, l, m, n, 2, label_ip);			
				}
				else if (within_extra_slice_range == 1 && theta_i >= int(1 * THETA_i / 4)  && theta_i < int(2 * THETA_i / 4)){
					CRSpline::fill_spine_label(pt.x, pt.y, pt.z, pt_last.x, pt_last.y, pt_last.z, l, m, n, 3, label_ip);			
				}
				else if (within_extra_slice_range == 1 && theta_i >= int(2 * THETA_i / 4)  && theta_i < int(3 * THETA_i / 4)){
					CRSpline::fill_spine_label(pt.x, pt.y, pt.z, pt_last.x, pt_last.y, pt_last.z, l, m, n, 4, label_ip);			
				}
				else if (within_extra_slice_range == 1 && theta_i >= int(3 * THETA_i / 4)  && theta_i < int(4 * THETA_i / 4)){
					CRSpline::fill_spine_label(pt.x, pt.y, pt.z, pt_last.x, pt_last.y, pt_last.z, l, m, n, 5, label_ip);			
				}
			}
			within_range = 0;
			within_extra_slice_range = 0;
		}
	}
}	

void CRSpline::get_spine_flexion_extension(int T_i, float* T, int image_type, int* i, float** p) {

	double PI = 3.141592;
	(*i) = T_i;
	(*p) = new float[T_i];
		
	for (int t_i = 0; t_i < T_i; t_i++) {		
		vec3 pt = CRSpline::GetInterpolatedSplinePoint(T[t_i]);
		vec3 v = CRSpline::GetInterpolatedSplineTangent(T[t_i]);			
		vec3 normal;		
		vec3 horizontal_line;
		if (image_type == 1) { //checks if the image is functional or anatomical since they are captured differently
			normal = v.cross(vec3(-1, 0, 0)); //this sets the normal to be in the ventral/dorsal direction for functional images (front = ventral, back = dorsal)
			horizontal_line = vec3(0,0,-1);
		}			
		else {	
			normal = v.cross(vec3(0, 0, -1)); //this sets the normal to be in the ventral/dorsal direction for anatomical images (front = ventral, back = dorsal)
			horizontal_line = vec3(-1,0,0);
		}			

		normal.normalize();
		cout<<"t_i: "<<t_i<<"; normal x/y/z: "<<normal.x<<"/"<<normal.y<<"/"<<normal.z<<"; horizontal_line x/y/z: "<<horizontal_line.x<<"/"<<horizontal_line.y<<"/"<<horizontal_line.z<<endl;
		float dot_prod = normal.x * horizontal_line.x + normal.y * horizontal_line.y + normal.z * horizontal_line.z;
		float mag_horizontal = sqrt(horizontal_line.x * horizontal_line.x + horizontal_line.y * horizontal_line.y + horizontal_line.z * horizontal_line.z);
		float mag_normal = sqrt(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
		float cos_angle = dot_prod / (mag_horizontal * mag_normal);
		float angle = acos(cos_angle) * 360 / (2 * PI);		
		cout<<"t_i: "<<t_i<<"; angle: "<<angle<<endl;
		if (normal.y < 0){ (*p)[t_i] = 0 - angle;}
		else{(*p)[t_i] = angle;}
	}
}


void CRSpline::convert_edge_index_distances_to_mm_distances(int T_i, float* T, int THETA_i, float* THETA, int S_i, float* S, int image_type, int axial, int l, int m, int n, unsigned short* label_ip, int newEdge_i, int newEdge_j, int* newEdge, int newEdge_mm_i, int newEdge_mm_j, float* newEdge_mm) {
	
	float r;
	float rad_cur;
	float rad_incr = 0.1;
	vec3 pt2;
	int ptx_max = l - 1;
	int pty_max = m - 1;
	int ptz_max = n - 1;
	int cur_pt;
	int indexValue = 0;
	float index;

	for (int t_i = 0; t_i < T_i; t_i++) {		
		vec3 pt = CRSpline::GetInterpolatedSplinePoint(T[t_i]);
		vec3 v = vec3(0,1,0);
		if (axial != 1) {
			vec3 v = CRSpline::GetInterpolatedSplineTangent(T[t_i]);			
		}
		vec3 normal;		
		if (image_type == 1) { //checks if the image is functional or anatomical since they are captured differently
			normal = v.cross(vec3(-1, 0, 0)); //this sets the normal to be in the ventral/dorsal direction for functional images (front = ventral, back = dorsal)
		}			
		else {	
			normal = v.cross(vec3(0, 0, -1)); //this sets the normal to be in the ventral/dorsal direction for anatomical images (front = ventral, back = dorsal)
		}			
		normal.normalize();
		vec3 pt_last = vec3(0,0,0);		
		for (int theta_i = 0; theta_i < THETA_i; theta_i++) { 
			float rot_angle;				

			if (image_type == 1){
				rot_angle = -THETA[theta_i]; //functional images need to be rotated in the opposite direction to the anatomical images
			}
			else{
				rot_angle = THETA[theta_i]; 
			}
			vec3 n_theta = v.rotatev(normal, rot_angle); //Points n_theta vector in the proper direction
			n_theta.normalize();
			//Look up the edgeIndex for given t and theta
			indexValue = t_i * THETA_i + theta_i;
			if (indexValue > (newEdge_i * newEdge_j - 1)){
				indexValue = 0;
			}
			index = newEdge[indexValue];
			
			//cout<<"theta_i="<<theta_i<<", index="<<index<<endl;
			
			rad_cur = 0;
			cur_pt = 0;
			float dist = 0;
			int infinite_loop_counter = 0;
			//cout<<"pt_last.x="<<pt_last.x<<", pt_last.y="<<pt_last.y<<", pt_last.z="<<pt_last.z<<endl;
			while (cur_pt < (index)) {
				vec3 pt2 = pt + n_theta * rad_cur;
				
				if (pt2.x > ptx_max){
					pt2.x = ptx_max;
				}
				else if (pt2.x < 0){
					pt2.x = 0;
				}

				if (pt2.y > pty_max){
					pt2.y = pty_max;
				}
				else if (pt2.y < 0){
					pt2.y = 0;
				}

				if (pt2.z > ptz_max){
					pt2.z = ptz_max;
				}
				else if (pt2.z < 0){
					pt2.z = 0;
				}			

				vec3 d = pt2 - pt;
				d.x *= S[0];
				d.y *= S[1];
				d.z *= S[2];
				dist = sqrt(d * d);
				
				int index_x = pt2.x;
				int index_y = pt2.y;
				int index_z = pt2.z;

				if ((floor(pt2.x) != floor(pt_last.x)) || (floor(pt2.y) != floor(pt_last.y)) || (floor(pt2.z) != floor(pt_last.z))){
					cur_pt++;
				}			
				rad_cur += rad_incr;		
				pt_last = pt2;
			
				infinite_loop_counter += 1;
				if (infinite_loop_counter > 1000){ //occurs at the edge of the screen, in which case a line will be drawn to the edge of the screen
					cout<<"Point hit at edge of screen. Drawing line to edge. pt2.x="<<pt2.x<<", pt2.y="<<pt2.y<<", pt2.z="<<pt2.z<<endl;
					break;
				}
			}
			//cout<<"t_i="<<t_i<<", theta_i="<<theta_i<<endl;
			if (index > 0){			
				newEdge_mm[indexValue] = dist;
			}
		}
	}
}	

void CRSpline::convert_edge_mm_distances_to_index_distances(int T_i, float* T, int THETA_i, float* THETA, int S_i, float* S, int image_type, int axial, int l, int m, int n, unsigned short* label_ip, int newEdge_mm_i, int newEdge_mm_j, float* newEdge_mm, int newEdge_i, int newEdge_j, int* newEdge) {
	
	float r;
	float rad_cur;
	float rad_incr = 0.1;
	vec3 pt2;
	int ptx_max = l - 1;
	int pty_max = m - 1;
	int ptz_max = n - 1;
	int cur_index;
	int indexValue = 0;
	float target_distance;

	for (int t_i = 0; t_i < T_i; t_i++) {		
		vec3 pt = CRSpline::GetInterpolatedSplinePoint(T[t_i]);
		vec3 v = vec3(0,1,0);
		if (axial != 1) {
			vec3 v = CRSpline::GetInterpolatedSplineTangent(T[t_i]);			
		}
		vec3 normal;		
		if (image_type == 1) { //checks if the image is functional or anatomical since they are captured differently
			normal = v.cross(vec3(-1, 0, 0)); //this sets the normal to be in the ventral/dorsal direction for functional images (front = ventral, back = dorsal)
		}			
		else {	
			normal = v.cross(vec3(0, 0, -1)); //this sets the normal to be in the ventral/dorsal direction for anatomical images (front = ventral, back = dorsal)
		}			
		normal.normalize();
		vec3 pt_last = vec3(0,0,0);		
		for (int theta_i = 0; theta_i < THETA_i; theta_i++) { 
			float rot_angle;				

			if (image_type == 1){
				rot_angle = -THETA[theta_i]; //functional images need to be rotated in the opposite direction to the anatomical images
			}
			else{
				rot_angle = THETA[theta_i]; 
			}
			vec3 n_theta = v.rotatev(normal, rot_angle); //Points n_theta vector in the proper direction
			n_theta.normalize();
			//Look up the edgeIndex for given t and theta
			indexValue = t_i * THETA_i + theta_i;
			if (indexValue > (newEdge_mm_i * newEdge_mm_j - 1)){
				indexValue = 0;
			}
			target_distance = newEdge_mm[indexValue];
			
			//cout<<"theta_i="<<theta_i<<", target_dist="<<target_distance<<endl;
			
			rad_cur = 0;
			cur_index = 0;
			float cur_dist = 0;
			int infinite_loop_counter = 0;
			//cout<<"pt_last.x="<<pt_last.x<<", pt_last.y="<<pt_last.y<<", pt_last.z="<<pt_last.z<<endl;
			while (cur_dist < target_distance) {
				vec3 pt2 = pt + n_theta * rad_cur;
				
				if (pt2.x > ptx_max){
					pt2.x = ptx_max;
				}
				else if (pt2.x < 0){
					pt2.x = 0;
				}

				if (pt2.y > pty_max){
					pt2.y = pty_max;
				}
				else if (pt2.y < 0){
					pt2.y = 0;
				}

				if (pt2.z > ptz_max){
					pt2.z = ptz_max;
				}
				else if (pt2.z < 0){
					pt2.z = 0;
				}			

				vec3 d = pt2 - pt;
				d.x *= S[0];
				d.y *= S[1];
				d.z *= S[2];
				cur_dist = sqrt(d * d);
				
				int index_x = pt2.x;
				int index_y = pt2.y;
				int index_z = pt2.z;

				if ((floor(pt2.x) != floor(pt_last.x)) || (floor(pt2.y) != floor(pt_last.y)) || (floor(pt2.z) != floor(pt_last.z))){
					cur_index++;
				}			
				rad_cur += rad_incr;
				pt_last = pt2;
			
				infinite_loop_counter += 1;
				if (infinite_loop_counter > 1000){ //occurs at the edge of the screen, in which case a line will be drawn to the edge of the screen
					cout<<"Point hit at edge of screen. Drawing line to edge. pt2.x="<<pt2.x<<", pt2.y="<<pt2.y<<", pt2.z="<<pt2.z<<endl;
					break;
				}
			}
			//cout<<"t_i="<<t_i<<", theta_i="<<theta_i<<endl;
			if (target_distance > 0){			
				newEdge[indexValue] = cur_index;
			}
		}
	}
}	

void CRSpline::fill_partially_segmented_spine(int T_i, float* T,  int THETA_i, float* THETA, int image_type, int t_start, int t_end, int S_i, float* S, int d, int e,  int f, unsigned short* input_label, int r, int s, int t, unsigned short* output_label) {

	int max_radius = 20;
	int ptx_max = d - 1;
	int pty_max = e - 1;
	int ptz_max = f - 1;
	float rad_incr = .1;
	float rad_cur;
	
	for (int t_i = 0; t_i < T_i; t_i++){		
		int within_region = 0;						
		if ((t_i >= t_start) && (t_i <= t_end)){
			within_region = 1;
			//cout<<"Within_region"<<endl;
		}
		if (within_region == 1){
			vec3 pt = CRSpline::GetInterpolatedSplinePoint(T[t_i]);
			vec3 v;
			v = CRSpline::GetInterpolatedSplineTangent(T[t_i]);
			vec3 normal;		
			if (image_type == 1) { //checks if the image is functional or anatomical since they are captured differently
				normal = v.cross(vec3(-1, 0, 0)); //this sets the normal to be in the ventral/dorsal direction for functional images (front = ventral, back = dorsal)
			}			
			else {	
				normal = v.cross(vec3(0, 0, -1)); //this sets the normal to be in the ventral/dorsal direction for anatomical images (front = ventral, back = dorsal)
			}
			normal.normalize(); 
			//cout<<"T="<<T[t_i]<<"pt="<<pt.x<<" "<<pt.y<<" "<<pt.z<<endl;
			vec3 pt_last = vec3(0,0,0);		
			for (int theta_i = 0; theta_i < THETA_i; theta_i++) {
				float rot_angle;				
				if (image_type == 1){
					rot_angle = -THETA[theta_i]; //functional images need to be rotated in the opposite direction to the anatomical images
				}
				else{
					rot_angle = THETA[theta_i]; 
				}
				vec3 n_theta = v.rotatev(normal, rot_angle); 
				n_theta.normalize();
				rad_cur = 0;
				float dist = 0;
				int on_edge = 0;
				while (rad_cur < max_radius) {
					vec3 pt2 = pt + n_theta * rad_cur;
					int index_x = pt2.x;
					int index_y = pt2.y;
					int index_z = pt2.z;	

					//Checks to see if it's a new point
					if ((floor(pt2.x) != floor(pt_last.x)) || (floor(pt2.y) != floor(pt_last.y)) || (floor(pt2.z) != floor(pt_last.z))){
						int value = input_label[IND(d, e, f, pt2.x, pt2.y, pt2.z)];
						
						if ((value == 0 && on_edge == 0) || (value == 1 && on_edge == 1)) {
							output_label[IND(r, s, t, pt2.x, pt2.y, pt2.z)] = 1;
						}
						
						if (value == 1 && on_edge == 0){
							on_edge = 1;
						}
					}

					if (pt2.x > ptx_max){break;}
					else if (pt2.x < 0){break;}
					if (pt2.y > pty_max){break;}
					else if (pt2.y < 0){break;}
					if (pt2.z > ptz_max){break;}
					else if (pt2.z < 0){break;}						

					rad_cur += rad_incr;
				}
			}
		}
	}
}

void CRSpline::compare_two_volumes(int T_i, float* T, int THETA_i, float* THETA, int t_start, int t_end, int S_i, float* S, int image_type, int l, int m, int n, unsigned short* label_ip_1, int l2, int m2, int n2, unsigned short* label_ip_2, int* pts_i, float **pts) {

	float rad_cur;
	float rad_incr = 0.1;
	vec3 pt2;
	int ptx_max = l - 1;
	int pty_max = m - 1;
	int ptz_max = n - 1;
	int last_x = 0;
	int last_y = 0;
	int last_z = 0;
	int value = 0;

	float dist1 = 0;
	float dist2 = 0;

	int num_pos = 0;
	int num_neg = 0;
	float sum_pos = 0;
	float sum_neg = 0;
	float sum_rel_pos = 0;
	float sum_rel_neg = 0;
	float avg_pos = 0;
	float avg_neg = 0;
	float max_pos = 0;
	float max_neg = 0;
	float min_pos = 0;
	float min_neg = 0;


	(*pts_i) = 8;
	(*pts) = new float[8];

	for (int t_i = 0; t_i < T_i; t_i++) {	

		if ((t_i >= t_start) && (t_i <= t_end)){
			vec3 pt = CRSpline::GetInterpolatedSplinePoint(T[t_i]);
			vec3 v = CRSpline::GetInterpolatedSplineTangent(T[t_i]);			
			vec3 normal;		
			if (image_type == 1) { //checks if the image is functional or anatomical since they are captured differently
				normal = v.cross(vec3(-1, 0, 0)); //this sets the normal to be in the ventral/dorsal direction for functional images (front = ventral, back = dorsal)
			}			
			else {	
				normal = v.cross(vec3(0, 0, -1)); //this sets the normal to be in the ventral/dorsal direction for anatomical images (front = ventral, back = dorsal)
			}			
			normal.normalize();

			for (int theta_i = 0; theta_i < THETA_i; theta_i++) { 
				float rot_angle;				

				if (image_type == 1){
					rot_angle = -THETA[theta_i]; //functional images need to be rotated in the opposite direction to the anatomical images
				}
				else{
					rot_angle = THETA[theta_i]; 
				}
				vec3 n_theta = v.rotatev(normal, rot_angle); //Points n_theta vector in the proper direction
				n_theta.normalize();

			
				//Calculate the distance to the edge for the manual (first) image
				rad_cur = 0;
				vec3 pt2 = pt + n_theta * rad_cur;
				int index_x = pt2.x;
				int index_y = pt2.y;
				int index_z = pt2.z;
				long int index_data = index_x * m * n + index_y * n + index_z;

				value = label_ip_1[ index_data ];
				while (value > 0) {
					vec3 pt2 = pt + n_theta * rad_cur;
					if (pt2.x > ptx_max){
						pt2.x = ptx_max;
					}
					else if (pt2.x < 0){
						pt2.x = 0;
					}
					if (pt2.y > pty_max){
						pt2.y = pty_max;
					}
					else if (pt2.y < 0){
						pt2.y = 0;
					}
					if (pt2.z > ptz_max){
						pt2.z = ptz_max;
					}
					else if (pt2.z < 0){
						pt2.z = 0;
					}			
					int index_x = pt2.x;
					int index_y = pt2.y;
					int index_z = pt2.z;
					long int index_data = index_x * m * n + index_y * n + index_z;
					value = label_ip_1[ index_data ]; 

					last_x = index_x;
					last_y = index_y;	
					last_z = index_z;
					//cout<<"t_i="<<t_i<<", theta_i="<<theta_i<<", pt.x="<<pt.x<<", pt.y="<<pt.y<<", pt.z="<<pt.z<<", pt2.x="<<pt2.x<<", pt2.y="<<pt2.y<<", pt2.z="<<pt2.z<<endl;
					//cout<<"value="<<value<<", rad_cur="<<rad_cur<<endl;				

					vec3 d = pt2 - pt;
					dist1 = 0;
					d.x *= S[0];
					d.y *= S[1];
					d.z *= S[2];
					dist1 = sqrt(d * d);

					rad_cur += rad_incr;		
				}

				//Do the same for the second (segmented) image
				rad_cur = 0;
				pt2 = pt + n_theta * rad_cur;
				index_x = pt2.x;
				index_y = pt2.y;
				index_z = pt2.z;
				index_data = index_x * m * n + index_y * n + index_z;
				value = label_ip_2[ index_data ];
				while (value > 0) {
					vec3 pt2 = pt + n_theta * rad_cur;
					if (pt2.x > ptx_max){
						pt2.x = ptx_max;
					}
					else if (pt2.x < 0){
						pt2.x = 0;
					}
					if (pt2.y > pty_max){
						pt2.y = pty_max;
					}
					else if (pt2.y < 0){
						pt2.y = 0;
					}
					if (pt2.z > ptz_max){
						pt2.z = ptz_max;
					}
					else if (pt2.z < 0){
						pt2.z = 0;
					}			
					int index_x = pt2.x;
					int index_y = pt2.y;
					int index_z = pt2.z;
					long int index_data = index_x * m * n + index_y * n + index_z;
					value = label_ip_2[ index_data ]; 

					last_x = index_x;
					last_y = index_y;	
					last_z = index_z;
					//cout<<"t_i="<<t_i<<", theta_i="<<theta_i<<", pt.x="<<pt.x<<", pt.y="<<pt.y<<", pt.z="<<pt.z<<", pt2.x="<<pt2.x<<", pt2.y="<<pt2.y<<", pt2.z="<<pt2.z<<endl;
					//cout<<"value="<<value<<", rad_cur="<<rad_cur<<endl;				

					vec3 d = pt2 - pt;
					dist2 = 0;
					d.x *= S[0];
					d.y *= S[1];
					d.z *= S[2];
					dist2 = sqrt(d * d);

					rad_cur += rad_incr;		
				}
			
				if(dist1 > 0 && dist2 > 0){
					float dist_diff = dist2 - dist1;
					if(dist_diff > 0){
						num_pos += 1;
						sum_pos += dist_diff;
						sum_rel_pos += dist_diff / dist1; //the difference in the distance divided by the manual segmentation difference
						if(dist_diff > max_pos){
							max_pos = dist_diff;
						}
						if(dist_diff < min_pos){
							min_pos = dist_diff;
						}
					
					}
					else{
						num_neg += 1;
						sum_neg += dist_diff;
						sum_rel_neg += dist_diff / dist1; //the difference in the distance divided by the manual segmentation difference
						if(dist_diff < max_neg){
							max_neg = dist_diff;
						}
						if(dist_diff > min_neg){
							min_neg = dist_diff;
						}
					}
				}

				else{
					cout<<"DISTANCE ERROR"<<endl;
				}
			

			}
		}
	}
	avg_pos = 1.0 * sum_pos / num_pos;
	avg_neg = 1.0 * sum_neg / num_neg;
	(*pts)[0] = num_pos;
	(*pts)[1] = avg_pos;
	(*pts)[2] = max_pos;
	(*pts)[3] = 1.0 * sum_rel_pos / num_pos;
	(*pts)[4] = num_neg;
	(*pts)[5] = avg_neg;
	(*pts)[6] = max_neg;
	(*pts)[7] = 1.0 * sum_rel_neg / num_neg;
}	


//takes a mask and set of t values as end-points and returns a new mask which is only between those end points
void CRSpline::create_submask(int T_i, float* T,  int THETA_i, float* THETA, int image_type, int t_start, int t_end, int S_i, float* S, int d, int e,  int f, unsigned short* input_label, int r, int s, int t, unsigned short* output_label) {
	int max_radius = 50;
	int ptx_max = d - 1;
	int pty_max = e - 1;
	int ptz_max = f - 1;
	float rad_incr = .1;
	float rad_cur;
	
	for (int t_i = 0; t_i < T_i; t_i++){		
		int within_region = 0;						
		if ((t_i >= t_start) && (t_i <= t_end)){
			within_region = 1;
			//cout<<"Within_region"<<endl;
		}
		if (within_region == 1){
			vec3 pt = CRSpline::GetInterpolatedSplinePoint(T[t_i]);
			vec3 v;
			v = CRSpline::GetInterpolatedSplineTangent(T[t_i]);
			vec3 normal;		
			if (image_type == 1) { //checks if the image is functional or anatomical since they are captured differently
				normal = v.cross(vec3(-1, 0, 0)); //this sets the normal to be in the ventral/dorsal direction for functional images (front = ventral, back = dorsal)
			}			
			else {	
				normal = v.cross(vec3(0, 0, -1)); //this sets the normal to be in the ventral/dorsal direction for anatomical images (front = ventral, back = dorsal)
			}
			normal.normalize(); 
			//cout<<"T="<<T[t_i]<<"pt="<<pt.x<<" "<<pt.y<<" "<<pt.z<<endl;
			vec3 pt_last = vec3(0,0,0);		
			for (int theta_i = 0; theta_i < THETA_i; theta_i++) {
				float rot_angle;				
				if (image_type == 1){
					rot_angle = -THETA[theta_i]; //functional images need to be rotated in the opposite direction to the anatomical images
				}
				else{
					rot_angle = THETA[theta_i]; 
				}
				vec3 n_theta = v.rotatev(normal, rot_angle); 
				n_theta.normalize();
				rad_cur = 0;
				float dist = 0;
				int on_edge = 0;
				while (rad_cur < max_radius) {
					vec3 pt2 = pt + n_theta * rad_cur;
					int index_x = pt2.x;
					int index_y = pt2.y;
					int index_z = pt2.z;	

					//Checks to see if it's a new point
					if ((floor(pt2.x) != floor(pt_last.x)) || (floor(pt2.y) != floor(pt_last.y)) || (floor(pt2.z) != floor(pt_last.z))){
						output_label[IND(r, s, t, pt2.x, pt2.y, pt2.z)] = input_label[IND(d, e, f, pt2.x, pt2.y, pt2.z)];
					}

					if (pt2.x > ptx_max){break;}
					else if (pt2.x < 0){break;}
					if (pt2.y > pty_max){break;}
					else if (pt2.y < 0){break;}
					if (pt2.z > ptz_max){break;}
					else if (pt2.z < 0){break;}						

					rad_cur += rad_incr;
				}
			}
		}
	}
}


void CRSpline::create_submask_from_filled_mask(int T_i, float* T,  int THETA_i, float* THETA, int image_type, int t_start, int t_end, int S_i, float* S, int d, int e,  int f, unsigned short* input_label) {
	//takes a label mask and deletes any areas that are not within a selected region

	int max_radius = 50;
	int ptx_max = d - 1;
	int pty_max = e - 1;
	int ptz_max = f - 1;
	float rad_incr = .1;
	float rad_cur;
	
	for (int t_i = 0; t_i < T_i; t_i++){		
		int within_region = 1;						
		if ((t_i <= t_start) || (t_i >= t_end)){
			within_region = 0;
			//cout<<"Within_region"<<endl;
		}
		if (within_region == 0){
			vec3 pt = CRSpline::GetInterpolatedSplinePoint(T[t_i]);
			vec3 v;
			v = CRSpline::GetInterpolatedSplineTangent(T[t_i]);
			vec3 normal;		
			if (image_type == 1) { //checks if the image is functional or anatomical since they are captured differently
				normal = v.cross(vec3(-1, 0, 0)); //this sets the normal to be in the ventral/dorsal direction for functional images (front = ventral, back = dorsal)
			}			
			else {	
				normal = v.cross(vec3(0, 0, -1)); //this sets the normal to be in the ventral/dorsal direction for anatomical images (front = ventral, back = dorsal)
			}
			normal.normalize(); 
			//cout<<"T="<<T[t_i]<<"pt="<<pt.x<<" "<<pt.y<<" "<<pt.z<<endl;
			vec3 pt_last = vec3(0,0,0);		
			for (int theta_i = 0; theta_i < THETA_i; theta_i++) {
				float rot_angle;				
				if (image_type == 1){
					rot_angle = -THETA[theta_i]; //functional images need to be rotated in the opposite direction to the anatomical images
				}
				else{
					rot_angle = THETA[theta_i]; 
				}
				vec3 n_theta = v.rotatev(normal, rot_angle); 
				n_theta.normalize();
				rad_cur = 0;
				float dist = 0;
				int on_edge = 0;
				while (rad_cur < max_radius) {
					vec3 pt2 = pt + n_theta * rad_cur;
					int index_x = pt2.x;
					int index_y = pt2.y;
					int index_z = pt2.z;	

					//Checks to see if it's a new point
					if ((floor(pt2.x) != floor(pt_last.x)) || (floor(pt2.y) != floor(pt_last.y)) || (floor(pt2.z) != floor(pt_last.z))){
						input_label[IND(d, e, f, pt2.x, pt2.y, pt2.z)] = 0;
					}

					if (pt2.x > ptx_max){break;}
					else if (pt2.x < 0){break;}
					if (pt2.y > pty_max){break;}
					else if (pt2.y < 0){break;}
					if (pt2.z > ptz_max){break;}
					else if (pt2.z < 0){break;}						

					rad_cur += rad_incr;
				}
			}
		}
		else{
			cout<<"Outside region. t: "<<t_i<<endl;
		}
	}
}


//uses the spline to create an output mask of the spline
void CRSpline::create_spline_mask(int T_i, float* T,  int image_type, int t_start, int t_end, int S_i, float* S, int r, int s, int t, unsigned short* output_label) {
	
	for (int t_i = 0; t_i < T_i; t_i++){		
		int within_region = 0;						
		if ((t_i >= t_start) && (t_i <= t_end)){
			within_region = 1;
			//cout<<"Within_region"<<endl;
		}
		if (within_region == 1){
			vec3 pt = CRSpline::GetInterpolatedSplinePoint(T[t_i]);
			output_label[IND(r, s, t, pt.x, pt.y, pt.z)] = 2;
			
		}
	}
}


void CRSpline::fill_partially_segmented_outlined_spine(int T_i, float* T,  int THETA_i, float* THETA, int image_type, int S_i, float* S, int d, int e,  int f, unsigned short* input_label, int r, int s, int t, unsigned short* output_label) {

	int max_radius = 20;
	int ptx_max = d - 1;
	int pty_max = e - 1;
	int ptz_max = f - 1;
	float rad_incr = .1;
	float rad_cur;
	
	for (int t_i = 0; t_i < T_i; t_i++){
		vec3 pt = CRSpline::GetInterpolatedSplinePoint(T[t_i]);
		vec3 v;
		v = CRSpline::GetInterpolatedSplineTangent(T[t_i]);
		vec3 normal;		
		if (image_type == 1) { //checks if the image is functional or anatomical since they are captured differently
			normal = v.cross(vec3(-1, 0, 0)); //this sets the normal to be in the ventral/dorsal direction for functional images (front = ventral, back = dorsal)
		}			
		else {	
			normal = v.cross(vec3(0, 0, -1)); //this sets the normal to be in the ventral/dorsal direction for anatomical images (front = ventral, back = dorsal)
		}
		normal.normalize(); 
		//cout<<"T="<<T[t_i]<<"pt="<<pt.x<<" "<<pt.y<<" "<<pt.z<<endl;


		//check to see if this value of t_i has a completely enclose segmentation to be filled
		int start_filling = 1;
		vec3 pt_last = vec3(0,0,0);		
		for (int theta_i = 0; theta_i < THETA_i; theta_i++) {
			int temp_enclosed = 0;
			float rot_angle;				
			if (image_type == 1){
				rot_angle = -THETA[theta_i]; //functional images need to be rotated in the opposite direction to the anatomical images
			}
			else{
				rot_angle = THETA[theta_i]; 
			}
			vec3 n_theta = v.rotatev(normal, rot_angle); 
			n_theta.normalize();
			rad_cur = 0;
			float dist = 0;
			int on_edge = 0;
			while (rad_cur < max_radius) {
				vec3 pt2 = pt + n_theta * rad_cur;
				//int index_x = pt2.x;
				//int index_y = pt2.y;
				//int index_z = pt2.z;	

				//Checks to see if it's a new point
				if ((floor(pt2.x) != floor(pt_last.x)) || (floor(pt2.y) != floor(pt_last.y)) || (floor(pt2.z) != floor(pt_last.z))){
					int value = input_label[IND(d, e, f, pt2.x, pt2.y, pt2.z)];
					
					if (value == 1){
						temp_enclosed = 1;
						break;			
					}
				}
				
				rad_cur += rad_incr;
			}
			if(rad_cur >= max_radius && temp_enclosed == 0){
				start_filling = 0;
				cout<<"Incomplete at: t = "<<t_i<<endl;
				break;
			}
				
		}



		if(start_filling == 1){
			cout<<"Filling at: t = "<<t_i<<endl;
			vec3 pt_last = vec3(0,0,0);		
			for (int theta_i = 0; theta_i < THETA_i; theta_i++) {
				float rot_angle;				
				if (image_type == 1){
					rot_angle = -THETA[theta_i]; //functional images need to be rotated in the opposite direction to the anatomical images
				}
				else{
					rot_angle = THETA[theta_i]; 
				}
				vec3 n_theta = v.rotatev(normal, rot_angle); 
				n_theta.normalize();
				rad_cur = 0;
				float dist = 0;
				int on_edge = 0;
				while (rad_cur < max_radius) {
					vec3 pt2 = pt + n_theta * rad_cur;
					int index_x = pt2.x;
					int index_y = pt2.y;
					int index_z = pt2.z;	

					//Checks to see if it's a new point
					if ((floor(pt2.x) != floor(pt_last.x)) || (floor(pt2.y) != floor(pt_last.y)) || (floor(pt2.z) != floor(pt_last.z))){
						int value = input_label[IND(d, e, f, pt2.x, pt2.y, pt2.z)];
					
						if ((value == 0 && on_edge == 0) || (value == 1 && on_edge == 1)) {
							output_label[IND(r, s, t, pt2.x, pt2.y, pt2.z)] = 1;
						}
					
						if (value == 1 && on_edge == 0){
							on_edge = 1;
						}
					}

					if (pt2.x > ptx_max){break;}
					else if (pt2.x < 0){break;}
					if (pt2.y > pty_max){break;}
					else if (pt2.y < 0){break;}
					if (pt2.z > ptz_max){break;}
					else if (pt2.z < 0){break;}						

					rad_cur += rad_incr;
				}
			}
		}
	
	}
}

void CRSpline::get_normal_to_plane(int T_i, float* T, int THETA_i, float* THETA, int image_type, int t_start, int t_end, int* j, int* k, float **pts) {
	(*j) = 4;
	(*k) = 3;
	(*pts) = new float[4 * 3];

	for (int t_i = 0; t_i < T_i; t_i++){

		if (t_i == t_start || t_i == t_end){
			vec3 pt = CRSpline::GetInterpolatedSplinePoint(T[t_i]);
			vec3 normal;
			normal = CRSpline::GetInterpolatedSplineTangent(T[t_i]);

			normal.normalize(); 
			cout<<"Normal: x/y/z: "<<normal.x<<" / "<<normal.y<<" / "<<normal.z<<endl;
			vec3 test_pt = pt + normal * 10;
			cout<<"Test Normal Pt: x/y/z: "<<test_pt.x<<" / "<<test_pt.y<<" / "<<test_pt.z<<endl;

			if (t_i == t_start){
				(*pts)[0] = pt.x;
				(*pts)[1] = pt.y;
				(*pts)[2] = pt.z;
				(*pts)[3] = normal.x;
				(*pts)[4] = normal.y;
				(*pts)[5] = normal.z;
				
			}
			else{
				(*pts)[6] = pt.x;
				(*pts)[7] = pt.y;
				(*pts)[8] = pt.z;
				(*pts)[9] = normal.x;
				(*pts)[10] = normal.y;
				(*pts)[11] = normal.z;
			}
		}
	}
}


void CRSpline::calculate_axial_coords(int T_i, float* T, int r, int s, int t, unsigned short* output_label, int* pts_i, int* pts_j, float **pts) {
	//returns the x and z co-ordinates of the spline at each axial slice

	int y_max = 500;
	int found = 0;
	int t_min, t_max, t_mid;
	(*pts_i) = y_max;
	(*pts_j) = 2;
	(*pts) = new float[y_max * 2];

	for(int a = 0; a < y_max; a++){
		(*pts)[a * 2 + 0] = 0;
		(*pts)[a * 2 + 1] = 0;
	}		
		

	for (int t_i = 0; t_i < T_i; t_i++){
		vec3 pt = CRSpline::GetInterpolatedSplinePoint(T[t_i]);
		//cout<<"Point: "<<pt.x<<" "<<pt.y<<" "<<pt.z<<endl;
		if ((*pts)[int(pt.y) * 2 + 0] < 1){
			(*pts)[int(pt.y) * 2 + 0] = pt.x;
			(*pts)[int(pt.y) * 2 + 1] = pt.z;
			//cout<<"["<<pt.x<<", "<<pt.z<<"]"<<endl;
		}
		output_label[IND(r, s, t, pt.x, pt.y, pt.z)] = 1;
	}
	


}

