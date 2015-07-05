#ifndef __SPLINE_H__
#define __SPLINE_H__

#include "vec3.h"
#include <vector>

class CRSpline {
public:

	// Constructors and destructor
	CRSpline();
	CRSpline(const CRSpline&);
	~CRSpline();

	// Operations
	void AddSplinePoint(const vec3& v);
	void add_spline_points(int i, int j, float* P);
	void update_spline_points(int i, int j, float* P);
	vec3 GetInterpolatedSplinePoint(float t); // t = 0...1; 0=vp[0] ... 1=vp[max]
	vec3 GetInterpolatedSplineTangent(float t); // t = 0...1; 0=vp[0] ... 1=vp[max]
	void get_interpolated_spline_point(float t, int* i, float** p);
	void get_interpolated_spline_point(int i, float* T, int* j, int* k,
			float** pts);
	void get_interpolated_spline_tangent(float t, int* i, float** p);
	void get_interpolated_spline_tangent(int i, float* T, int* j, int* k,
			float** pts);
	void get_interpolated_spline_distance(int i, float* T, int S_i, float* S, int* V_i, float** V);
	float get_interpolated_spline_distance(float T,	int S_i, float* S);
	void get_relative_spline_distance(int MM_i, float* MM, int S_i, float* S, int* V_i, float** V);
	
	void find_spline_proj(int T_i, float* T, int S_i, float* S,
			int P_i, int P_j, float* P, int* V_i, float** V,
			int* A_i, float **A, int* B_i, float **B);
	int GetNumPoints();
	vec3& GetNthPoint(int n);
	void get_control_points(int* j, int* k, float** pts);
	
	float sum_line(float fx, float fy, float fz, float fxx, float fyy,
			float fzz, int i, int j, int k, float* data);
	// sums along a pipe. with radius r.
	float sum_line_pipe(float fx, float fy, float fz, float fxx, float fyy,
			float fzz, int i, int j, int k, float* data, int l, int m, int n,
			unsigned short* label_ip, float r = 0, int method = 1,
			long int *count = 0);
	float sum_spline(int i, int j, int k, float* data, int N = 50);
	float sum_spline_pipe(int i, int j, int k, float* data, int l, int m, int n,
			unsigned short* label_ip, float r = 0, int N = 50, int method = 1);
	float minimize_pipe(int i, int j, int k, float* data, int l, int m, int n,
			unsigned short* label_ip, float r = 0, int N = 50, int method = 1,
			int opt_method = 1);
	float minimize_pipe_adam(int i, int j, int k, float* data, int l, int m, int n,
			unsigned short* label_ip, float r = 0, int N = 50, int method = 1,
			int opt_method = 1);
	void draw_line(int x, int y, int z, int xx, int yy, int zz, int i, int j,
			int k, float* data_ip, float value, int r = 0);
	void draw_spline(int i, int j, int k, float* data_ip, float value,
			int r = 0, int N = 50);

	static vec3 Eq(float t, const vec3& p1, const vec3& p2, const vec3& p3,
			const vec3& p4);
	static vec3 d_dt__Eq(float t, const vec3& p1, const vec3& p2,
			const vec3& p3, const vec3& p4);

	float find_line(float fx, float fy, float fz, float fxx, float fyy,
			float fzz,
			int l, int m, int n, unsigned short* label_ip);
	void extract_spine_cylinder_coord(int i, float* T,
			int j, float* THETA,
			int l, int m, int n, unsigned short* label_ip,
			int* a, int* b, float **D);

	void parse_spine_cylinder_coord(int T_i, float* T,
			int THETA_i, float* THETA,
			int i, int j, int k, float* data,
			float min_r, float max_r, int method,
			int* a, int* b, float **D);

	float parse_line(float fx, float fy, float fz,
			float fxx, float fyy, float fzz,
			int i, int j, int k, float* data, int method);

	void parse_spine_cylinder_coord_adam(int T_i, float* T, int THETA_i,
		float* THETA, int i, int j, int k, float* data, float min_r,
		float max_r, int method, int* a, int* b, float **D);

	void get_spline_normal(int T_i, float* T, int* a, float **Output);

	float parse_line_zero_holes(float fx, float fy, float fz, float fxx, float fyy,
		float fzz, int i, int j, int k, float* data, int l, int m, int n, 
		unsigned short* label_ip, float search_r, int method, float min_value);

	void parse_spine_cylinder_coord_find_radius(int T_i, float* T, int THETA_i,
		float* THETA, int i, int j, int k, float* data, int l, int m, int n, 
		unsigned short* label_ip, float max_r, float search_r, int holes_i, 
		float* holes, int method, float min_value, int* a, int* b, float **D);

	void get_interpolated_spline_normal(int i, float* T, float theta, int* j, int* k, float** pts);

	void get_radius_control_point(int T_i, float* T, int a, int b, float* grid, 
		int segments, int* j, int* k, float **pts); 

	void get_new_angular_control_point(int T_i, float* T, int angle, int a, int b, 
		float* grid, int holes_i, float* holes, int holes_d_i, int holes_d_j, float* holes_d, 
		int l, int m, int n, unsigned short* label_ip, int* j, int* k, float **pts);  

	void fill_spine_label(float fx, float fy, float fz, float fxx, float fyy, float fzz, int l, int m, int n, int output_value, 
		unsigned short* label_ip);

	void gradient_calc(int i, int j, int k, float* data, int r, int s,  int t, unsigned short* output_label);

	void find_spline_proj_point(int T_i, float* T,  int S_i, float* S,
		int P_i, int P_j,  float* P, int* V_i, float** V, int* A_i, float **A);

	void create_overlay(int T_i, float* T,  int THETA_i, float* THETA, int radial_grid_i, float* radial_grid, int S_i, float* S, float threshold,
		int i, int j,  int k, float* data, 
		int d, int e,  int f, float* out_data);

	float get_slice_area(int T_i, float* T,  int THETA_i, float* THETA, int P_i, float* P, 
		int a, int b, float* grid, int S_i, float* S);

	void load_functional_images(int i, int j, int k, float* data, int current_volume, int slices_per_vol, int offset, int total_volumes, int r, int s,  int t, unsigned short* output_label);

	void refine_spline(int T_i, float* T, int THETA_i, float* THETA, int S_i, float* S, int iterations, float radial_search_percentage, int image_type,
		int i, int j, int k, float* data, 
		int radii_y, int radii_x, float* radii, int* weight_i, float **weight, int* pts_i, int* pts_j, float **pts);

	void find_radius_from_gradient(int T_i, float* T, int THETA_i, float* THETA, int S_i, float* S, int image_type,
		int i, int j, int k, float* data, int radii_y, int radii_x, float* radii, float max_radius, int r, int s,  int t, unsigned short* output_label);

	void get_vector_information(int T_i, float* T, int THETA_i, float* THETA, int radii_y, int radii_x, float* radii, int i, int j, int k, float* data,  
					int* U_i, int* U_j, int* U_k, float **U, int* V_i, int* V_j, int* V_k, float **V, int* W_i, int* W_j, int* W_k, float **W);

	void create_image_subsegment(int i, int j, int k, float* data, int P_i, int P_j, float* P, int l, int m, int n, unsigned short* label_ip);
	
	void gaussian_of_gradient(int i, int j, int k, float* data, int r, int s,  int t, unsigned short* output_label, int a, int b,  int c, unsigned short* output_label_2);

	void gradient_of_gradient(int i, int j, int k, float* data, int r, int s,  int t, unsigned short* output_label, int a, int b,  int c, unsigned short* output_label_2);

	void calculate_new_center_points(int T_i, float* T, int THETA_i, float* THETA, int S_i, float* S,  
		int radii_y, int radii_x, float* radii, int* pts_i, int* pts_j, float **pts);

	void get_radial_coordinates(int T_i, float* T, int THETA_i, float* THETA, int theta_targ, int S_i, float* S,  
		int radii_y, int radii_x, float* radii, int* pts_i, int* pts_j, float **pts);

	void get_grad_values_line(float T, float THETA, int search_points, int S_i, float* S, int i, int j, int k, float* data, int image_type,
		int* pts_i, float **pts);

	void create_templates(int T_i, float* T,  int THETA_i, float* THETA, int axial, float start_t, int theta_start, int theta_end, int i, int j, int k, float* data, int d, int e,  int f, unsigned short* input_label, 
		int points, int S_i, float* S, int image_type, int* a, int* b, float **D, int* A_i, float **A, int* B_i, float **B, int* pts_i, int* pts_j, float **pts);

	void get_full_radial_values(int T_i, float* T,  int THETA_i, float* THETA, int axial, int i, int j, int k, float* data,  
		int points, int S_i, float* S, int image_type, int* a, int* b, float **D);

	void create_new_edge_overlay_from_index_values(int T_i, float* T, int THETA_i, float* THETA, int S_i, float* S, int image_type, int axial, int newEdge_i, int newEdge_j, int* newEdge, int l, int m, int n, unsigned short* label_ip);

	void create_new_edge_overlay_from_distance_values(int T_i, float* T, int THETA_i, float* THETA, int S_i, float* S, int image_type, int axial, int a, int b, float* grid, 
	int l, int m, int n, unsigned short* label_ip);

	void draw_normals(int T_i, float* T, int image_type, int divider, int l, int m, int n, unsigned short* label_ip);

	void calculate_center_from_filled(int T_i, float* T, int THETA_i, float* THETA, int image_type, int axial, int l, int m, int n, unsigned short* label_ip, int* pts_i, int* pts_j, float **pts);

	void get_points_at_edge_of_surface(int T_i, float* T, int THETA_i, float* THETA, int image_type, int l, int m, int n, unsigned short* label_ip, int* pts_i, int* pts_j, float **pts);

	void get_radii_from_edgeindex(int T_i, float* T, int THETA_i, float* THETA, int S_i, float* S, int image_type, int axial, int newEdge_i, int newEdge_j, int* newEdge, int* a, int* b, float **D);

	void calculate_center_from_origin_vectors(int T_i, float* T, int originVector_i, int originVector_j, float* originVector, int S_i, float* S, int image_type, int* a, int* b, float **D);

	void calculate_center_points_from_edge_segmentation(int T_i, float* T, int THETA_i, float* THETA, int S_i, float* S, int image_type, int axial, int newEdge_i, int newEdge_j, int* newEdge, int matchPercent_i, int matchPercent_j, float* matchPercent, int l, int m, int n, unsigned short* label_ip, int* a, int* b, float **D);

	void create_flattened_spine(int T_i, float* T, int THETA_i, float* THETA, int S_i, float* S, int image_type, int radial_points, int sag_view, int i, int j, int k, float* data, int* a, int* b, float **D);
	
	void get_segment_voxels(int T_i, float* T,  int THETA_i, float* THETA, int axial, int image_type, int t_start, int t_end, int S_i, float* S, int d, int e,  int f, unsigned short* input_label, int r, int s, int t, unsigned short* output_label, int* voxels_i, float** voxels);

	void create_coordinate_system_slices(int T_i, float* T, int THETA_i, float* THETA, int S_i, float* S, int image_type, int radius, int slice, int l, int m, int n, unsigned short* label_ip);

	void get_spine_flexion_extension(int T_i, float* T, int image_type, int* i, float** p);

	void convert_edge_index_distances_to_mm_distances(int T_i, float* T, int THETA_i, float* THETA, int S_i, float* S, int image_type, int axial, int l, int m, int n, unsigned short* label_ip, int newEdge_i, int newEdge_j, int* newEdge, int newEdge_mm_i, int newEdge_mm_j, float* newEdge_mm);

	void convert_edge_mm_distances_to_index_distances(int T_i, float* T, int THETA_i, float* THETA, int S_i, float* S, int image_type, int axial, int l, int m, int n, unsigned short* label_ip, int newEdge_mm_i, int newEdge_mm_j, float* newEdge_mm, int newEdge_i, int newEdge_j, int* newEdge);

	void fill_partially_segmented_spine(int T_i, float* T,  int THETA_i, float* THETA, int image_type, int t_start, int t_end, int S_i, float* S, int d, int e,  int f, unsigned short* input_label, int r, int s, int t, unsigned short* output_label);

	void compare_two_volumes(int T_i, float* T, int THETA_i, float* THETA, int t_start, int t_end, int S_i, float* S, int image_type, int l, int m, int n, unsigned short* label_ip_1, int l2, int m2, int n2, unsigned short* label_ip_2, int* pts_i, float **pts);

	void create_submask(int T_i, float* T,  int THETA_i, float* THETA, int image_type, int t_start, int t_end, int S_i, float* S, int d, int e,  int f, unsigned short* input_label, int r, int s, int t, unsigned short* output_label);

	void create_spline_mask(int T_i, float* T,  int image_type, int t_start, int t_end, int S_i, float* S, int r, int s, int t, unsigned short* output_label);

	void create_submask_from_filled_mask(int T_i, float* T,  int THETA_i, float* THETA, int image_type, int t_start, int t_end, int S_i, float* S, int d, int e,  int f, unsigned short* input_label);

	void fill_partially_segmented_outlined_spine(int T_i, float* T,  int THETA_i, float* THETA, int image_type, int S_i, float* S, int d, int e,  int f, unsigned short* input_label, int r, int s, int t, unsigned short* output_label);

	void get_normal_to_plane(int T_i, float* T, int THETA_i, float* THETA, int image_type, int t_start, int t_end, int* j, int* k, float **pts);

	void calculate_axial_coords(int T_i, float* T, int r, int s, int t, unsigned short* output_label, int* pts_i, int* pts_j, float **pts);

private:
	std::vector<vec3> vp;
	float delta_t;
};

#endif //__SPLINE_H__
