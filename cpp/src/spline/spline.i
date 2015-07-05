%module spline
%{
#define SWIG_FILE_WITH_INIT
#include "vec3.h"
#include "spline.h"
%}

%include "../numpy.i"

%init %{
    import_array();
%}

%apply (int DIM1, float* IN_ARRAY1)
        {(int i, float* T),
	 (int MM_i, float* MM),
         (int j, float* THETA),
         (int T_i, float* T),
	 (int i, float* P),
  	 (int P_i, float* P),
         (int THETA_i, float* THETA),
	 (int holes_i, float* holes),
         (int S_i, float* S),
	 (int radial_grid_i, float* radial_grid),
 	 (int ANG_i, float* ANG),
 	 (int newEdge_i, float* newEdge)}

%apply (int DIM1, int* IN_ARRAY1)
        {(int newEdge_i, int* newEdge)}

%apply (int DIM1, int DIM2, int* IN_ARRAY2)
        {(int newEdge_i, int newEdge_j, int* newEdge),
         (int matchPercent_i, int matchPercent_j, int* matchPercent)}

%apply (int DIM1, int DIM2, float* IN_ARRAY2)
        {(int i, int j,  float* P),
         (int P_i, int P_j,  float* P),
         (int newEdge_mm_i, int newEdge_mm_j, float* newEdge_mm),
	 	 (int holes_d_i, int holes_d_j, float* holes_d),
	 	 (int radii_y, int radii_x, float* radii),
	 	 (int a, int b, float* grid),
	 	 (int originVector_i, int originVector_j, float* originVector),
	 	 (int matchPercent_i, int matchPercent_j, float* matchPercent)}

%apply (int DIM1, int DIM2, int DIM3, float* IN_ARRAY3)
        {(int i, int j, int k, float* data),
	 (int d, int e, int f, float* out_data),
	 (int i, int j, int k, unsigned short* data)}

%apply (int* DIM1, float** ARGOUTVIEWM_ARRAY1)
        {(int* i, float** p),
	 (int* voxels_i, float** voxels),
	 (int* vox_values_i, float** vox_values)}

%apply (int* DIM1, float** ARGOUTVIEWM_ARRAY1)
        {(int* U_i, float **U),
	 (int* V_i, float **V), 
	 (int* W_i, float **W),
         (int* A_i, float **A),
         (int* B_i, float **B),
	 (int* pts_i, float **pts),
	 (int* weight_i, float **weight)}

%apply (int* DIM1, int* DIM2, int** ARGOUTVIEWM_ARRAY2)
        {(int* j, int* k, int **pts)}

%apply (int* DIM1, int* DIM2, float** ARGOUTVIEWM_ARRAY2)
        {(int* j, int* k, float **pts),
	 (int* pts_i, int* pts_j, float **pts),
         (int* a, int* b, float **D),
	 (int* radii_x2, int* radii_y2, float **radii2),
	 (int* fourier_x, int* fourier_y, float **fourier)}

%apply (int* DIM1, int* DIM2, int* DIM3, float** ARGOUTVIEWM_ARRAY3)
	{(int* U_i, int* U_j, int* U_k, float **U),
	 (int* V_i, int* V_j, int* V_k, float **V),
	 (int* W_i, int* W_j, int* W_k, float **W),
	 (int* vox_values_i, int* vox_values_j, int* vox_values_k, float **vox_values_3),
	 (int* rad_dist_i, int* rad_dist_j, int* rad_dist_k, float **rad_dist)}

%apply (int DIM1, float* INPLACE_ARRAY1)
        {(int pts_i, float* pts),
	 (int output_i, float* output)}

%apply (int DIM1, int DIM2, int DIM3, float* INPLACE_ARRAY3)
        {(int i, int j,  int k, float* data_ip),
	 (int roidata_i, int roidata_j, int roidata_k, float* roidata)}

%apply (int DIM1, int DIM2, int DIM3, unsigned short* INPLACE_ARRAY3)
        {(int l, int m,  int n, unsigned short* label_ip),
  	 (int l, int m,  int n, unsigned short* label_ip_1),
  	 (int l2, int m2,  int n2, unsigned short* label_ip_2),
	 (int d, int e,  int f, unsigned short* input_label),
	 (int r, int s,  int t, unsigned short* output_label),
	 (int a, int b,  int c, unsigned short* output_label_2)}

%include "std_string.i"
%include "std_vector.i"
%include "typemaps.i"
%include "std_list.i"

# setup exceptions handler
%exception {
        try{
                $action
                }
        catch (const char*& err_msg) {
                PyErr_SetString(PyExc_RuntimeError, err_msg);
                return NULL;
   }
}

%include "spline.h"


