#include <math.h>
#include "app.h"

__device__ float d_length(float _v[3])
{
        return(sqrt(_v[X] * _v[X] + _v[Y] * _v[Y] + _v[Z] * _v[Z]));
}

__device__ float d_distance(float _v1[3], float _v2[3])
{
        return(sqrt((_v1[X] - _v2[X]) * (_v1[X] - _v2[X]) + (_v1[Y] - _v2[Y]) * (_v1[Y] - _v2[Y]) + (_v1[Z] - _v2[Z]) * (_v1[Z] - _v2[Z])));
}

__device__ void d_add(float _v1[3], float _v2[3])
{
        _v1[X] += _v2[X];
        _v1[Y] += _v2[Y];
        _v1[Z] += _v2[Z];
}

__device__ void d_scale(float _v[3], float _s)
{
        _v[X] *= _s;
        _v[Y] *= _s;
        _v[Z] *= _s;
}

__device__ void d_normVec(float _v[3])
{
	float _length = d_length(_v);
	_length += (_length == 0);
	_v[X] /= _length;
	_v[Y] /= _length;
	_v[Z] /= _length;
}

__device__ float d_dot(float _v1[3], float _v2[3])
{
	return(_v1[X] * _v2[X] + _v1[Y] * _v2[Y] + _v1[Z] * _v2[Z]);
}

__device__ void d_cross(float _v1[3], float _v2[3], float _vOut[3])
{
	_vOut[X] = _v1[Y] * _v2[Z] - _v1[Z] * _v2[Y];
	_vOut[Y] = _v1[Z] * _v2[X] - _v1[X] * _v2[Z];
	_vOut[Z] = _v1[X] * _v2[Y] - _v1[Y] * _v2[X];
}

__device__ void d_normal(float _p1[3], float _p2[3], float _p3[3], float _vOut[3])
{
	float _v1[3], _v2[3];

	for(unsigned int i = 0; i < 3; i++){
		_v1[i] = _p3[i] - _p2[i];
		_v2[i] = _p1[i] - _p2[i];
	}
	d_cross(_v1, _v2, _vOut);
	d_normVec(_vOut);
}

__device__ void d_vertNorm(float _v1[3], float _v2[3], float _v3[3], float _v4[3], float _v5[3], float _v6[3], float _vOut[3])
{
	_vOut[X] = _v1[X] + _v2[X] + _v3[X] + _v4[X] + _v5[X] + _v6[X];
	_vOut[Y] = _v1[Y] + _v2[Y] + _v3[Y] + _v4[Y] + _v5[Y] + _v6[Y];
	_vOut[Z] = _v1[Z] + _v2[Z] + _v3[Z] + _v4[Z] + _v5[Z] + _v6[Z];
	d_normVec(_vOut);
}

__device__ void d_setPre(float _pre[3], float _point[3])
{
        _pre[X] = _point[X];
        _pre[Y] = _point[Y];
        _pre[Z] = _point[Z];
}

__device__ void d_force(float point[3], float pre[3], float weight, float time, float dt)
{
	float w = 0.1f * (__sinf(time) * 0.5f + 0.5f);
	float acc[3] = {w, -G / 100.0f, 0.0f};
	d_scale(acc, 0.5f * dt * dt / M);

	float vel[3] = {point[X] - pre[X], point[Y] - pre[Y], point[Z] - pre[Z]};
	d_add(vel, acc);
	d_scale(vel, weight);
	d_setPre(pre, point);
	d_add(point, vel);
}

__device__ float d_calcOffset(int x, int y, int offset_x, int offset_y, int kinds)
{
	offset_x *= 1 - 2 * (2 <= x * kinds % 4);
	offset_y *= 1 - 2 * (2 <= y * kinds % 4);

	return((y + offset_y) * DIV_W + (DIV_W + x + offset_x) % DIV_W) * (2 * (y + offset_y < DIV_H) - 1);
}

__device__ unsigned int d_calcRestPos(int y, int offset_x, int offset_y, int kinds)
{

	return(y + offset_y * (y * kinds % 4 < 2) + 2 * (offset_y < 0) / kinds);
}

__device__ unsigned int d_calcRestNo(int offset_x, int offset_y, int kinds)
{
	return(kinds * abs(offset_x) / 2 + kinds * abs(offset_y) - 3 * kinds + 5);
}

__device__ void d_spring(float point[3], float offset[3], float weight, float rest, float dt)
{
	float d = d_distance(point, offset);
	float f = (d - rest) * 100.0f;
	float dx[3] = {offset[X] - point[X], offset[Y] - point[Y], offset[Z] - point[Z]};
	d_normVec(dx);
	d_scale(dx, weight * f * dt * dt * 0.5 / M);
	d_add(point, dx);
}

__device__ void d_constraint(unsigned int index, int x, int y, int offset_x, int offset_y, int kinds, float (*point)[3], float *weight, float (*rest)[6], float dt)
{
	int offset_no, rest_no, rest_pos;
	offset_no = d_calcOffset(x, y, offset_x, offset_y, kinds);
	rest_no = d_calcRestNo(offset_x, offset_y, kinds);
	rest_pos = d_calcRestPos(y, offset_x, offset_y, kinds);
	if(0 <= offset_no)
		d_spring(point[index], point[offset_no], weight[index], rest[rest_pos][rest_no], dt);
}

__device__ void d_calcVertNorm(unsigned int index, int y, float (*norm)[3], float (*norm_surf)[3])
{
	int temp_1, temp_2, temp_3;
	if(y == 0){
		for(unsigned int i = 0; i < DIV_W; i++){
			norm[index][X] += norm_surf[i][X];
			norm[index][Y] += norm_surf[i][Y];
			norm[index][Z] += norm_surf[i][Z];
		}
		d_normVec(norm[index]);
	}else{
		temp_1 = ((index - DIV_W) / DIV_W) * (DIV_W * 2 + 2) + 2 * (index % DIV_W);
		temp_2 = (index / DIV_W) * (DIV_W * 2 + 2) + 2 * (index % DIV_W);
		temp_3 = (index % DIV_W ==0) * DIV_W * 2;
		d_vertNorm(norm_surf[temp_1 - 1 + temp_3], norm_surf[temp_1], norm_surf[temp_1 + 1], norm_surf[temp_2 - 2 + temp_3], norm_surf[temp_2 - 1 + temp_3], norm_surf[temp_2], norm[index]);
	}
}

__global__ void d_update(float (*point)[3], float (*pre)[3], float *weight, float (*rest)[6], float time, float dt, float (*norm)[3], float (*norm_surf)[3])
{
	unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
        if (index >= NUM_POINTS)
                return;
	
	//外力を加える
	d_force(point[index], pre[index], weight[index], time, dt);
	__syncthreads();

	//制約充足
	int x = index % DIV_W;
	int y = index / DIV_W;
	for (unsigned int i = 0; i < 5; i++){
		d_constraint(index, x, y, 1, 0 , 2, point, weight, rest, dt);
		__syncthreads();

		d_constraint(index, x, y, -1, 0, 2, point, weight, rest, dt);
		__syncthreads();

		d_constraint(index, x, y, 0, 1, 2, point, weight, rest, dt);
		__syncthreads();
		
		d_constraint(index, x, y, 0, -1, 2, point, weight, rest, dt);
		__syncthreads();

		d_constraint(index, x, y, 1, 1, 2, point, weight, rest, dt);
		__syncthreads();

		d_constraint(index, x, y, -1, 1, 2, point, weight, rest, dt);
		__syncthreads();

		d_constraint(index, x, y, 1, -1, 2, point, weight, rest, dt);
		__syncthreads();

		d_constraint(index, x, y, -1, -1, 2, point, weight, rest, dt);
		__syncthreads();

		d_constraint(index, x, y, 2, 0, 1, point, weight, rest, dt);
		__syncthreads();

		d_constraint(index, x, y, -2, 0, 1, point, weight, rest, dt);
		__syncthreads();

		d_constraint(index, x, y, 0, 2, 1, point, weight, rest, dt);
		__syncthreads();

		d_constraint(index, x, y, 0, -2, 1, point, weight, rest, dt);
		__syncthreads();

		d_spring(point[index], point[y * DIV_W + (x + DIV_W / 2) % DIV_W], weight[index], rest[y][5], dt);
		__syncthreads();
	}

	//法線計算
	d_calcVertNorm(index, y, norm, norm_surf);
}

__global__ void d_calcSurfNorm(float (*point)[3], float (*norm_surf)[3], unsigned short *point_index)
{
	unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index >= NUM_INDEX - 2)
		return;

	int temp = (index % 2 == 0);
	d_normal(point[point_index[index]], point[point_index[index + 1 + temp]], point[point_index[index + 2 - temp]], norm_surf[index]);
}

void launchGPUKernel(float (*point)[3], float (*pre)[3], float *weight, float(*rest)[6], float time, float dt, float (*norm)[3], float (*norm_surf)[3], unsigned short *point_index)
{
	dim3 grid(NUM_POINTS / 512 + 1, 1);
	dim3 block(512, 1, 1);
	d_update <<< grid, block >>> (point, pre, weight, rest, time, dt, norm, norm_surf);

	dim3 grid_2(NUM_INDEX / 512 + 1, 1);
	dim3 block_2(512, 1, 1);
	d_calcSurfNorm <<< grid_2, block_2 >>> (point, norm_surf, point_index);
}
