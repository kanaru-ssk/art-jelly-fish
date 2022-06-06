#include "vec.h"

float distance(float _v1[3], float _v2[3])
{
	return(sqrt((_v1[X] - _v2[X]) * (_v1[X] - _v2[X]) + (_v1[Y] - _v2[Y]) * (_v1[Y] - _v2[Y]) + (_v1[Z] - _v2[Z]) * (_v1[Z] - _v2[Z])));
}

float dot(float _v1[3], float _v2[3])
{
	return(_v1[X] * _v2[X] + _v1[Y] * _v2[Y] + _v1[Z] * _v2[Z]);
}

void cross(float _v1[3], float _v2[3], float _vOut[3])
{
	_vOut[X] = _v1[Y] * _v2[Z] - _v1[Z] * _v2[Y];
	_vOut[Y] = _v1[Z] * _v2[X] - _v1[X] * _v2[Z];
	_vOut[Z] = _v1[X] * _v2[Y] - _v1[Y] * _v2[X];
}

void normVec(float _v[3])
{
	float norm;
	norm = sqrt(_v[X] * _v[X] + _v[Y] * _v[Y] + _v[Z] * _v[Z]);
	if(norm != 0){
		_v[X] /= norm;
		_v[Y] /= norm;
		_v[Z] /= norm;
	}
}

void normal(float _p1[3], float _p2[3], float _p3[3], float _vOut[3])
{
	float _v1[3], _v2[3];

	for(unsigned int i = 0; i < 3; i++){
		_v1[i] = _p3[i] - _p2[i];
		_v2[i] = _p1[i] - _p2[i];
	}
	cross(_v1, _v2, _vOut);
	normVec(_vOut);
}

void vertNorm(float _v1[3], float _v2[3], float _v3[3], float _v4[3], float _v5[3], float _v6[3], float _vOut[3])
{
	_vOut[X] = _v1[X] + _v2[X] + _v3[X] + _v4[X] + _v5[X] + _v6[X];
	_vOut[Y] = _v1[Y] + _v2[Y] + _v3[Y] + _v4[Y] + _v5[Y] + _v6[Y];
	_vOut[Z] = _v1[Z] + _v2[Z] + _v3[Z] + _v4[Z] + _v5[Z] + _v6[Z];
	normVec(_vOut);
}
