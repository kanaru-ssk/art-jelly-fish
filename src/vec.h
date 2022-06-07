#pragma once

#include <stdio.h>
#include <math.h>

#define X 0
#define Y 1
#define Z 2

float distance(float _v1[3], float _v2[3]);
float dot(float _v1[3], float _v2[3]);
void cross(float _v1[3], float _v2[3], float _vOut[3]);
void normVec(float _v[3]);
void normal(float _p1[3], float _p2[3], float _p3[3], float _vOut[3]);
void vertNorm(float _v1[3], float _v2[3], float _v3[3], float _v4[3], float _v5[3], float _v6[3], float _vOut[3]);
