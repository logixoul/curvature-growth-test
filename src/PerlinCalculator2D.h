#pragma once
#include "precompiled.h"
#include "util.h"

struct PerlinCalculator2D
{
	vector<Vec3f> gradients;
	static const int size = 16;
	PerlinCalculator2D();
	Vec3f& getGradient(Vec3i const& v) {
		return gradients[v.z * size * size + v.y * size + v.x];
	}
	// p loops in [0, 1]
	float fade(float f) { return smoothstep(0.0, 1.0, f); }
	float pos_mod(float a, float b) { float c = fmod(a, b); if(c < 0.0f) c += b; return c; }
	float calcAt(Vec3f const& p);
};
