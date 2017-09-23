#include "precompiled.h"
#include "PerlinCalculator2D.h"

float PerlinCalculator2D::calcAt(Vec3f const& p){
	Vec3f pMod = Vec3f(pos_mod(p.x, 1.0f), pos_mod(p.y, 1.0f), pos_mod(p.z, 1.0f));
	Vec3f pScaled = pMod * (size - 1);
	Vec3f floor(std::floor(pScaled.x), std::floor(pScaled.y), std::floor(pScaled.z));
	Vec3i ifloor = floor;

	Vec3f fract = pScaled - floor;
	float fx = fade(fract.x);
	float fy = fade(fract.y);
	float fz = fade(fract.z);
	float dot[2][2][2];
	for(int x = 0; x <= 1; x++)
		for(int y = 0; y <= 1; y++)
			for(int z = 0; z <= 1; z++)
			{
				Vec3f const& grad = getGradient(ifloor + Vec3i(x, y, z));
				dot[x][y][z] = ci::dot(grad, Vec3f(x, y, z) - fract);
			}
		
	// back->front = y, bottom->top = z, left->right = x
	float lerpXTopBack = lerp(dot[0][0][1], dot[1][0][1], fx);
	float lerpXTopFront = lerp(dot[0][1][1], dot[1][1][1], fx);
	float lerpXBottomBack = lerp(dot[0][0][0], dot[1][0][0], fx);
	float lerpXBottomFront = lerp(dot[0][1][0], dot[1][1][0], fx);
	float lerpZBack = lerp(lerpXBottomBack, lerpXTopBack, fz);
	float lerpZFront = lerp(lerpXBottomFront, lerpXTopFront, fz);
	float lerpY = lerp(lerpZBack, lerpZFront, fy);
	return lerpY;
}

PerlinCalculator2D::PerlinCalculator2D()
{
	gradients = vector<Vec3f>(size*size*size);

	for(int x = 0; x < size; x++)
	{
		for(int y = 0; y < size; y++)
		{
			for(int z = 0; z < size; z++)
			{
				auto& gradient = getGradient(Vec3i(x, y, z));
				gradient = ci::Rand::randVec3f();
			}
		}
	}
	for(int x = 0; x < size; x++)
	{
		for(int y = 0; y < size; y++)
		{
			for(int z = 0; z < size; z++)
			{
				auto& gradient = getGradient(Vec3i(x, y, z));
				int wx = x % (size - 1);
				int wy = y % (size - 1);
				int wz = z % (size - 1);
				gradient = getGradient(Vec3i(wx, wy, wz));
			}
		}
	}
}