#pragma once
#include "precompiled.h"
#include "TextureCache.h"

namespace gpuBlur2_5 {
	gl::TextureRef run(gl::TextureRef src, int lvls);
	gl::TextureRef run_longtail(gl::TextureRef src, int lvls, float lvlmul);
	float getGaussW();
	float gauss(float f, float width);
	gl::TextureRef upscale(gl::TextureRef src, ci::ivec2 toSize, TextureCache* textureCache = nullptr);
	gl::TextureRef upscale(gl::TextureRef src, float hscale, float vscale, TextureCache* textureCache = nullptr);
	gl::TextureRef singleblur(gl::TextureRef src, float hscale, float vscale, TextureCache* textureCache = nullptr);
}

namespace gpuBlur = gpuBlur2_5;