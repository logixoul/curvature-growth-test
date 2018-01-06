#pragma once
#include "precompiled.h"
#include "shade.h"
#include "cfg1.h"
#include "stuff.h"
#include "shade.h"
#include "gpgpu.h" // for shade2
#include "gpuBlur2_5.h"
#include "stefanfw.h"

namespace gpuBlur2_5 {

	static TextureCache zoomstateCache;
	static TextureCache upscaleCache;

	gl::TextureRef run(gl::TextureRef src, int lvls) {
		auto state = shade2(src, "_out = fetch3();");

		for (int i = 0; i < lvls; i++) {
			state = singleblur(state, .5, .5, &zoomstateCache);
		}
		state = upscale(state, src->getSize(), &zoomstateCache);
		return state;
	}
	
	gl::TextureRef run_longtail(gl::TextureRef src, int lvls, float lvlmul) {
		vector<float> weights;
		float sumw = 0.0f;
		for (int i = 0; i < lvls; i++) {
			float w = pow(lvlmul, float(i));
			weights.push_back(w);
			sumw += w;
		}
		foreach(float& w, weights) {
			w /= sumw;
		}
		/*cout << "Weights: ";
		foreach(float w, weights) {
			cout << w << ", ";
		}
		cout << "\n";*/
		vector<gl::TextureRef> zoomstates;
		zoomstates.push_back(src);
		globaldict["_mul"] = 1.0 / sumw;
		zoomstates[0] = shade2(zoomstates[0],
			"_out = fetch3() * _mul;");
		for (int i = 0; i < lvls; i++) {
			auto newZoomstate = singleblur(zoomstates[i], .5, .5, &zoomstateCache);
			zoomstates.push_back(newZoomstate);
			if (newZoomstate->getWidth() < 1 || newZoomstate->getHeight() < 1) throw runtime_error("too many blur levels");
		}
		for (int i = lvls - 1; i > 0; i--) {
			auto upscaled = upscale(zoomstates[i], zoomstates[i-1]->getSize(), &upscaleCache);
			globaldict["_mul"] = lvlmul;// weights[i];
			zoomstates[i-1] = shade2(zoomstates[i-1], upscaled,
				"vec3 acc = fetch3(tex);"
				"vec3 nextzoom = fetch3(tex2);"
				"vec3 c = acc + nextzoom * _mul;"
				"_out = c;"
				, ShadeOpts().texCache(&zoomstateCache)
			);
		}
		return zoomstates[0];
	}
	float getGaussW() {
		// default value determined by trial and error
		return cfg1::getOpt("gaussW", 0.75f, [&]() { return keys['/']; },
			[&]() { return exp(mouseY-0.5); });
	}
	float gauss(float f, float width) {
		return exp(-f*f / (width*width));
	}
	gl::TextureRef upscale(gl::TextureRef src, ci::ivec2 toSize, TextureCache* textureCache) {
		return upscale(src, float(toSize.x) / src->getWidth(), float(toSize.y) / src->getHeight(), textureCache);
	}
	gl::TextureRef upscale(gl::TextureRef src, float hscale, float vscale, TextureCache* textureCache) {
		globaldict["gaussW"] = getGaussW();
		string lib =
			"float gauss(float f, float width) {"
			"	return exp(-f*f/(width*width));"
			"}"
			;
		string shader =
			"	vec2 offset = vec2(GB2_offsetX, GB2_offsetY);"
			// here tc2 is half a texel TO THE TOP LEFT of the texel center. IT IS IN UV SPACE.
			"	vec2 tc2 = floor(tc * texSize) / texSize;"
			// here I make tc2 be the texel center
			"	tc2 += tsize / 2.0;"
			// frXY is in PIXEL SPACE. its x and y go from -.5 to .5
			"	vec2 frXY = (tc - tc2) * texSize;"
			"	float fr = (GB2_offsetX == 1.0) ? frXY.x : frXY.y;"
			"	vec3 aM1 = fetch3(tex, tc2 + (-1.0) * offset * tsize);"
			"	vec3 a0 = fetch3(tex, tc2 + (0.0) * offset * tsize);"
			"	vec3 aP1 = fetch3(tex, tc2 + (+1.0) * offset * tsize);"
			"	"
			"	float wM1=gauss(-1.0-fr, gaussW);"
			"	float w0=gauss(-fr, gaussW);"
			"	float wP1=gauss(1.0-fr, gaussW);"
			"	float sum=wM1+w0+wP1;"
			"	wM1/=sum;"
			"	w0/=sum;"
			"	wP1/=sum;"
			"	_out = wM1*aM1 + w0*a0 + wP1*aP1;";
		globaldict["GB2_offsetX"] = 1.0;
		globaldict["GB2_offsetY"] = 0.0;
		setWrapBlack(src);
		auto hscaled = shade2(src, shader, ShadeOpts().scale(hscale, 1.0f).texCache(textureCache), lib);
		globaldict["GB2_offsetX"] = 0.0;
		globaldict["GB2_offsetY"] = 1.0;
		setWrapBlack(hscaled);
		auto vscaled = shade2(hscaled, shader, ShadeOpts().scale(1.0f, vscale).texCache(textureCache), lib);
		return vscaled;
	}
	gl::TextureRef singleblur(gl::TextureRef src, float hscale, float vscale, TextureCache* textureCache) {
		float gaussW = .75;

		float w0 = gauss(0.0, gaussW);
		float w1 = gauss(1.0, gaussW);
		float sum =2.0f*w1 + w0;
		w1 /= sum;
		w0 /= sum;
		stringstream weights;
		weights << fixed << "float w0=" << w0 << ", w1=" << w1 << ";" << endl;
		string shader =
			"vec2 offset = vec2(GB2_offsetX, GB2_offsetY);"
			"vec3 aM1 = fetch3(tex, tc + (-1.0) * offset * tsize);"
			"vec3 a0 = fetch3(tex, tc + (0.0) * offset * tsize);"
			"vec3 aP1 = fetch3(tex, tc + (+1.0) * offset * tsize);"
			""
			+ weights.str() +
			"_out = w1 * (aM1 + aP1) + w0 * a0;";

		globaldict["GB2_offsetX"] = 1.0;
		globaldict["GB2_offsetY"] = 0.0;
		setWrapBlack(src);
		auto hscaled = shade2(src, shader, ShadeOpts().scale(hscale, 1.0f).texCache(textureCache));
		globaldict["GB2_offsetX"] = 0.0;
		globaldict["GB2_offsetY"] = 1.0;
		setWrapBlack(hscaled);
		auto vscaled = shade2(hscaled, shader, ShadeOpts().scale(1.0f, vscale).texCache(textureCache));
		return vscaled;
	}

}