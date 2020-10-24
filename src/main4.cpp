#include "precompiled.h"
#include "util.h"
#include "stuff.h"
#include "shade.h"
#include "gpgpu.h"
#include "gpuBlur2_4.h"
#include "stefanfw.h"

#include "colorspaces.h"
#include "easyfft.h"

#define GLSL(sh) #sh

int wsx=1280, wsy = 720;
int scale = 6;
int sx = wsx / ::scale;
int sy = wsy / ::scale;



Array2D<float> img(sx, sy);


bool pause;



void updateConfig() {
}

struct SApp : App {
	void setup()
	{
		enableDenormalFlushToZero();

		createConsole();
		disableGLReadClamp();
		stefanfw::eventHandler.subscribeToEvents(*this);
		setWindowSize(wsx, wsy);

		forxy(img) {
			img(p) = ci::randFloat();
		}
	}
	void update()
	{
		stefanfw::beginFrame();
		stefanUpdate();
		stefanDraw();
		stefanfw::endFrame();
	}
	void keyDown(KeyEvent e)
	{
		if(keys['p'] || keys['2'])
		{
			pause = !pause;
		}
	}
	float noiseProgressSpeed;
	
	void stefanUpdate() {
		if(!pause) {
			img = gaussianBlur<float, WrapModes::GetWrapped>(img, 9);
			//auto imgb = gaussianBlur<float, WrapModes::GetWrapped>(img, 3);
			
			for(int i = 0; i < 1; i++) {
				int r = 1 << int(i * 1.5);
				r += 1;
				//auto imgb = gaussianBlur<float, WrapModes::GetWrapped>(img, r);
				//auto img2 = zeros_like(img);
				auto gradients = get_gradients(img);
				forxy(gradients) {
					vec2 pf = p;
					vec2 grad = safeNormalized(gradients(p));
					vec2 gradPerp = vec2(-grad.y, grad.x);
					vec2 grad_a = safeNormalized(getBilinear(gradients, pf+gradPerp));
					grad_a = -vec2(-grad_a.y, grad_a.x);
					vec2 grad_b = safeNormalized(getBilinear(gradients, pf-gradPerp));
					grad_b = vec2(-grad_b.y, grad_b.x);
					vec2 dir = grad_a + grad_b;
					if(dot(dir, grad) < 0.0) {
						//if(getBilinear(imgb, vec2(p+dir)) < .5f)
							img(p) += length(dir) * 1.0 * img(p);
					}
				}
				/*auto img2b = gaussianBlur<float, WrapModes::GetWrapped>(img2, r);
				forxy(img) {
					img(p) += img2b(p);
				}*/
			}
		
			vec2 center(sx/2, sy/2);
			float maxDist = distance(vec2(), center);

			float sum = std::accumulate(img.begin(), img.end(), 0.0f);
			float avg = sum / img.area;
			float mul = .5 / avg;
			forxy(img) {
				img(p) *= mul;
				img(p) = smoothstep(0.0, 1.0, img(p));
			}

			//img = to01(img);
			
			if(mouseDown_[0])
			{
				drawCircle(vec2(mouseX * sx, mouseY * sy));
			}

			/*float t = getElapsedSeconds();
			vec2 circlePos(sin(t*.345f)*.5+.5, cos(t * 0.5) * .5 + .5);
			circlePos *= vec2(sx, sy);
			drawCircle(circlePos);*/
		}
		if (pause)
			Sleep(50);
	}
	void drawCircle(vec2 scaledm) {
		Area a(scaledm, scaledm);
		int r = 10;
		a.expand(r, r);
		for (int x = a.x1; x <= a.x2; x++)
		{
			for (int y = a.y1; y <= a.y2; y++)
			{
				vec2 v = vec2(x, y) - scaledm;
				float w = max(0.0f, 1.0f - length(v) / r);
				w = max(0.0f, w);
				w = 3 * w * w - 2 * w * w * w;
				w = 3 * w * w - 2 * w * w * w;
				w = 3 * w * w - 2 * w * w * w;
				img.wr(x, y) = lerp(img.wr(x, y), 1.0f, w);
			}
		}
	}
	void stefanDraw() {
		gl::clear(Color(0, 0, 0));

		auto tex = gtex(img);
		tex = shade2(tex,
			"float f = fetch1(tex);"
			"float fw = fwidth(f);"
			"f = smoothstep (.5 - fw / 2, .5 + fw / 2, f);"
			//"f = 1.0 - f;"
			"_out = vec3(f);"
			, ShadeOpts().scale(2).ifmt(GL_RGB8)
			);
		auto texb = tex;//gpuBlur2_4::run(tex, 1);
		auto texbg = get_gradients_tex(texb);
		auto texbgc = shade2(texbg,
			"vec2 grad = fetch2(tex);"
			"vec3 rbow = rainbow(grad);"
			"_out = rbow;"
			,ShadeOpts().ifmt(GL_RGB16F),
			"vec3 rainbow(float x, float br)  {"
			"vec3 c = .5 + .5 * cos(6.2832*(x - vec3(0,1,2)/3.));"
			"c = pow(c, vec3(3.0));" // emphasise red, green and blue hues
			"return c * br;"
			"}"
			"vec3 rainbow(vec2 C)   { return rainbow(atan(C.y,C.x)/3.1416/2.0 + .5, length(C)*10.0); }"
			);
		auto texbgcb = gpuBlur2_4::run_longtail(texbgc, 2, 1.0f);
		texbgcb = shade2(texbgcb,
			"vec3 c = fetch3();"
			"vec3 hsl = rgb2hsl(c);"
			"hsl[1] = 1+0*pow(hsl[1], .2);"
			"c = hsl2rgb(hsl);"
			"_out = c;"
			,
			ShadeOpts(),
			FileCache::get("stuff.fs"));
		tex = shade2(texb, texbgcb,
			"float b = fetch1();"
			"vec3 cb = b * vec3(0.6);"
			"vec3 rbow = fetch3(tex2);"
			"_out = cb + rbow;"
			"_out = max(vec3(0), min(vec3(1), _out));"
			"_out = pow(_out, vec3(1.0/2.2));"
			//"_out = 1.0 - _out;"
			, ShadeOpts().ifmt(GL_RGB16F)
			);
		tex->setTopDown(true);
		gl::draw(tex, getWindowBounds());
	}
};

CINDER_APP(SApp, RendererGl)
