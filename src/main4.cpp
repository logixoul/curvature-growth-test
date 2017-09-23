#include "precompiled.h"
#if 1
#include "util.h"
#include "stuff.h"
#include "shade.h"
#include "gpgpu.h"
#include "gpuBlur2_4.h"
#include "cfg1.h"
#include "sw.h"
#include "my_console.h"
#include "hdrwrite.h"
#include <float.h>
#include "simplexnoise.h"
#include "mainfunc_impl.h"
#include "colorspaces.h"
#include "easyfft.h"

#define GLSL(sh) #sh

int wsx=800, wsy = 800 * (800.0f / 1280.0f);
int scale = 4;
int sx = wsx / scale;
int sy = wsy / scale;
bool mouseDown_[3];
bool keys[256];
gl::Texture::Format gtexfmt;
gl::Texture texToDraw;
bool texOverride = false;

Array2D<float> img(sx, sy);

float mouseX, mouseY;
bool pause;
bool keys2[256];


void updateConfig() {
}

const int numDetailsX = 5;
const float nscale = numDetailsX / (float)sx;

static float noiseXAt(Vec2f p, float z) {
	float noiseX = ::octave_noise_3d(1, .5, 1.0, p.x * nscale, p.y * nscale, z);
	return noiseX;
}
	
static float noiseYAt(Vec2f p, float z) {
	float noiseY = ::octave_noise_3d(1, .5, 1.0, p.x * nscale, p.y * nscale + numDetailsX, z);
	return noiseY;
}
	

struct SApp : AppBasic {
	Rectf area;
		
	void setup()
	{
		//keys2['0']=keys2['1']=keys2['2']=keys2['3']=true;
		//_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

		_controlfp(_DN_FLUSH, _MCW_DN);

		area = Rectf(0, 0, (float)sx-1, (float)sy-1).inflated(Vec2f::zero());

		glClampColor(GL_CLAMP_FRAGMENT_COLOR, GL_FALSE);
		glClampColor(GL_CLAMP_READ_COLOR, GL_FALSE);
		glClampColor(GL_CLAMP_VERTEX_COLOR, GL_FALSE);
		gtexfmt.setInternalFormat(hdrFormat);
		setWindowSize(wsx, wsy);

		glEnable(GL_POINT_SMOOTH);

		Vec2f center = Vec2f(img.Size()) / 2.0f;
		forxy(img) {
			img(p) = ci::randFloat();
		}
	}
	void keyDown(KeyEvent e)
	{
		keys[e.getChar()] = true;
		if(e.isControlDown()&&e.getCode()!=KeyEvent::KEY_LCTRL)
		{
			keys2[e.getChar()] = !keys2[e.getChar()];
			return;
		}
		if(keys['r'])
		{
		}
		if(keys['p'] || keys['2'])
		{
			pause = !pause;
		}
	}
	void keyUp(KeyEvent e)
	{
		keys[e.getChar()] = false;
	}
	
	void mouseDown(MouseEvent e)
	{
		mouseDown_[e.isLeft() ? 0 : e.isMiddle() ? 1 : 2] = true;
	}
	void mouseUp(MouseEvent e)
	{
		mouseDown_[e.isLeft() ? 0 : e.isMiddle() ? 1 : 2] = false;
	}
	Vec2f direction;
	Vec2f lastm;
	void mouseDrag(MouseEvent e)
	{
		mm();
	}
	void mouseMove(MouseEvent e)
	{
		mm();
	}
	void mm()
	{
		direction = getMousePos() - lastm;
		lastm = getMousePos();
	}
	Vec2f reflect(Vec2f const & I, Vec2f const & N)
	{
		return I - N * N.dot(I) * 2.0f;
	}
	float noiseProgressSpeed;
	
	void draw()
	{
		::texOverride = false;

		my_console::beginFrame();
		sw::beginFrame();
		static bool first = true;
		first = false;

		wsx = getWindowSize().x;
		wsy = getWindowSize().y;

		mouseX = getMousePos().x / (float)wsx;
		mouseY = getMousePos().y / (float)wsy;
		/*noiseProgressSpeed=cfg1::getOpt("noiseProgressSpeed", .00008f,
			[&]() { return keys['s']; },
			[&]() { return expRange(mouseY, 0.01f, 100.0f); });*/
		
		gl::clear(Color(0, 0, 0));

		updateIt();
		
		renderIt();

		/*Sleep(50);*/my_console::clr();
		sw::endFrame();
		cfg1::print();
		my_console::endFrame();

		if(pause)
			Sleep(50);
		//Sleep(2000);
	}
	void updateIt() {
		if(!pause) {
			img = gaussianBlur<float, WrapModes::GetWrapped>(img, 9);
			
			for(int i = 0; i < 1; i++) {
				int r = 1 << int(i * 1.5);
				r += 1;
				auto imgb = gaussianBlur<float, WrapModes::GetWrapped>(img, r);
				auto img2 = zeros_like(img);
				auto gradients = get_gradients(imgb);
				forxy(gradients) {
					Vec2f pf = p;
					Vec2f grad = gradients(p).safeNormalized();
					Vec2f gradPerp = Vec2f(-grad.y, grad.x);
					Vec2f grad_a = getBilinear(gradients, pf+gradPerp).safeNormalized();
					grad_a = -Vec2f(-grad_a.y, grad_a.x);
					Vec2f grad_b = getBilinear(gradients, pf-gradPerp).safeNormalized();
					grad_b = Vec2f(-grad_b.y, grad_b.x);
					Vec2f dir = grad_a + grad_b;
					if(dir.dot(grad) < 0.0) {
						//if(imgb(p) < .5f)
						img2(p) += dir.length()*.5;
					}
				}
				auto img2b = gaussianBlur<float, WrapModes::GetWrapped>(img2, r);
				forxy(img) {
					img(p) += img2b(p);
				}
			}
		
			Vec2f center(sx/2, sy/2);
			float maxDist = Vec2f::zero().distance(center);
			forxy(img) {
				//img(p) *= smoothstep(maxDist, 0.0, Vec2f(p).distance(center));
			}

			float sum = std::accumulate(img.begin(), img.end(), 0.0f);
			float avg = sum / img.area;
			float mul = .5 / avg;
			forxy(img) {
				img(p) *= mul;
				img(p) = smoothstep(0.0, 1.0, img(p));
				//img(p) = smoothstep(0.0, 1.0, img(p));
			}

			auto img2 = zeros_like(img);

			forxy(img) {
				Vec2f move;
				move.x = noiseXAt(Vec2f(p), getElapsedFrames() / 100.0f);
				move.y = noiseYAt(Vec2f(p), getElapsedFrames() / 100.0f);
				aaPoint(img2, Vec2f(p) + move, img(p));
				//img2(p) = getBilinear(img, Vec2f(p) + move);
			}

			img = img2;

			img = to01(img);
			
			if(mouseDown_[0])
			{
				Vec2f scaledm = Vec2f(mouseX * (float)sx, mouseY * (float)sy);
				Area a(scaledm, scaledm);
				int r = 10;
				a.expand(r, r);
				for(int x = a.x1; x <= a.x2; x++)
				{
					for(int y = a.y1; y <= a.y2; y++)
					{
						Vec2f v = Vec2f(x, y) - scaledm;
						float w = max(0.0f, 1.0f - v.length() / r);
						w=max(0.0f,w);
						w = 3 * w * w - 2 * w * w * w;
						w = 3 * w * w - 2 * w * w * w;
						w = 3 * w * w - 2 * w * w * w;
						img.wr(x, y) = lerp(img.wr(x, y), 1.0f, w);
					}
				}
			}
		}
	}
	void renderIt() {
		auto tex = gtex(img);
		
		gl::draw(tex, getWindowBounds());
	}
#endif
};
int WINAPI WinMain(HINSTANCE, HINSTANCE, LPSTR, int) {
	return mainFuncImpl(new SApp());
}
