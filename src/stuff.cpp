#include "precompiled.h"
#include "stuff.h"
#include "gpgpu.h"

int denormal_check::num;

std::map<string,string> FileCache::db;

string FileCache::get(string filename) {
	if(db.find(filename)==db.end()) {
		std::vector<unsigned char> buffer;
		loadFile(buffer,filename);
		string bufferStr(buffer.data(), buffer.data() + buffer.size());
		db[filename]=bufferStr;
	}
	return db[filename];
}

Array2D<vec3> resize(Array2D<vec3> src, ivec2 dstSize, const ci::FilterBase &filter)
{
	ci::SurfaceT<float> tmpSurface(
		(float*)src.data, src.w, src.h, /*rowBytes*/sizeof(vec3) * src.w, ci::SurfaceChannelOrder::RGB);
	auto resizedSurface = ci::ip::resizeCopy(tmpSurface, tmpSurface.getBounds(), dstSize, filter);
	Array2D<vec3> resultArray = resizedSurface;
	return resultArray;
}

Array2D<float> resize(Array2D<float> src, ivec2 dstSize, const ci::FilterBase &filter)
{
	ci::ChannelT<float> tmpSurface(
		src.w, src.h, /*rowBytes*/sizeof(float) * src.w, 1, src.data);
	ci::ChannelT<float> resizedSurface(dstSize.x, dstSize.y);
	ci::ip::resize(tmpSurface, &resizedSurface, filter);
	Array2D<float> resultArray = resizedSurface;
	return resultArray;
}

float sq(float f) {
	return f * f;
}

vector<float> getGaussianKernel(int ksize, float sigma) {
	vector<float> result;
	int r=ksize/2;
	float sum=0.0f;
	for(int i=-r;i<=r;i++) {
		float exponent = -(i*i/sq(2*sigma));
		float val = exp(exponent);
		sum += val;
		result.push_back(val);
	}
	for(int i=0; i<result.size(); i++) {
		result[i] /= sum;
	}
	return result;
}

float sigmaFromKsize(float ksize) {
	float sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8;
	return sigma;
}

float ksizeFromSigma(float sigma) {
	// ceil just to be sure
	int ksize = ceil(((sigma - 0.8) / 0.3 + 1) / 0.5 + 1);
	if(ksize % 2 == 0)
		ksize++;
	return ksize;
}

void disableGLReadClamp() {
	glClampColor(GL_CLAMP_READ_COLOR, GL_FALSE);
}

void enableDenormalFlushToZero() {
	_controlfp(_DN_FLUSH, _MCW_DN);
}

gl::TextureRef redToLuminance(gl::TextureRef const& in) {
	return shade2(in,
		"_out.rgb = vec3(fetch1());",
		ShadeOpts().ifmt(GL_RGBA16F)
	);
}
mat2 rotate(float angle) {
	float c = cos(angle), s = sin(angle);
	mat2 mat;
	mat[0] = vec2(c, -s);
	mat[1] = vec2(s, c);
	return mat;
}