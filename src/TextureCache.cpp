#include "precompiled.h"
#include "TextureCache.h"
#include "stuff.h"

gl::TextureRef TextureCache::get(TextureCacheKey const & key)
{
	auto it = cache.find(key);
	if (it == cache.end()) {
		gl::TextureRef tex = maketex(key.size.x, key.size.y, key.ifmt);
		cache.insert(std::make_pair(key, tex));
		return tex;
	}
	else {
		return it->second;
	}
}
