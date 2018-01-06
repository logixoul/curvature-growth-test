#pragma once
#include "precompiled.h"

struct TextureCacheKey {
	ivec2 size;
	GLenum ifmt;
	
	bool operator==(const TextureCacheKey &other) const
	{
		return size == other.size
			&& ifmt == other.ifmt;
	}
};

namespace std {

	template <>
	struct hash<TextureCacheKey>
	{
		std::size_t operator()(const TextureCacheKey& k) const
		{
			return k.size.x ^ k.size.y ^ k.ifmt;
		}
	};

}

class TextureCache
{
public:
	gl::TextureRef get(TextureCacheKey const& key);

private:
	std::unordered_map<TextureCacheKey, gl::TextureRef> cache;
};
