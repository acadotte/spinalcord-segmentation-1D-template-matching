#ifndef vec3HPP
#define vec3HPP

#include <cmath>

/// Minimal 3-dimensional vector abstraction
class vec3 {
public:

	// Constructors
	vec3() :
			x(0), y(0), z(0) {
	}

	vec3(float vx, float vy, float vz) {
		x = vx;
		y = vy;
		z = vz;
	}

	vec3(const vec3& v) {
		x = v.x;
		y = v.y;
		z = v.z;
	}

	// Destructor
	~vec3() {
	}

	// A minimal set of vector operations
	vec3 operator *(float mult) const // result = this * arg
			{
		return vec3(x * mult, y * mult, z * mult);
	}

	float operator *(const vec3& v) const // result = this * arg
			{
		return x * v.x + y * v.y + z * v.z;
	}

	vec3 operator /(float div) const // result = this * arg
			{
		return vec3(x / div, y / div, z / div);
	}

	vec3 operator +(const vec3& v) const // result = this + arg
			{
		return vec3(x + v.x, y + v.y, z + v.z);
	}

	vec3 operator -(const vec3& v) const // result = this - arg
			{
		return vec3(x - v.x, y - v.y, z - v.z);
	}

	float norm2() {
		return x * x + y * y + z * z;
	}

	float norm() {
		return sqrt(x * x + y * y + z * z);
	}

	void normalize() {
		float n = norm();
		if (n == 0)
			return;

		x = x / n;
		y = y / n;
		z = z / n;
	}

	vec3 cross(const vec3& v) const	{
		return vec3(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x);
	}

	// rotate v around this in theta radians.
	vec3 rotatev(const vec3& v0, float theta) {
		float x = v0.x;
		float y = v0.y;
		float z = v0.z;
		float u = this->x;
		float v = this->y;
		float w = this->z;

		float r2 = (u * u + v * v + w * w);
		float r = sqrt(r2);

		float t1 = (u * x + v * y + w * z) * (1 - cos(theta));
		float t2 = r2 * cos(theta);
		float t3 = r * sin(theta);

		vec3 rv;

		rv.x = 1 / r2 * (u * t1 + x * t2 + (-w * y + v * z) * t3);
		rv.y = 1 / r2 * (v * t1 + y * t2 + (w * x - u * z) * t3);
		rv.z = 1 / r2 * (w * t1 + z * t2 + (-v * x + u * y) * t3);

		return rv;
	}

	float x, y, z;
};

#endif
