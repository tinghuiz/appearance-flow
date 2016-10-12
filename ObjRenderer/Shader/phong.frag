#version 330

uniform sampler2D envmap;
uniform sampler2D diffTex;
uniform uint seed;
uniform vec3 ka;
uniform vec3 kd;
uniform vec3 ks;
uniform float d;
uniform float s;

in vec3 v, n, t;

uniform vec3 eyePos;

out vec4 color;

#define PI 3.14159265358979

#define RAND_MAX 1000u

uint next_rand(uint seed)
{
    return seed * 15485863u + 32452843u;
}

float randf(uint seed) 
{
    return float(seed % RAND_MAX) / float(RAND_MAX-1u);
}

mat3 rotationMatrix(vec3 axis, float angle)
{
    axis = normalize(axis);
    float s = sin(angle);
    float c = cos(angle);
    float oc = 1.0 - c;
    
    return mat3(oc * axis.x * axis.x + c,           oc * axis.x * axis.y - axis.z * s,  oc * axis.z * axis.x + axis.y * s,
                oc * axis.x * axis.y + axis.z * s,  oc * axis.y * axis.y + c,           oc * axis.y * axis.z - axis.x * s,
                oc * axis.z * axis.x - axis.y * s,  oc * axis.y * axis.z + axis.x * s,  oc * axis.z * axis.z + c);
}

vec3 getColor(sampler2D map, vec3 dir, float s = 1.0) 
{
    uint local_seed = next_rand(seed);
    float delta_phi = randf(local_seed);
    local_seed = next_rand(local_seed);
    float delta_theta = randf(local_seed);
    local_seed = next_rand(local_seed);
    float angle = randf(local_seed)*2.0*PI;
    delta_phi *= PI/6.0;
    delta_theta *= 2.0*PI;

    dir = rotationMatrix(vec3(cos(angle), 0, sin(angle)), delta_phi) * dir;
    dir = rotationMatrix(vec3(0, 1, 0), delta_theta) * dir;

    vec2 uv;
    float theta = atan(dir.z, dir.x)+PI;
    float phi = acos(clamp(dir.y, -1, 1));
    uv.x = theta / (2*PI);
    uv.y = phi / PI;

    float delta_shininess = acos(clamp(pow(0.9, 1/s), -1, 1))/PI*2.0;

    float drdx = length(dFdx(dir));
    float drdy = length(dFdy(dir));
    float dr = max(drdx, drdy);
    float da = max(asin(dr*0.5)*2.0/PI, delta_shininess);
    ivec2 size = textureSize(map, 0);
    da *= sqrt(float(size.x*size.y));
    float mml = max(log2(da), 0);
    return textureLod(map, vec2(uv.x, uv.y), mml).rgb;
}

vec4 shadeGlass(float refr_idx)
{
    vec3 in_vec = normalize(eyePos - v);
    vec3 out_vec = dot(n, in_vec) * n * 2 - in_vec;
    vec3 refl_color = getColor(envmap, out_vec, 1e5)*0.1;
    float _cos = 1-abs(dot(in_vec, n));
    float _cos5 = _cos*_cos;
    _cos5 *= _cos5*_cos;
    float r = (refr_idx-1.f)/(refr_idx+1.f);
    r *= r;
    float refl_ratio = r + (1-r)*_cos5;
    refl_ratio = refl_ratio*0.5+0.5;
    return vec4(refl_color, refl_ratio);
}

vec4 shadePhong()
{
    if(d < 1)
        return shadeGlass(1.5);
    vec3 in_vec = normalize(eyePos - v);
    vec3 normal = normalize(n);
    vec3 out_vec = dot(normal, in_vec) * normal * 2 - in_vec;
    vec3 c = kd * getColor(envmap, normal);
    c *= texture(diffTex, t.xy).rgb;
    c += ka;
    c += ks * getColor(envmap, normalize(out_vec), s);
    return vec4(c, d);
}


