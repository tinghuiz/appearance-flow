#version 330

uniform sampler3D brdfTex;
in vec3 v, n, t, bn;
uniform vec3 eyePos;

#define PI 3.14159265358979

// For MERL BRDF data

vec4 shadeBRDF()
{
    vec3 pl = eyePos;

    vec3 bbn = cross(n, bn);
     
    vec3 in_vec = normalize(pl - v);

    float d = dot(in_vec, n);

    if(d < 0)
        return vec4(0);

    vec3 out_vec = normalize(eyePos - v);
    vec3 h_vec = normalize(in_vec + out_vec);
    float theta_h = acos(clamp(dot(h_vec, normalize(n)), -1, 1));
    float theta_d = acos(clamp(dot(h_vec, in_vec), -1, 1));
    float phi_d = atan(dot(bn, in_vec), dot(bbn, in_vec));

    ivec3 tSize = textureSize(brdfTex, 0);

    vec3 coord;
    coord.x = (phi_d/PI*(tSize.x-1.0) + 0.5) / tSize.x;
    coord.y = (theta_d*2/PI*(tSize.y-1.0) + 0.5) / tSize.y;
    coord.z = (theta_h*2/PI*(tSize.z-1.0) + 0.5) / tSize.z;
    //coord.x = 0.5;
    vec3 color = texture(brdfTex, coord).rgb;
    return vec4(d*color.rgb*3e-3+0.1, 1);
}
