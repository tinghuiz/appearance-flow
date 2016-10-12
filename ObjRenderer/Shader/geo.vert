#version 330 core

// Input vertex data, different for all executions of this shader.
in vec3 vertex;
in vec3 normal;
in vec2 texCoord;
in vec3 binormal;

uniform mat4 viewMat;
uniform mat4 projMat;

out vec3 v, n, t;
out vec3 bn;
out float s;

void main()
{
    gl_Position = projMat * viewMat * vec4(vertex, 1);
    v = vertex;
    n = normal;
    bn = binormal;
    t = vec3(texCoord, 0);
}

