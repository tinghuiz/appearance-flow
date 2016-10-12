/* 
 * File:   ObjRenderer.cpp
 * Author: swl
 * 
 * Created on November 29, 2015, 12:02 PM
 */
#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"
#include <limits>
#include <GL/glew.h>
#include "ShaderUtils.h"
#include "ShaderData.h"
#include "ObjRenderer.h"
#define GLM_FORCE_RADIANS
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include <bits/unordered_map.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "ImageUtils.h"

#define BUFFER_OFFSET(i) ((char *)NULL + (i))

GLuint ObjRenderer::colorTexID = 0;
GLuint ObjRenderer::depthBufferID = 0;
GLuint ObjRenderer::frameBufferID = 0;
GLuint ObjRenderer::shaderProgID = 0;
GLuint ObjRenderer::vertexBufferID = 0;
GLuint ObjRenderer::renderSize = 512;
GLuint ObjRenderer::nTriangles = 0;
GLuint ObjRenderer::envmapID = 0;
GLuint ObjRenderer::blankTexID = 0;
unsigned ObjRenderer::shaderSeed = 0;
unsigned ObjRenderer::shaderOutputID = 0;
bool ObjRenderer::flipNormals = false;
bool ObjRenderer::faceNormals = false;
bool ObjRenderer::reverseNormals = false;
std::vector<ObjRenderer::MatGroupInfo> ObjRenderer::opaqueMatGroupInfoList;
std::vector<ObjRenderer::MatGroupInfo> ObjRenderer::transparentMatGroupInfoList;
std::vector<glm::vec3> ObjRenderer::vertices;
std::unordered_map<std::string, GLuint> ObjRenderer::texPath2texID;
std::unordered_map<std::string, GLuint> ObjRenderer::shaderTexName2texUnit;
glm::vec3 ObjRenderer::eyeFocus(0, 0, 0);
glm::vec3 ObjRenderer::eyeUp(0, 1, 0);
glm::vec3 ObjRenderer::eyePos(2, 2, 2);

GLuint ObjRenderer::getTexID(const std::string& path, bool flip)
{
    if (texPath2texID.find(path) == texPath2texID.end())
    {
        cv::Mat image = loadImage(path);
        if(image.dims == 0)
            texPath2texID[path] = 0;
        else
            texPath2texID[path] = makeTex(image, flip);
    }
    return texPath2texID[path];
}

void ObjRenderer::clearTextures()
{
    for (auto it = texPath2texID.begin(); it != texPath2texID.end(); it++)
    {
        glDeleteTextures(1, &it->second);
    }
    texPath2texID.clear();
}

void ObjRenderer::init(unsigned size)
{
    renderSize = size;
    int ac = 0;
    char** av;
    glutInit(&ac, av);
    glutInitWindowSize(renderSize, renderSize);
    glutCreateWindow("SketchRenderer");
    glewInit();

    shaderSeed = time(0);

    glutDisplayFunc(render);
    glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
    glClearColor(0, 0, 0, 0);

    std::vector<std::string> fragList;
    fragList.push_back("Shader/main.frag");
    fragList.push_back("Shader/coord.frag");
    fragList.push_back("Shader/phong.frag");
    fragList.push_back("Shader/brdf.frag");

    shaderProgID = loadShaders("Shader/geo.vert", fragList);

    glGenTextures(1, &colorTexID);
    glBindTexture(GL_TEXTURE_2D, colorTexID);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);

    //NULL means reserve texture memory, but texels are undefined
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, renderSize, renderSize,
            0, GL_RGBA, GL_FLOAT, NULL);
    //-------------------------
    glGenFramebuffers(1, &frameBufferID);
    glBindFramebuffer(GL_FRAMEBUFFER, frameBufferID);
    //Attach 2D texture to this FBO
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, colorTexID, 0);
    //-------------------------
    glGenRenderbuffers(1, &depthBufferID);
    glBindRenderbuffer(GL_RENDERBUFFER, depthBufferID);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT32,
            renderSize, renderSize);
    //-------------------------
    //Attach depth buffer to FBO
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthBufferID);

    glGenBuffers(1, &vertexBufferID);

    cv::Mat blank_image(8, 8, CV_8UC1, cv::Scalar(255));

    blankTexID = makeTex(blank_image);

}

GLuint ObjRenderer::makeTex(const std::vector<cv::Mat>& mipmap, bool flip)
{
    GLuint textureID;

    glGenTextures(1, &textureID);

    GLenum int_format = 0;
    GLenum format = 0;
    GLenum type = 0;

    GLint swizzleMask[] = {GL_RED, GL_GREEN, GL_BLUE, GL_ALPHA};
    switch (mipmap.front().type())
    {
        case CV_8UC4:
            type = GL_UNSIGNED_BYTE;
            format = GL_RGBA;
            int_format = GL_BGRA;
            break;
        case CV_8UC3:
            type = GL_UNSIGNED_BYTE;
            format = GL_RGB;
            int_format = GL_BGR;
            break;
        case CV_32FC1:
            type = GL_FLOAT;
            format = GL_R32F;
            int_format = GL_RED;
            swizzleMask[1] = swizzleMask[2] = swizzleMask[3] = GL_RED;
            break;
        case CV_8UC1:
            type = GL_UNSIGNED_BYTE;
            format = GL_RED;
            int_format = GL_RED;
            swizzleMask[1] = swizzleMask[2] = swizzleMask[3] = GL_RED;
            break;
        case CV_32FC3:
            type = GL_FLOAT;
            format = GL_RGB32F;
            int_format = GL_BGR;
            break;
        default:
            glDeleteTextures(1, &textureID);
            return 0;
    }

    switch (mipmap.front().dims)
    {
        case 2:
            glBindTexture(GL_TEXTURE_2D, textureID);

            glTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_RGBA, swizzleMask);

            for(int i=0; i<mipmap.size(); i++)
            {
                const cv::Mat& tex = mipmap[i];
                cv::Mat texFlip;
                if (flip)
                    cv::flip(tex, texFlip, 0);
                else
                    texFlip = tex;
                glTexImage2D(GL_TEXTURE_2D, i, format, tex.cols, tex.rows, 0,
                        int_format, type, texFlip.data);
            }
            
            if(mipmap.size() > 1)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, mipmap.size()-1);

            float fLargest;
            glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &fLargest);
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, fLargest);

            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);

            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);

            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

            if(mipmap.size() == 1)
                glGenerateMipmap(GL_TEXTURE_2D);

        case 3:
            glBindTexture(GL_TEXTURE_3D, textureID);

            glTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_RGBA, swizzleMask);

            for(int i=0; i<mipmap.size(); i++)
            {
                const cv::Mat& tex = mipmap[i];
                cv::Mat texFlip;
                if (flip)
                    cv::flip(tex, texFlip, 0);
                else
                    texFlip = tex;
                glTexImage3D(GL_TEXTURE_3D, i, format, tex.size[2], tex.size[1], tex.size[0], 0,
                    int_format, type, texFlip.data);
            }

            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);


            glTexParameterf(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_REPEAT);

            glTexParameterf(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_REPEAT);

            if(mipmap.size() == 1)
                glGenerateMipmap(GL_TEXTURE_3D);
            break;
    }

    return textureID;
}

void ObjRenderer::nextSeed()
{
    shaderSeed = shaderSeed * 86028121 + 236887691;
}

void ObjRenderer::loadEnvMap(const std::string& path, bool gray)
{
    cv::Mat colored_map = loadImage(path);
    cv::Mat map;

    if (gray)
    {
        cv::cvtColor(colored_map, map, CV_BGR2GRAY);
    } else
    {
        map = colored_map;
    }

    cv::pow(map, 1 / 2.2, map);
    
    int w=1, h=1;
    while(w<map.cols) w <<= 1;
    while(h<map.rows) h <<= 1;
    cv::resize(map, map, cv::Size(w, h));
    
    std::vector<cv::Mat> mipmap;
    for(int i=1; (1<<(i+1)) < std::min(w, h); i++)
    {
        mipmap.push_back(map);
        cv::pyrDown(map, map);
        if(gray)
            map = filter_envmap<float>(map);
        else
            map = filter_envmap<cv::Vec3f>(map);
    }

    glUseProgram(shaderProgID);

    envmapID = makeTex(mipmap);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT);
    
    useTexture("envmap", envmapID);
}

void ObjRenderer::useTexture(const std::string& shaderVarName, GLuint texID, GLenum type)
{
    GLuint unit = 0;
    if (shaderTexName2texUnit.find(shaderVarName) == shaderTexName2texUnit.end())
    {
        unit = shaderTexName2texUnit.size();
        shaderTexName2texUnit[shaderVarName] = unit;
    } else
    {
        unit = shaderTexName2texUnit[shaderVarName];
    }

    glUseProgram(shaderProgID);

    glUniform1i(glGetUniformLocation(shaderProgID, shaderVarName.c_str()), unit + 1);

    glActiveTexture(GL_TEXTURE1 + unit);
    glBindTexture(type, texID);
    if (texID == 0)
        glBindTexture(type, blankTexID);
    glActiveTexture(GL_TEXTURE1 + shaderTexName2texUnit.size());
}

void ObjRenderer::loadModel(const std::string& path, bool unitize)
{
    transparentMatGroupInfoList.clear();
    opaqueMatGroupInfoList.clear();
    clearTextures();

    std::string::size_type pos = path.rfind('/');
    std::string mtl_base_path = "";
    if (pos != std::string::npos)
    {
        mtl_base_path = path.substr(0, pos + 1);
    }
    // load obj -- begin
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string err;
    bool ret = tinyobj::LoadObj(shapes, materials, err, path.c_str(), mtl_base_path.c_str());
    if (!err.empty())
    { // `err` may contain warning message.
        std::cerr << err << std::endl;
    }
    if (!ret)
    {
        exit(1);
    }
    // load obj -- end

    // find bound -- begin
    glm::vec3 lb(std::numeric_limits<float>::max());
    glm::vec3 ub(std::numeric_limits<float>::min());

    for (size_t shape_index = 0; shape_index < shapes.size(); shape_index++)
    {
        const tinyobj::mesh_t & mesh = shapes[shape_index].mesh;
        for (size_t index = 0; index < mesh.positions.size(); index++)
        {
            const float pos = mesh.positions[index];
            lb[index % 3] = lb[index % 3] > pos ? pos : lb[index % 3];
            ub[index % 3] = ub[index % 3] < pos ? pos : ub[index % 3];
        }
    }

    glm::vec3 center = (ub + lb) / 2.0f;
    float diagLen = glm::length(ub - lb);
    // find bound -- end

    // make attribute buffers -- begin

    std::vector<Attribute> attributeData;
    std::unordered_map<unsigned, std::vector<Attribute> > mat_id2attrList;

    unsigned mat_id = 0;

    for (size_t shape_index = 0; shape_index < shapes.size(); shape_index++)
    {
        const tinyobj::mesh_t& mesh = shapes[shape_index].mesh;
        Attribute attr;

        for (size_t index = 0; index < mesh.indices.size(); index++)
        {
            const unsigned vert_index = mesh.indices[index];

            if (index % 3 == 0)
            {
                unsigned i1 = mesh.indices[index];
                unsigned i2 = mesh.indices[index + 1];
                unsigned i3 = mesh.indices[index + 2];
                glm::vec3 v1 = getVec<glm::vec3>(&mesh.positions[i1 * 3]);
                glm::vec3 v2 = getVec<glm::vec3>(&mesh.positions[i2 * 3]);
                glm::vec3 v3 = getVec<glm::vec3>(&mesh.positions[i3 * 3]);
                attr.normal = glm::normalize(glm::cross(v2 - v1, v3 - v2));
                if (flipNormals)
                    attr.normal = -attr.normal;

                if (materials.size() && index / 3 < mesh.material_ids.size())
                    mat_id = mesh.material_ids[index / 3];

                if (mat_id >= materials.size())
                    mat_id = 0;

                attr.binormal = glm::normalize(v2 - v1);

                if (i1 * 2 + 1 < mesh.texcoords.size() &&
                        i2 * 2 + 1 < mesh.texcoords.size() &&
                        i3 * 2 + 1 < mesh.texcoords.size())
                {
                    glm::vec2 tc1 = getVec<glm::vec2>(&mesh.texcoords[i1 * 2]);
                    glm::vec2 tc2 = getVec<glm::vec2>(&mesh.texcoords[i2 * 2]);
                    glm::vec2 tc3 = getVec<glm::vec2>(&mesh.texcoords[i3 * 2]);
                    glm::vec2 tc12 = tc2 - tc1;
                    glm::vec2 tc13 = tc3 - tc1;
                    attr.binormal = tc13.y * (v2 - v1) - tc12.y * (v3 - v1);
                    attr.binormal = glm::normalize(attr.binormal);
                }

            }

            if (!faceNormals && vert_index * 3 + 2 < mesh.normals.size())
            {
                attr.normal = getVec<glm::vec3>(&mesh.normals[vert_index * 3]);
                if(reverseNormals)
                    std::swap(attr.normal.x, attr.normal.z);
                if (flipNormals)
                    attr.normal = -attr.normal;
            }

            if (vert_index * 2 + 1 < mesh.texcoords.size())
            {
                attr.texCoord = getVec<glm::vec2>(&mesh.texcoords[vert_index * 2]);
            }

            attr.vertex = getVec<glm::vec3>(&mesh.positions[vert_index * 3]);

            if (unitize)
            {
                glm::vec3 unit_pos;
                unit_pos = getVec<glm::vec3>(&mesh.positions[vert_index * 3]);
                unit_pos = (unit_pos - center) * 2.0f / diagLen;
                attr.vertex = unit_pos;
            }
            attributeData.push_back(attr);
            mat_id2attrList[mat_id].push_back(attr);
        }
    }

    unsigned current_size = 0;
    // iterate through material groups

    for (auto it = mat_id2attrList.begin(); it != mat_id2attrList.end(); it++)
    {
        MatGroupInfo info;
        info.offset = current_size;
        info.size = it->second.size();

        for (unsigned i = 0; i < it->second.size(); i++)
            attributeData[current_size + i] = it->second[i];
        current_size += it->second.size();

        if (it->first < materials.size())
        {
            const tinyobj::material_t& mat = materials[it->first];
            info.shaderData = makeMaterial(mat, mtl_base_path);
        }
        if (info.shaderData->transparent())
            transparentMatGroupInfoList.push_back(info);
        else
            opaqueMatGroupInfoList.push_back(info);
    }

    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferID);

    glBufferData(GL_ARRAY_BUFFER, sizeof (Attribute) * attributeData.size(),
            attributeData.data(), GL_STATIC_DRAW);



    GLuint loc;

    loc = glGetAttribLocation(shaderProgID, "vertex");
    glEnableVertexAttribArray(loc);
    glVertexAttribPointer(loc, 3, GL_FLOAT, GL_FALSE,
            sizeof (Attribute), BUFFER_OFFSET(offsetof(Attribute, vertex)));

    loc = glGetAttribLocation(shaderProgID, "normal");
    glEnableVertexAttribArray(loc);
    glVertexAttribPointer(loc, 3, GL_FLOAT, GL_FALSE,
            sizeof (Attribute), BUFFER_OFFSET(offsetof(Attribute, normal)));

    loc = glGetAttribLocation(shaderProgID, "texCoord");
    glEnableVertexAttribArray(loc);
    glVertexAttribPointer(loc, 2, GL_FLOAT, GL_FALSE,
            sizeof (Attribute), BUFFER_OFFSET(offsetof(Attribute, texCoord)));

    loc = glGetAttribLocation(shaderProgID, "binormal");
    glEnableVertexAttribArray(loc);
    glVertexAttribPointer(loc, 3, GL_FLOAT, GL_FALSE,
            sizeof (Attribute), BUFFER_OFFSET(offsetof(Attribute, binormal)));

    // make attribute buffers -- end
}

void ObjRenderer::renderView()
{

    glBindFramebuffer(GL_FRAMEBUFFER, frameBufferID);
    glUseProgram(shaderProgID);

    glUniform1ui(glGetUniformLocation(shaderProgID, "seed"), shaderSeed);
    glUniform1ui(glGetUniformLocation(shaderProgID, "outputID"), shaderOutputID);

    glm::vec3 front = eyeFocus - eyePos;

    glm::vec3 right = glm::normalize(glm::cross(front, eyeUp));
    glm::vec3 rectUp = glm::cross(right, front);

    glm::mat4 projMat = glm::perspective<float>(30*M_PI/180.0, 1, 0.01, 100);

    glUniformMatrix4fv(glGetUniformLocation(shaderProgID, "projMat"),
            1, false, (float*) &projMat);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glPushAttrib(GL_ALL_ATTRIB_BITS);

    glEnable(GL_DEPTH_TEST);

    glm::mat4 viewMat = glm::lookAt(eyePos, eyeFocus, rectUp);

    glUniformMatrix4fv(glGetUniformLocation(shaderProgID, "viewMat"),
            1, false, (float*) &viewMat);

    glUniform3fv(glGetUniformLocation(shaderProgID, "eyePos"),
            1, (float*) &eyePos);

    useTexture("envmap", envmapID);

    for (unsigned i = 0; i < opaqueMatGroupInfoList.size(); i++)
    {
        const MatGroupInfo &info = opaqueMatGroupInfoList[i];
        info.shaderData->send2shader(shaderProgID);
        if (shaderOutputID > 0)
            glUniform1ui(glGetUniformLocation(shaderProgID, "outputID"), shaderOutputID);
        glDrawArrays(GL_TRIANGLES, info.offset, info.size);
    }
    
    
    glEnable(GL_CULL_FACE); // enable cull face to handle double sided transparent object.
    glCullFace(GL_FRONT);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    for (unsigned i = 0; i < transparentMatGroupInfoList.size(); i++)
    {
        const MatGroupInfo &info = transparentMatGroupInfoList[i];
        sortMatGroupTriangles(info);
        info.shaderData->send2shader(shaderProgID);
        if (shaderOutputID > 0)
            glUniform1ui(glGetUniformLocation(shaderProgID, "outputID"), shaderOutputID);
        glDrawArrays(GL_TRIANGLES, info.offset, info.size);
    }
    glDisable(GL_BLEND);
    glDisable(GL_CULL_FACE); 
    glPopAttrib();
}

template<typename T> void quicksort(T a[], const int size)
{
    if(size <= 1)
        return;
    int l = 0;
    int r = size-1;
    T pivotValue = a[random()%size];
    while(1)
    {
        while(l < r && a[l] < pivotValue) l++;
        if(l == r) break;
        while(l < r && !(a[r] < pivotValue)) r--;
        if(l == r) break;
        T temp = a[l];
        a[l] = a[r];
        a[r] = temp;
    }
    if(r == size-1 || r == 0) return;
    quicksort(a, r);
    quicksort(a+r, size - r);
}

void ObjRenderer::sortMatGroupTriangles(const MatGroupInfo& info)
{

    struct Triangle
    {
        unsigned index;
        glm::vec3 eyePos;
        glm::vec3 v[3];

        bool operator<(const Triangle& tri) const
        {
            glm::vec3 n = glm::normalize(glm::cross(v[1] - v[0], v[2] - v[1]));
            float d[3];
            for (int k = 0; k < 3; k++)
                d[k] = glm::dot(n, tri.v[k] - v[0]);
            if(d[0] * d[1] >= 0 && d[1] * d[2] >= 0 && d[0] * d[2] >= 0); // triangle 1 can be a shield
                return glm::dot(eyePos - v[0], n) * glm::dot(tri.v[0] - v[0], n) < 0;

            n = glm::normalize(glm::cross(tri.v[1] - tri.v[0], tri.v[2] - tri.v[1]));
            for (int k = 0; k < 3; k++)
                d[k] = glm::dot(n, v[k] - tri.v[0]);
            if(d[0] * d[1] >= 0 && d[1] * d[2] >= 0 && d[0] * d[2] >= 0) // triangle 2 can be a shield
                return glm::dot(eyePos - tri.v[0], n) * glm::dot(v[0] - tri.v[0], n) > 0;
            return index < tri.index;
        }
    };
    std::vector<Attribute> inList(info.size);
    std::vector<Attribute> outList(info.size);
    std::vector<Triangle> triList(info.size / 3);
    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferID);
    glGetBufferSubData(GL_ARRAY_BUFFER, info.offset * sizeof (Attribute),
            info.size * sizeof (Attribute), inList.data());
    for (unsigned i = 0; i < triList.size(); i++)
    {
        triList[i].eyePos = eyePos;
        triList[i].index = i;
        for (int k = 0; k < 3; k++)
        {
            triList[i].v[k] = inList[i * 3 + k].vertex;
        }
    }

    quicksort<Triangle>(triList.data(), triList.size());

    for (unsigned i = 0; i < triList.size(); i++)
    {
        for (int k = 0; k < 3; k++)
        {
            outList[i * 3 + k] = inList[triList[i].index * 3 + k];
        }
    }
    glBufferSubData(GL_ARRAY_BUFFER, info.offset * sizeof (Attribute),
            info.size * sizeof (Attribute), outList.data());
}

cv::Mat4f ObjRenderer::genShading()
{
    cv::Mat4f image(renderSize, renderSize);
    image.setTo(0.0);

    renderView();
    glReadPixels(0, 0, renderSize, renderSize, GL_RGBA, GL_FLOAT, image.data);
    cv::flip(image, image, 0);
    cv::cvtColor(image, image, CV_RGBA2BGRA);
    return image;
}

void ObjRenderer::render()
{
    renderView();

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glUseProgram(0);
    
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, colorTexID);
    glEnable(GL_TEXTURE_2D);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, 1, 0, 1);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glColor4f(1, 1, 1, 1);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    glBegin(GL_QUADS);
    glTexCoord2f(0, 0);
    glVertex2f(0, 0);
    glTexCoord2f(0, 1);
    glVertex2f(0, 1);
    glTexCoord2f(1, 1);
    glVertex2f(1, 1);
    glTexCoord2f(1, 0);
    glVertex2f(1, 0);
    glEnd();

    glDisable(GL_TEXTURE_2D);

    glFlush();
    glutSwapBuffers();
}