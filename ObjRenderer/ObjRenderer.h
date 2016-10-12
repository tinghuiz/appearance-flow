/* 
 * File:   ObjRenderer.h
 * Author: swl
 *
 * Created on November 29, 2015, 12:02 PM
 */

#ifndef OBJRENDERER_H
#define	OBJRENDERER_H

#include "ShaderData.h"

#include <opencv2/opencv.hpp>
#include <memory>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <stdio.h>
#include <vector>
#include <unordered_map>
#include "tiny_obj_loader.h"

template<typename T> inline T getVec(const float* data)
{
    T v;
    memcpy(&v, data, sizeof(T));
    return v;
}

class ObjRenderer
{
    friend class ShaderDataPhong;
    friend class ShaderDataBRDF;
public:
    
    static void init(unsigned size = 512);
    static void nextSeed();
    static void loadEnvMap(const std::string& path, bool gray = false);
    static void loadModel(const std::string& path, bool unitize = true);
    static void setEyeUp(const glm::vec3& up) { eyeUp = up; }
    static void setEyeFocus(const glm::vec3& focus) { eyeFocus = focus; }
    static void setEyePos(const glm::vec3& pos) { eyePos = pos; }
    static cv::Mat4f genShading();
    static void setReverseNormals(bool reverse) { reverseNormals = reverse; }
    static void setShaderOutputID(int id) 
    { 
        shaderOutputID = id;
    }
    
protected:
    struct MatGroupInfo
    {
        unsigned offset;
        unsigned size;
        std::shared_ptr<ShaderData> shaderData;
    };
    static std::vector<MatGroupInfo> opaqueMatGroupInfoList;
    static std::vector<MatGroupInfo> transparentMatGroupInfoList;
    struct Attribute
    {
        glm::vec3 vertex;
        glm::vec3 normal;
        glm::vec2 texCoord;
        glm::vec3 binormal;
    };
    static void render();
    static void sortMatGroupTriangles(const MatGroupInfo& info);
    static glm::vec3 eyePos, eyeFocus, eyeUp;
    static void renderView();
    static void clearTextures();
    static GLuint makeTex(const cv::Mat& tex, bool flip = true) 
    { 
        return makeTex(std::vector<cv::Mat>(1, tex), flip); 
    }
    static GLuint makeTex(const std::vector<cv::Mat>& mipmap, bool flip = true);
    static GLuint getTexID(const std::string& path, bool flip = true);
    static std::shared_ptr<ShaderData> makeMaterial(const tinyobj::material_t& mat, 
        const std::string& mtl_base_path);
    static void useTexture(const std::string& shaderVarName, GLuint texID, GLenum type = GL_TEXTURE_2D);
    static std::vector<glm::vec3> vertices;
    static GLuint colorTexID;
    static GLuint depthBufferID;
    static GLuint frameBufferID;
    static GLuint shaderProgID;
    static GLuint renderSize;
    static GLuint vertexBufferID;
    static GLuint nTriangles;
    static GLuint envmapID;
    static GLuint blankTexID;
    static GLuint forceOutputID;
    static unsigned shaderSeed;
    static unsigned shaderOutputID;
    static bool flipNormals;
    static bool reverseNormals;
    static bool faceNormals;
    static std::unordered_map<std::string, GLuint> shaderTexName2texUnit;
    static std::unordered_map<std::string, GLuint> texPath2texID;
};

#endif	/* SKETCHRENDERER_H */

