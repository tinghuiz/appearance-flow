/* 
 * File:   ShaderData.h
 * Author: swl
 *
 * Created on February 6, 2016, 10:02 AM
 */

#ifndef SHADERDATA_H
#define	SHADERDATA_H

#include "ShaderUtils.h"
#include <glm/glm.hpp>
#include "tiny_obj_loader.h"

class ShaderData
{
public:
    virtual void loadData(const tinyobj::material_t& mat, 
            const std::string& mtl_base_path){}
    virtual void send2shader(GLuint shaderProgID) const {}
    virtual bool transparent() const { return false; }
};

class ShaderDataPhong : public ShaderData
{
protected:
    unsigned outputID;
    glm::vec3 ka;
    glm::vec3 kd;
    glm::vec3 ks;
    float d;
    float s;
    GLuint diffTexID;
public:
    void send2shader(GLuint shaderProgID) const;
    void loadData(const tinyobj::material_t& mat, 
        const std::string& mtl_base_path);
    virtual bool transparent() const { return d < 1; }
};

class ShaderDataBRDF : public ShaderData
{
protected:
    unsigned outputID;
    GLuint brdfTexID;
public:
    void send2shader(GLuint shaderProgID) const;
    void loadData(const tinyobj::material_t& mat, 
        const std::string& mtl_base_path);
};

#endif	/* SHADERDATA_H */

