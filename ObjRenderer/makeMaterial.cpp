#include "ObjRenderer.h"

std::shared_ptr<ShaderData>
ObjRenderer::makeMaterial(const tinyobj::material_t& mat, 
        const std::string& mtl_base_path)
{
    std::shared_ptr<ShaderData> data;
    if(mat.unknown_parameter.find("brdf") == mat.unknown_parameter.end())
    {
        data.reset(new ShaderDataPhong());
    }
    else
    {
        data.reset(new ShaderDataBRDF());
    }
    data->loadData(mat, mtl_base_path);
    return data;
}