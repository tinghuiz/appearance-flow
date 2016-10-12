/* 
 * File:   main.cpp
 * Author: swl
 *
 * Created on January 16, 2016, 2:05 PM
 */

#include "genViewsShapenet.h"
#include "Viewer.h"

int main(int argc, char** argv) 
{   
    /*ObjRenderer::init(800);
    ObjRenderer::setReverseNormals(true);
    ObjRenderer::loadEnvMap("envmaps/warehouse.hdr", false);
    ObjRenderer::setShaderOutputID(0);
    ObjRenderer::loadModel("models/car1/model.obj");
    Viewer::init();
    Viewer::run();
    return 0;*/
    
    Args args;
    char folderPath[1024];
    char envPath[1024];
    args.theta_inc = 30;
    args.phi_inc = 10;
    args.phi_max = 30;
    args.brightness = 1;
    args.output_coord = 1;
    args.render_size = 512;
    args.output_size = 256;
    
    FILE *file = fopen("config.txt", "r");
    fscanf(file, "folder_path = %s\n", &folderPath);
    fscanf(file, "envmap_path = %s\n", &envPath);
    fscanf(file, "theta_inc = %d\n", &args.theta_inc);
    fscanf(file, "phi_inc = %d\n", &args.phi_inc);
    fscanf(file, "phi_max = %d\n", &args.phi_max);
    fscanf(file, "output_coord = %d\n", &args.output_coord);
    fscanf(file, "render_size = %d\n", &args.render_size);
    fscanf(file, "output_size = %d\n", &args.output_size);
    fscanf(file, "reverse_normals = %d\n", &args.reverse_normals);
    fscanf(file, "brightness = %f\n", &args.brightness);
    fclose(file);
    
    genViews(envPath, folderPath, args);
    return 0;
}

