/* 
 * File:   genViewsHardCode.h
 * Author: swl
 *
 * Created on January 18, 2016, 8:39 PM
 */

#ifndef GENVIEWSSHAPENET_H
#define	GENVIEWSSHAPENET_H

#include "ObjRenderer.h"
#include <vector>
#include <unistd.h>
#include <sys/types.h>
#include <dirent.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <FreeImagePlus.h>

struct Args
{
    int theta_inc;
    int phi_inc;
    int phi_max;
    float brightness;
    bool output_coord;
    unsigned render_size;
    unsigned output_size;
    bool reverse_normals;
};

void processModel(const std::string& rootPath, 
        const std::string& filePath,
        const Args& args)
{
    ObjRenderer::loadModel(rootPath+filePath);
    ObjRenderer::nextSeed();
    
    char cmd[1024];
    
    std::string viewFolderPath = rootPath+
            filePath.substr(0, filePath.length()-4)+"_views";
    
    sprintf(cmd, "rm -f -r %s", viewFolderPath.c_str());
    
    system(cmd);
    
    mkdir(viewFolderPath.c_str(), 0777);
    
    char fn[1024];
    
    
    
    for(int theta = 0; theta < 360; theta += args.theta_inc)
    {
        for(int phi = 0; phi <= args.phi_max; phi += args.phi_inc)
        {
            float t = theta * M_PI / 180;
            float p = phi * M_PI / 180;
            
            glm::vec3 eyePos(cos(p)*cos(t), sin(p), cos(p)*sin(t));
            
            cv::Mat4f image;
            cv::Mat4f aa_image;   
            
                    
            // output image
            ObjRenderer::setEyePos(eyePos*4.f);
            ObjRenderer::setShaderOutputID(2);
            image = ObjRenderer::genShading();
            
            unsigned filterSize = round(double(args.render_size) / args.output_size)*2+1;
            cv::GaussianBlur(image, image, cv::Size(filterSize, filterSize), 0, 0);
            cv::resize(image, aa_image, cv::Size(args.output_size, args.output_size));
            
            for(int i=0; i<aa_image.rows; i++)
            {
                for(int j=0; j<aa_image.cols; j++)
                {
                    cv::Vec4f c = aa_image.at<cv::Vec4f>(i, j);
                    aa_image.at<cv::Vec4f>(i, j) = cv::Vec4f(c[0]*args.brightness,
                            c[1]*args.brightness, c[2]*args.brightness, c[3]);
                }
            }
            
            sprintf(fn, "%s/%d_%d.png", viewFolderPath.c_str(), theta, phi);
            cv::imwrite(fn, aa_image*255.0);
            
            if(!args.output_coord)
                continue;
            
            // output vertex
            ObjRenderer::setShaderOutputID(1);
            image = ObjRenderer::genShading();
            cv::GaussianBlur(image, image, cv::Size(filterSize, filterSize), 0, 0);
            cv::resize(image, aa_image, cv::Size(args.output_size, args.output_size));
            
            sprintf(fn, "%s/%d_%d.exr", viewFolderPath.c_str(), theta, phi);
            
            fipImage hdr_image(FIT_RGBF, aa_image.cols, aa_image.rows, 96);
            cv::Vec3f* data = (cv::Vec3f*)hdr_image.accessPixels();
            
            for(int i=0; i<aa_image.rows; i++)
            {
                for(int j=0; j<aa_image.cols; j++)
                {
                    const cv::Vec4f& c = aa_image.at<cv::Vec4f>(aa_image.rows-i-1, j);
                    data[i*aa_image.rows+j][0] = c[0];
                    data[i*aa_image.rows+j][1] = c[1];
                    data[i*aa_image.rows+j][2] = c[2];
                }
            }
            
            hdr_image.save(fn);
        }
    }
}

void
findModelsInFolder(const std::string& root, std::vector<std::pair<std::string, std::string> >& pathList)
{
    DIR *dp;
    struct dirent *dirp;
    if((dp  = opendir(root.c_str())) == NULL) {
        std::cout << "Error(" << errno << ") opening " << root
                << root << std::endl;

    }

    while ((dirp = readdir(dp)) != NULL) {
        std::string path = std::string(dirp->d_name);
        if(path == "." || path == "..")
            continue;
        if(int(dirp->d_type) == 4)
        {
            findModelsInFolder(root+path+"/", pathList);
            continue;
        }
        
        if(path.substr(path.length()-4, 4) == ".obj")
        {
            pathList.push_back(std::make_pair(root, path));
        } 
    }
    closedir(dp);
}

void genViews(const std::string envPath, const std::string& folderPath, const Args& args)
{
    ObjRenderer::init(args.render_size);
    ObjRenderer::loadEnvMap(envPath, false);
    ObjRenderer::setReverseNormals(args.reverse_normals);
    
    glutHideWindow();
    
    std::vector<std::pair<std::string, std::string> > pathList;
    findModelsInFolder(folderPath, pathList);
    
    struct timeval ts, te;
    gettimeofday(&ts, NULL);
    for(unsigned i=0; i<pathList.size(); i++)
    {
        std::string fullPath = pathList[i].first + pathList[i].second;
        printf("Processing %s (%d / %d)\n", fullPath.c_str(), 
                i+1, pathList.size());
        processModel(pathList[i].first, pathList[i].second, args);
        gettimeofday(&te, NULL);
        double dt = te.tv_sec-ts.tv_sec+(te.tv_usec-ts.tv_usec)*1e-6;
        double eta = dt/(i+1)*(pathList.size()-i-1);
        printf("ETA = %02d:%02d:%02d\n", unsigned(eta)/60/60, 
                (unsigned(eta)/60)%60, unsigned(eta)%60);
    }
}

#endif	/* GENVIEWSHARDCODE_H */

