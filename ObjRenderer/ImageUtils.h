/* 
 * File:   ImageUtils.h
 * Author: swl
 *
 * Created on January 18, 2016, 2:16 PM
 */

#ifndef IMAGEUTILS_H
#define	IMAGEUTILS_H

#include <opencv/cv.h>
#include <FreeImagePlus.h>

cv::Mat fi2mat(FIBITMAP* src)
{
    cv::Mat dst;
    //FIT_BITMAP    //standard image : 1 - , 4 - , 8 - , 16 - , 24 - , 32 - bit
    //FIT_UINT16    //array of unsigned short : unsigned 16 - bit
    //FIT_INT16     //array of short : signed 16 - bit
    //FIT_UINT32    //array of unsigned long : unsigned 32 - bit
    //FIT_INT32     //array of long : signed 32 - bit
    //FIT_FLOAT     //array of float : 32 - bit IEEE floating point
    //FIT_DOUBLE    //array of double : 64 - bit IEEE floating point
    //FIT_COMPLEX   //array of FICOMPLEX : 2 x 64 - bit IEEE floating point
    //FIT_RGB16     //48 - bit RGB image : 3 x 16 - bit
    //FIT_RGBA16    //64 - bit RGBA image : 4 x 16 - bit
    //FIT_RGBF      //96 - bit RGB float image : 3 x 32 - bit IEEE floating point
    //FIT_RGBAF     //128 - bit RGBA float image : 4 x 32 - bit IEEE floating point

    int bpp = FreeImage_GetBPP(src);
    FREE_IMAGE_TYPE fit = FreeImage_GetImageType(src);

    int cv_type = -1;
    int cv_cvt = -1;
    
    switch (fit)
    {
    case FIT_UINT16: cv_type = cv::DataType<ushort>::type; break;
    case FIT_INT16: cv_type = cv::DataType<short>::type; break;
    case FIT_UINT32: cv_type = cv::DataType<unsigned>::type; break;
    case FIT_INT32: cv_type = cv::DataType<int>::type; break;
    case FIT_FLOAT: cv_type = cv::DataType<float>::type; break;
    case FIT_DOUBLE: cv_type = cv::DataType<double>::type; break;
    case FIT_COMPLEX: cv_type = cv::DataType<cv::Complex<double>>::type; break;
    case FIT_RGB16: cv_type = cv::DataType<cv::Vec<ushort, 3>>::type; cv_cvt = CV_RGB2BGR; break;
    case FIT_RGBA16: cv_type = cv::DataType<cv::Vec<ushort, 4>>::type; cv_cvt = CV_RGBA2BGRA; break;
    case FIT_RGBF: cv_type = cv::DataType<cv::Vec<float, 3>>::type; cv_cvt = CV_RGB2BGR; break;
    case FIT_RGBAF: cv_type = cv::DataType<cv::Vec<float, 4>>::type; cv_cvt = CV_RGBA2BGRA; break;
    case FIT_BITMAP:
        switch (bpp) {
        case 8: cv_type = cv::DataType<cv::Vec<uchar, 1>>::type; break;
        case 16: cv_type = cv::DataType<cv::Vec<uchar, 2>>::type; break;
        case 24: cv_type = cv::DataType<cv::Vec<uchar, 3>>::type; break;
        case 32: cv_type = cv::DataType<cv::Vec<uchar, 4>>::type; break;
        default:
            // 1, 4 // Unsupported natively
            cv_type = -1;
        }
        break;
    default:
        // FIT_UNKNOWN // unknown type
        dst = cv::Mat(); // return empty Mat
        return dst;
    }

    int width = FreeImage_GetWidth(src);
    int height = FreeImage_GetHeight(src);
    int step = FreeImage_GetPitch(src);
    if (cv_type >= 0) {
        dst = cv::Mat(height, width, cv_type, FreeImage_GetBits(src), step);
        if (cv_cvt > 0)
        {
            cv::cvtColor(dst, dst, cv_cvt);
        }
    }
    else {

        cv::vector<uchar> lut;
        int n = pow(2, bpp);
        for (int i = 0; i < n; ++i)
        {
            lut.push_back(static_cast<uchar>((255 / (n - 1))*i));
        }

        FIBITMAP* palletized = FreeImage_ConvertTo8Bits(src);
        BYTE* data = FreeImage_GetBits(src);
        for (int r = 0; r < height; ++r) {
            for (int c = 0; c < width; ++c) {
                dst.at<uchar>(r, c) = cv::saturate_cast<uchar>(lut[data[r*step + c]]);
            }
        }
    }
    return dst.clone();
}

template<typename T> 
cv::Mat filter_envmap(const cv::Mat& map)
{
    cv::Mat filtered;
    map.copyTo(filtered);
    for(unsigned i=0; i<map.rows; i++)
    {
        float sin_phi = sin(i/float(map.rows-1)*M_PI);
        int filter_len = map.cols; 
        if(sin_phi*map.cols > 1.0)
            filter_len = int(1.0/sin_phi);
        filtered.at<T>(i, 0) = 0;
        for(int j=0; j<filter_len; j++)
        {
            int index = j-filter_len/2;
            if(index < 0) index += map.cols;
            if(index >= map.cols) index -= map.cols;
            filtered.at<T>(i, 0) += map.at<T>(i, index)/filter_len;
        }
        for(int j=1; j<map.cols; j++)
        {
            filtered.at<T>(i, j) = filtered.at<T>(i, j-1);
            int index = j+filter_len-filter_len/2-1;
            if(index >= map.cols) index -= int(map.cols);
            filtered.at<T>(i, j) += map.at<T>(i, index)/filter_len;
            index = j-filter_len/2-1;
            if(index < 0) index += int(map.cols);
            filtered.at<T>(i, j) -= map.at<T>(i, index)/filter_len;
        }
    }
    return filtered;
}

cv::Mat loadImage(const std::string& path)
{
    cv::Mat image;
    
    fipImage fip_map;
    
    if(path.substr(path.length()-3, 3) == "hdr" ||
        path.substr(path.length()-3, 3) == "tga")
    {
        fip_map.load(path.c_str());
        image = fi2mat(fip_map);
        if(path.substr(path.length()-3, 3) == "tga")
            cv::flip(image, image, 0);
    }
    else if(path.substr(path.length()-6, 6) == "binary") // read brdf file
    {
    
        FILE* file = fopen(path.c_str(), "rb");
        cv::Vec3i dim;

        fread(&dim[0], sizeof(cv::Vec3i), 1, file);
        std::vector<double> data(dim[0]*dim[1]*dim[2]*3);
        fread(data.data(), sizeof(double), data.size(), file);

        image = cv::Mat3f(3, &dim[0]);
        
        size_t size = dim[0]*dim[1]*dim[2];

        for(int i=0; i<image.size[0]; i++)
        {
            for(int j=0; j<image.size[1]; j++)
            {
                for(int k=0; k<image.size[2]; k++)
                {
                    size_t index = i*image.size[1]*image.size[2];
                    index += j*image.size[2] + k;
                    cv::Vec3f& color = image.at<cv::Vec3f>(i, j, k);
                    color[0] = data[index+size*2];
                    color[1] = data[index+size];
                    color[2] = data[index];
                }
            }
        }
        
        fclose(file);
    }
    else
    {
        image = cv::imread(path);
    }
    return image;
}

#endif	/* IMAGEUTILS_H */

