/* 
 * File:   ShaderUtils.h
 * Author: swl
 *
 * Created on January 16, 2016, 7:36 PM
 */

#ifndef SHADERUTILS_H
#define	SHADERUTILS_H

#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/freeglut.h>
#include <iostream>
#include <vector>

inline char *textFileRead(const char *fn) 
{
    FILE *fp;
    char *content = NULL;

    int count=0;

    if (fn != NULL) {

        fp = fopen(fn,"rt");
        if (fp != NULL) {

            fseek(fp, 0, SEEK_END);
            count = ftell(fp);
            rewind(fp);

            if (count > 0) {
                content = (char *)malloc(sizeof(char) * (count+1));
                count = fread(content,sizeof(char),count,fp);
                content[count] = '\0';
            }
            fclose(fp);

        }
    }

    return content;
}

inline void printShaderInfoLog(GLuint obj)
{
    int infologLength = 0;
    int charsWritten  = 0;
    char *infoLog;

        glGetShaderiv(obj, GL_INFO_LOG_LENGTH,&infologLength);

    if (infologLength > 0)
    {
        infoLog = (char *)malloc(infologLength);
        glGetShaderInfoLog(obj, infologLength, &charsWritten, infoLog);
        printf("%s",infoLog);
        free(infoLog);
    }
}

inline void printProgramInfoLog(GLuint obj)
{
    int infologLength = 0;
    int charsWritten  = 0;
    char *infoLog;

        glGetProgramiv(obj, GL_INFO_LOG_LENGTH,&infologLength);

    if (infologLength > 0)
    {
        infoLog = (char *)malloc(infologLength);
        glGetProgramInfoLog(obj, infologLength, &charsWritten, infoLog);
                printf("%s\n",infoLog);
        free(infoLog);
    }
}

inline GLuint loadShaders(const std::vector<std::string>& vertList, const std::vector<std::string>& fragList) 
{
    char *vs,*fs;

    GLuint p = glCreateProgram();

    for(unsigned i=0; i<vertList.size(); i++)
    {
        GLuint v = glCreateShader(GL_VERTEX_SHADER);
        vs = textFileRead(vertList[i].c_str());
        const char * vv = vs;
        glShaderSource(v, 1, &vv,NULL);
        free(vs);
        glCompileShader(v);
        printShaderInfoLog(v);
        glAttachShader(p,v);
        glDeleteShader(v);
    }

    for(unsigned i=0; i<fragList.size(); i++)
    {
        GLuint f = glCreateShader(GL_FRAGMENT_SHADER);	
        fs = textFileRead(fragList[i].c_str());
        const char * ff = fs;
        glShaderSource(f, 1, &ff,NULL);
        free(fs);
        glCompileShader(f);
        printShaderInfoLog(f);
        glAttachShader(p,f);
        glDeleteShader(f);
    }

    glLinkProgram(p);
    printProgramInfoLog(p);

    glUseProgram(p);

    return p;
}

inline GLuint loadShaders(const char* vert, const char* frag) 
{     
    return loadShaders(std::vector<std::string>(1, vert), std::vector<std::string>(1, frag));
}

inline GLuint loadShaders(const char* vert, const std::vector<std::string>& fragList) 
{     
    return loadShaders(std::vector<std::string>(1, vert), fragList);
}




#endif	/* SHADERUTILS_H */

