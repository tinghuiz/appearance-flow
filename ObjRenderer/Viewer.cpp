/* 
 * File:   Viewer.cpp
 * Author: swl
 * 
 * Created on February 5, 2016, 5:07 PM
 */

#include "Viewer.h"
#include "ObjRenderer.h"

float theta = M_PI/4;
float phi = M_PI/12;
float dist = 4;

int mouse_x, mouse_y;

void mouseFunc(int button, int state, int x, int y)
{
    if(button == GLUT_LEFT_BUTTON && state == GLUT_DOWN)
    {
        mouse_x = x;
        mouse_y = y;
    }
    else if(button == 3)
        dist *= 1.05;
    else if(button == 4)
        dist *= 0.95;
}

void motionFunc(int x, int y)
{
    int dx = x - mouse_x;
    int dy = y - mouse_y;
    theta += dx*0.01;
    if(theta > 2*M_PI)
        theta -= 2*M_PI;
    if(theta < -2*M_PI)
        theta += 2*M_PI;
    phi -= dy*0.01;
    if(phi < M_PI/12)
        phi = M_PI/12;
    if(phi > 11*M_PI/12)
        phi = 11*M_PI/12;
    mouse_x = x;
    mouse_y = y;
}

void keyboardFunc(unsigned char key, int x, int y)
{
    if(key >= '1' && key <= '9')
        ObjRenderer::setShaderOutputID(key-'0');
    switch(key)
    {
    case 27:
        exit(0);
        break;
    }
}

void idleFunc()
{
    ObjRenderer::setEyePos(dist*glm::vec3(sin(phi)*cos(theta), cos(phi), sin(phi)*sin(theta)));
    glutPostRedisplay();
}

void Viewer::run()
{
    glutShowWindow();
    glutMainLoop();
}

void Viewer::init()
{
    glutMotionFunc(motionFunc);
    glutMouseFunc(mouseFunc);
    glutIdleFunc(idleFunc);
    glutKeyboardFunc(keyboardFunc);
    ObjRenderer::setEyePos(dist*glm::vec3(sin(phi)*cos(theta), cos(phi), sin(phi)*sin(theta)));
}

