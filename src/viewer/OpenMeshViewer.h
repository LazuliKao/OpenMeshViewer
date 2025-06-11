#pragma once

#include <iostream>
#include <string>
#include <QMainWindow>
#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QOpenGLBuffer>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLShaderProgram>
#include <QMouseEvent>
#include <QFileDialog>
#include <QMessageBox>

#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
#include "MeshDecimation.h"

// Define the mesh type
typedef OpenMesh::TriMesh_ArrayKernelT<> Mesh;

enum RenderMode
{
    Solid,    // Normal rendering
    Wireframe // Wireframe rendering
};

class MeshViewerWidget : public QOpenGLWidget, protected QOpenGLFunctions
{
    Q_OBJECT

public:
    MeshViewerWidget(QWidget *parent = nullptr);
    ~MeshViewerWidget();

    bool loadMesh(const QString &filename);
    void resetView();
    void toggleRenderMode();
    void meshDecimation();
    RenderMode renderMode = Solid;

protected:
    void initializeGL() override;
    void paintGL() override;
    void resizeGL(int width, int height) override;

    void mousePressEvent(QMouseEvent *event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void wheelEvent(QWheelEvent *event) override;

private:
    void updateMeshBuffers();
    Mesh mesh;
    bool meshLoaded;
    MeshDecimation meshDecimator;

    QOpenGLShaderProgram *solidProgram = nullptr;
    QOpenGLShaderProgram *wireframeProgram = nullptr;

    QOpenGLVertexArrayObject vao;
    QOpenGLBuffer vertexBuffer;
    QOpenGLBuffer indexBuffer;
    int indexCount;

    QMatrix4x4 modelMatrix;
    QMatrix4x4 viewMatrix;
    QMatrix4x4 projectionMatrix;

    QPoint lastMousePosition;
    float rotationX, rotationY;
    float zoom;
    float translateX, translateY, translateZ; // 添加平移变量
};

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);

private slots:
    void openFile();
    void meshDecimation();
    void toggleRenderMode();

public slots:
    void loadDefaultModel();

private:
    MeshViewerWidget *meshViewer;
    void createActions();
    void createMenus();
};
