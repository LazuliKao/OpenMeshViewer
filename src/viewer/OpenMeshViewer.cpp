#include "OpenMeshViewer.h"
#include <QApplication>
#include <QMainWindow>
#include <QMenuBar>
#include <QStatusBar>
#include <QVBoxLayout>
#include <QMatrix4x4>
#include <QFile>
#include <QTextStream>
#include <QStandardPaths>
#include <cmath>

MeshViewerWidget::MeshViewerWidget(QWidget *parent)
    : QOpenGLWidget(parent),
      meshLoaded(false),
      //program(nullptr),
      vertexBuffer(QOpenGLBuffer::VertexBuffer),
      indexBuffer(QOpenGLBuffer::IndexBuffer),
      rotationX(0.0f),
      rotationY(0.0f),
      zoom(5.0f),
      translateX(0.0f),
      translateY(0.0f),
      translateZ(0.0f),
      indexCount(0)
{
    setFocusPolicy(Qt::StrongFocus);
}

MeshViewerWidget::~MeshViewerWidget()
{
    makeCurrent();

    if (solidProgram)
    {
        delete solidProgram;
        solidProgram = nullptr;
    }
    if (wireframeProgram)
    {
        delete wireframeProgram;
        wireframeProgram = nullptr;
    }

    vao.destroy();
    vertexBuffer.destroy();
    indexBuffer.destroy();

    doneCurrent();
}

bool MeshViewerWidget::loadMesh(const QString &filename)
{
    if (!OpenMesh::IO::read_mesh(mesh, filename.toStdString()))
    {
        return false;
    }

    // Update vertex normals
    mesh.request_face_normals();
    mesh.request_vertex_normals();
    mesh.update_normals();

    meshLoaded = true;

    // If context is already available, update mesh buffers
    if (context())
    {
        makeCurrent();
        updateMeshBuffers();
        doneCurrent();
    }

    resetView();
    update();

    return true;
}

void MeshViewerWidget::resetView()
{
    rotationX = 0.0f;
    rotationY = 0.0f;
    zoom = 5.0f;
    translateX = 0.0f;
    translateY = 0.0f;
    translateZ = 0.0f;

    modelMatrix.setToIdentity();
    viewMatrix.setToIdentity();
    viewMatrix.translate(0.0f, 0.0f, -zoom);

    update();
}

void MeshViewerWidget::initializeGL()
{
    initializeOpenGLFunctions();
    glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
    glEnable(GL_DEPTH_TEST);

    // Create shader program
    //program = new QOpenGLShaderProgram();
     /*
    program->addShaderFromSourceFile(QOpenGLShader::Vertex, ":/shaders/Shaders/basic.vert.glsl");
    program->addShaderFromSourceFile(QOpenGLShader::Fragment, ":/shaders/Shaders/basic.frag.glsl");
    program->link();*/

    solidProgram = new QOpenGLShaderProgram();
    solidProgram->addShaderFromSourceFile(QOpenGLShader::Vertex, ":/shaders/Shaders/solid.vert.glsl");
    solidProgram->addShaderFromSourceFile(QOpenGLShader::Fragment, ":/shaders/Shaders/solid.frag.glsl");
    solidProgram->link();

    wireframeProgram = new QOpenGLShaderProgram();
    wireframeProgram->addShaderFromSourceFile(QOpenGLShader::Vertex, ":/shaders/Shaders/basic.vert.glsl");
    wireframeProgram->addShaderFromSourceFile(QOpenGLShader::Fragment, ":/shaders/Shaders/basic.frag.glsl");
    wireframeProgram->link();

    // Create VAO and buffers
    vao.create();
    vao.bind();

    vertexBuffer.create();
    indexBuffer.create();

    vao.release();

    // If a mesh is already loaded, update buffers
    if (meshLoaded)
    {
        updateMeshBuffers();
    }
}

void MeshViewerWidget::updateMeshBuffers()
{
    if (!meshLoaded)
        return;

    vao.bind();

    // 准备顶点数据（位置和法线）
    QVector<GLfloat> vertices;
    for (Mesh::VertexIter v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); ++v_it)
    {
        Mesh::Point p = mesh.point(*v_it);
        Mesh::Normal n = mesh.normal(*v_it);

        // 顶点位置
        vertices << p[0] << p[1] << p[2];
        // 顶点法线（如果你有需要可以使用）
        vertices << n[0] << n[1] << n[2];
    }

    // 准备索引数据，假设每个面是三角形
    QVector<GLuint> indices;
    indices.clear();
    for (Mesh::FaceIter f_it = mesh.faces_begin(); f_it != mesh.faces_end(); ++f_it)
    {
        QList<GLuint> faceIndices;
        for (Mesh::FaceVertexIter fv_it = mesh.fv_iter(*f_it); fv_it.is_valid(); ++fv_it)
        {
            faceIndices.append(fv_it->idx());
        }

        if (faceIndices.size() == 3)
        {
            if (renderMode == Wireframe)
            {
                // 三条边作为线段
                indices << faceIndices[0] << faceIndices[1];
                indices << faceIndices[1] << faceIndices[2];
                indices << faceIndices[2] << faceIndices[0];
            }
            else
            {
                // 三角面片
                indices << faceIndices[0] << faceIndices[1] << faceIndices[2];
            }
        }
    }

    // 存储索引数量
    indexCount = indices.size();

    // 上传顶点数据
    vertexBuffer.bind();
    vertexBuffer.allocate(vertices.constData(), vertices.size() * sizeof(GLfloat));

    // 设置顶点属性指针
    /*program->bind();*/
    QOpenGLShaderProgram* activeProgram = (renderMode == Solid) ? solidProgram : wireframeProgram;
    activeProgram->bind();

    // 位置属性
    int posAttr = activeProgram->attributeLocation("position");
    activeProgram->enableAttributeArray(posAttr);
    activeProgram->setAttributeBuffer(posAttr, GL_FLOAT, 0, 3, 6 * sizeof(GLfloat));

    // 法线属性
    int normalAttr = activeProgram->attributeLocation("normal");
    activeProgram->enableAttributeArray(normalAttr);
    activeProgram->setAttributeBuffer(normalAttr, GL_FLOAT, 3 * sizeof(GLfloat), 3, 6 * sizeof(GLfloat));

    // 上传索引数据
    indexBuffer.bind();
    indexBuffer.allocate(indices.constData(), indices.size() * sizeof(GLuint));

    activeProgram->release();
    vao.release();
}

void MeshViewerWidget::paintGL()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (!meshLoaded)
        return;

    QOpenGLShaderProgram* activeProgram = (renderMode == Solid) ? solidProgram : wireframeProgram;
    if (!activeProgram) return;

    activeProgram->bind();
    vao.bind();

    modelMatrix.setToIdentity();
    modelMatrix.rotate(rotationX, 1.0f, 0.0f, 0.0f);
    modelMatrix.rotate(rotationY, 0.0f, 1.0f, 0.0f);

    viewMatrix.setToIdentity();
    viewMatrix.translate(translateX, translateY, -zoom + translateZ);

    activeProgram->setUniformValue("model", modelMatrix);
    activeProgram->setUniformValue("view", viewMatrix);
    activeProgram->setUniformValue("projection", projectionMatrix);

    if (renderMode == Solid)
    {
        glDrawElements(GL_TRIANGLES, indexCount, GL_UNSIGNED_INT, 0);
    }
    else if (renderMode == Wireframe)
    {
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        glDrawElements(GL_LINES, indexCount, GL_UNSIGNED_INT, 0);
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }

    vao.release();
    activeProgram->release();
}

void MeshViewerWidget::resizeGL(int width, int height)
{
    // Update projection matrix
    projectionMatrix.setToIdentity();
    projectionMatrix.perspective(45.0f, width / float(height), 0.1f, 100.0f);
}

void MeshViewerWidget::toggleRenderMode()
{
    if (renderMode == Solid)
    {
        renderMode = Wireframe;
    }
    else
    {
        renderMode = Solid;
    }

    makeCurrent();       
    updateMeshBuffers();  
    doneCurrent();

    update(); 
}

void MeshViewerWidget::mousePressEvent(QMouseEvent *event)
{
    lastMousePosition = event->pos();
}

void MeshViewerWidget::mouseMoveEvent(QMouseEvent *event)
{
    QPoint delta = event->pos() - lastMousePosition;
    
    if (event->buttons() & Qt::LeftButton)
    {
        // 旋转
        rotationY += 0.5f * delta.x();
        rotationX += 0.5f * delta.y();
        update();
    }
    else if (event->buttons() & Qt::RightButton)
    {
        // 平移
        float sensitivity = 0.01f;
        translateX += sensitivity * delta.x();
        translateY -= sensitivity * delta.y(); // 注意Y轴方向与屏幕坐标系相反
        update();
    }
    else if (event->buttons() & Qt::MiddleButton)
    {
        // Z轴平移
        float sensitivity = 0.01f;
        translateZ += sensitivity * delta.y();
        update();
    }

    lastMousePosition = event->pos();
}

void MeshViewerWidget::wheelEvent(QWheelEvent *event)
{
    zoom -= event->angleDelta().y() / 120.0f;
    zoom = qMax(1.0f, qMin(zoom, 15.0f));

    viewMatrix.setToIdentity();
    viewMatrix.translate(0.0f, 0.0f, -zoom);

    update();
}

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
{
    // Create central widget with mesh viewer
    QWidget *centralWidget = new QWidget();
    QVBoxLayout *layout = new QVBoxLayout(centralWidget);

    meshViewer = new MeshViewerWidget();
    layout->addWidget(meshViewer);

    setCentralWidget(centralWidget);

    // Create menus and actions
    createActions();
    createMenus();

    // Set window properties
    setWindowTitle("OpenMesh Viewer");
    resize(800, 600);

    // Set status bar message
    statusBar()->showMessage("Ready");
}

void MainWindow::createActions()
{
    QMenu* processMenu = menuBar()->addMenu("&Process");

    QAction* remeshAction = new QAction("Remeshing", this);
    connect(remeshAction, &QAction::triggered, this, &MainWindow::remeshMesh);
    processMenu->addAction(remeshAction);

    QMenu *fileMenu = menuBar()->addMenu("&File");

    QAction *openAction = new QAction("&Open", this);
    openAction->setShortcut(QKeySequence::Open);
    connect(openAction, &QAction::triggered, this, &MainWindow::openFile);
    fileMenu->addAction(openAction);

    fileMenu->addSeparator();

    QAction *exitAction = new QAction("E&xit", this);
    exitAction->setShortcut(QKeySequence::Quit);
    connect(exitAction, &QAction::triggered, this, &QWidget::close);
    fileMenu->addAction(exitAction);

    // Add the render mode toggle action
    QMenu* ToggleMenu = menuBar()->addMenu("&Toggle");
    QAction* toggleRenderModeAction = new QAction("Toggle Render Mode", this);
    connect(toggleRenderModeAction, &QAction::triggered, this, &MainWindow::toggleRenderMode);
    ToggleMenu->addAction(toggleRenderModeAction);
}

void MainWindow::createMenus()
{
    // Menus are already created in createActions()
}

void MainWindow::openFile()
{
    QString filename = QFileDialog::getOpenFileName(this,
                                                    "Open Mesh File", "", "Mesh Files (*.obj *.off *.stl *.ply);;All Files (*)");

    if (filename.isEmpty())
        return;

    statusBar()->showMessage("Loading mesh...");

    if (meshViewer->loadMesh(filename))
    {
        statusBar()->showMessage("Mesh loaded successfully", 3000);
    }
    else
    {
        QMessageBox::critical(this, "Error", "Failed to load mesh file");
        statusBar()->showMessage("Failed to load mesh", 3000);
    }
}

void MainWindow::loadDefaultModel()
{
    QFile resourceFile(":/models/Models/Dino.ply");
    if (resourceFile.open(QIODevice::ReadOnly))
    {
        QByteArray data = resourceFile.readAll();

        QString tempFilePath = QStandardPaths::writableLocation(QStandardPaths::TempLocation) + "/Dino_temp.ply";
        QFile tempFile(tempFilePath);
        if (tempFile.open(QIODevice::WriteOnly))
        {
            tempFile.write(data);
            tempFile.close();
            meshViewer->loadMesh(tempFilePath);
			tempFile.remove();
		}
		else
		{
			QMessageBox::critical(this, "Error", "Failed to create temporary file");
        }
    }
    
}

void MainWindow::toggleRenderMode()
{
    // Toggle the render mode between Solid and Wireframe
    if (meshViewer)
    {
        meshViewer->toggleRenderMode();
    }

    statusBar()->showMessage("Render mode toggled", 2000);
}
void MainWindow::remeshMesh()
{
    if (meshViewer) {
        meshViewer->remesh();
        statusBar()->showMessage("Remeshing completed", 2000);
    }
}
void MeshViewerWidget::remesh()
{
    if (!meshLoaded) return;

    mesh.request_face_normals();
    mesh.request_vertex_normals();

    mesh.request_edge_status();
    mesh.request_face_status();
    mesh.request_vertex_status();

    // 手动计算 bounding box 对角线长度（作为模型尺度参考）
    OpenMesh::Vec3f bb_min(FLT_MAX, FLT_MAX, FLT_MAX);
    OpenMesh::Vec3f bb_max(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    for (auto v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); ++v_it)
    {
        auto p = mesh.point(*v_it);
        for (int i = 0; i < 3; ++i) {
            bb_min[i] = std::min(bb_min[i], p[i]);
            bb_max[i] = std::max(bb_max[i], p[i]);
        }
    }
    float model_size = (bb_max - bb_min).norm();
    float targetLength = 0.05f * model_size; // 目标边长 = 模型尺度的 5%

    int tooLong = 0, tooShort = 0;
    float totalLen = 0.0f;
    int edgeCount = 0;

    for (auto e_it = mesh.edges_begin(); e_it != mesh.edges_end(); ++e_it)
    {
        float len = mesh.calc_edge_length(*e_it);
        totalLen += len;
        ++edgeCount;

        if (len > 4.0f / 3.0f * targetLength)
            ++tooLong;
        else if (len < 4.0f / 5.0f * targetLength)
            ++tooShort;
    }

    float avgLen = totalLen / edgeCount;

    qDebug() << "[Remesh Info] Edges:" << edgeCount
        << " Avg Length:" << avgLen
        << " Target:" << targetLength
        << " Too Long:" << tooLong
        << " Too Short:" << tooShort;

    // 重计算法线（你可加 smoothing 后再重建法线）
    mesh.update_normals();

    // 清理状态位
    mesh.release_edge_status();
    mesh.release_face_status();
    mesh.release_vertex_status();

    updateMeshBuffers();  // 刷新OpenGL缓存
    update();             // 重绘窗口
}


int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    MainWindow mainWindow;
    mainWindow.show();
	mainWindow.loadDefaultModel();
    return app.exec();
}