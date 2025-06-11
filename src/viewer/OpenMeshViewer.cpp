#include "OpenMeshViewer.h"
#include "MeshDecimation.h"
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
#include <unordered_map>
#include <Eigen/Dense>
#include <QDebug>
#include <QInputDialog>
#include <QDoubleValidator>
#include <QLabel>
#include <QLineEdit>
#include <QDialogButtonBox>
#include <QFormLayout>
#include <QDialog>

// Hash function for OpenMesh::Vec3f to be used in unordered_map
namespace std
{
    template <>
    struct hash<OpenMesh::Vec3i>
    {
        size_t operator()(const OpenMesh::Vec3i &key) const
        {
            return hash<int>()(key[0]) ^ hash<int>()(key[1]) ^ hash<int>()(key[2]);
        }
    };
}

MeshViewerWidget::MeshViewerWidget(QWidget *parent)
    : QOpenGLWidget(parent),
      meshLoaded(false),
      // program(nullptr),
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
    // program = new QOpenGLShaderProgram();
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
    QOpenGLShaderProgram *activeProgram = (renderMode == Solid) ? solidProgram : wireframeProgram;
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

    QOpenGLShaderProgram *activeProgram = (renderMode == Solid) ? solidProgram : wireframeProgram;
    if (!activeProgram)
        return;

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
    QMenu *processMenu = menuBar()->addMenu("&Process");

    QAction *meshDecimationAction = new QAction("MeshDecimationing", this);
    connect(meshDecimationAction, &QAction::triggered, this, &MainWindow::meshDecimation);
    processMenu->addAction(meshDecimationAction);

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
    QMenu *ToggleMenu = menuBar()->addMenu("&Toggle");
    QAction *toggleRenderModeAction = new QAction("Toggle Render Mode", this);
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
    QFile resourceFile(":/models/Models/Horse.ply");
    if (resourceFile.open(QIODevice::ReadOnly))
    {
        QByteArray data = resourceFile.readAll();

        QString tempFilePath = QStandardPaths::writableLocation(QStandardPaths::TempLocation) + "/Horse_temp.ply";
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
void MainWindow::meshDecimation()
{
    if (!meshViewer || !meshViewer->meshLoaded)
    {
        QMessageBox::warning(this, "警告", "请先加载一个网格模型！");
        return;
    }

    // Create a dialog for setting decimation parameters
    QDialog dialog(this);
    dialog.setWindowTitle("网格简化参数设置");
    dialog.resize(400, 200);

    QFormLayout *layout = new QFormLayout(&dialog);

    // Current mesh info
    int currentVertexCount = meshViewer->mesh.n_vertices();
    int currentFaceCount = meshViewer->mesh.n_faces();

    QLabel *infoLabel = new QLabel(QString("当前网格：%1 个顶点，%2 个面片")
                                       .arg(currentVertexCount)
                                       .arg(currentFaceCount));
    layout->addRow(infoLabel);

    // Target vertex count input
    QLineEdit *targetVertexEdit = new QLineEdit();
    targetVertexEdit->setText(QString::number(currentVertexCount / 2)); // Default to half
    targetVertexEdit->setValidator(new QIntValidator(10, currentVertexCount - 1, this));
    layout->addRow("目标顶点数：", targetVertexEdit);

    // Max error input
    QLineEdit *maxErrorEdit = new QLineEdit();
    maxErrorEdit->setText("0.001");
    maxErrorEdit->setValidator(new QDoubleValidator(0.0, 1.0, 6, this));
    layout->addRow("最大误差阈值：", maxErrorEdit);

    // Buttons
    QDialogButtonBox *buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
    connect(buttonBox, &QDialogButtonBox::accepted, &dialog, &QDialog::accept);
    connect(buttonBox, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);
    layout->addRow(buttonBox);

    if (dialog.exec() == QDialog::Accepted)
    {
        // Get the parameters
        int targetVertexCount = targetVertexEdit->text().toInt();
        double maxError = maxErrorEdit->text().toDouble();

        // Validate parameters
        if (targetVertexCount >= currentVertexCount)
        {
            QMessageBox::warning(this, "参数错误", "目标顶点数必须小于当前顶点数！");
            return;
        }

        // Set parameters and perform decimation
        statusBar()->showMessage("正在执行网格简化...");

        // Configure the decimator
        meshViewer->meshDecimator.setTargetVertexCount(targetVertexCount);
        meshViewer->meshDecimator.setMaxError(maxError);

        // Perform decimation
        meshViewer->meshDecimation();

        // Show results
        int finalVertexCount = meshViewer->mesh.n_vertices();
        int finalFaceCount = meshViewer->mesh.n_faces();

        QString resultMessage = QString("简化完成！\n原始：%1 顶点，%2 面片\n简化后：%3 顶点，%4 面片\n简化率：%.1f%%")
                                    .arg(currentVertexCount)
                                    .arg(currentFaceCount)
                                    .arg(finalVertexCount)
                                    .arg(finalFaceCount)
                                    .arg(100.0 * (currentVertexCount - finalVertexCount) / currentVertexCount);

        QMessageBox::information(this, "简化完成", resultMessage);
        statusBar()->showMessage("网格简化完成", 3000);
    }
}
void MeshViewerWidget::meshDecimation()
{
    if (!meshLoaded)
        return;

    // Use the MeshDecimation class to perform decimation
    meshDecimator.performDecimation(mesh);

    // After decimation, update the mesh buffers
    makeCurrent();
    updateMeshBuffers();
    doneCurrent();

    update();
}

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    MainWindow mainWindow;
    mainWindow.show();
    mainWindow.loadDefaultModel();
    return app.exec();
}