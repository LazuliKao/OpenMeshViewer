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

/**
 * @brief MeshViewerWidget构造函数
 * @param parent 父窗口指针
 *
 * 初始化OpenGL网格查看器控件，设置默认的视图参数，
 * 包括旋转角度、缩放因子、平移量等初始值
 */
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

/**
 * @brief MeshViewerWidget析构函数
 *
 * 清理OpenGL资源，包括删除着色器程序、销毁VAO和缓冲区对象，
 * 确保在对象销毁时正确释放所有OpenGL相关资源
 */
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

/**
 * @brief 加载网格模型文件
 * @param filename 网格文件路径
 * @return 成功返回true，失败返回false
 *
 * 使用OpenMesh库读取网格文件，支持多种格式（obj, off, stl, ply等），
 * 加载成功后更新顶点和面片法线，刷新OpenGL缓冲区，
 * 并自动调整视图以适应模型大小
 */
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
    autoFitView(); // 自动调整视图以适应模型大小
    update();
    return true;
}

/**
 * @brief 重置视图参数
 *
 * 将所有视图变换参数重置为默认值，包括旋转角度、缩放、平移等，
 * 重置模型矩阵和视图矩阵到初始状态，并刷新显示
 */
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

/**
 * @brief 计算网格模型的边界盒
 * @param minPoint 返回边界盒的最小点坐标
 * @param maxPoint 返回边界盒的最大点坐标
 *
 * 遍历网格的所有顶点，计算包围整个模型的最小边界盒，
 * 用于自动视图调整和模型居中显示
 */
void MeshViewerWidget::computeBoundingBox(Mesh::Point &minPoint, Mesh::Point &maxPoint)
{
    if (!meshLoaded || mesh.n_vertices() == 0)
        return;

    // 初始化边界框
    auto v_it = mesh.vertices_begin();
    minPoint = maxPoint = mesh.point(*v_it);
    ++v_it;

    // 遍历所有顶点计算边界框
    for (; v_it != mesh.vertices_end(); ++v_it)
    {
        Mesh::Point p = mesh.point(*v_it);

        for (int i = 0; i < 3; ++i)
        {
            if (p[i] < minPoint[i])
                minPoint[i] = p[i];
            if (p[i] > maxPoint[i])
                maxPoint[i] = p[i];
        }
    }
}

/**
 * @brief 自动调整视图以适应模型大小
 *
 * 根据模型的边界盒自动计算合适的缩放比例和视图位置，
 * 使模型在视口中完整显示且占据合适的比例，
 * 同时将模型居中显示
 */
void MeshViewerWidget::autoFitView()
{
    if (!meshLoaded)
        return;

    Mesh::Point minPoint, maxPoint;
    computeBoundingBox(minPoint, maxPoint);

    // 计算模型的尺寸
    Mesh::Point size = maxPoint - minPoint;
    float maxSize = std::max({size[0], size[1], size[2]});

    // 如果模型太小或太大，自动调整zoom值
    if (maxSize > 0.0f)
    {
        // 根据模型大小计算合适的zoom值
        // 通常我们希望模型占据视口的合适比例
        float targetSize = 2.0f;            // 目标显示大小
        zoom = maxSize / targetSize * 3.0f; // 调整系数

        // 限制zoom的范围
        zoom = qMax(1.0f, qMin(zoom, 50.0f));
    }

    // 计算模型中心点，用于平移调整
    Mesh::Point center = (minPoint + maxPoint) * 0.5f;

    // 重新设置视图矩阵
    viewMatrix.setToIdentity();
    viewMatrix.translate(-center[0], -center[1], -zoom);
    update();
}

/**
 * @brief 初始化OpenGL上下文
 *
 * OpenGL初始化函数，设置OpenGL状态（深度测试、背景色等），
 * 创建和编译着色器程序（实体渲染和线框渲染），
 * 初始化VAO和缓冲区对象，为网格渲染做准备
 */
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

/**
 * @brief 更新网格缓冲区数据
 *
 * 将网格的顶点数据（位置和法线）和索引数据上传到OpenGL缓冲区，
 * 根据渲染模式（实体或线框）生成不同的索引数据，
 * 配置顶点属性指针以供着色器使用
 */
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

/**
 * @brief OpenGL绘制函数
 *
 * 主要的渲染函数，清除颜色和深度缓冲区，
 * 设置模型-视图-投影矩阵，根据当前渲染模式绘制网格，
 * 支持实体渲染（三角形）和线框渲染（线段）两种模式
 */
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

    // 更新视图矩阵，考虑模型中心和缩放
    QMatrix4x4 tempViewMatrix;
    tempViewMatrix.setToIdentity();
    tempViewMatrix.translate(translateX, translateY, translateZ);
    tempViewMatrix = tempViewMatrix * viewMatrix;
    activeProgram->setUniformValue("model", modelMatrix);
    activeProgram->setUniformValue("view", tempViewMatrix);
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

/**
 * @brief 窗口大小改变时的回调函数
 * @param width 新的窗口宽度
 * @param height 新的窗口高度
 *
 * 当OpenGL窗口大小改变时更新投影矩阵，
 * 保持正确的宽高比和视角参数
 */
void MeshViewerWidget::resizeGL(int width, int height)
{
    // Update projection matrix
    projectionMatrix.setToIdentity();
    projectionMatrix.perspective(45.0f, width / float(height), 0.1f, 100.0f);
}

/**
 * @brief 切换渲染模式
 *
 * 在实体渲染和线框渲染模式之间切换，
 * 切换后重新更新网格缓冲区以适应新的渲染模式，
 * 并刷新显示
 */
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

/**
 * @brief 鼠标按下事件处理
 * @param event 鼠标事件对象
 *
 * 记录鼠标按下时的位置，为后续的鼠标拖拽操作做准备
 */
void MeshViewerWidget::mousePressEvent(QMouseEvent *event)
{
    lastMousePosition = event->pos();
}

/**
 * @brief 鼠标移动事件处理
 * @param event 鼠标事件对象
 *
 * 根据鼠标按键状态执行不同操作：
 * - 左键拖拽：旋转模型（绕X轴和Y轴）
 * - 右键拖拽：平移模型（在XY平面内）
 * - 中键拖拽：Z轴方向平移（前后移动）
 */
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

/**
 * @brief 鼠标滚轮事件处理
 * @param event 滚轮事件对象
 *
 * 响应鼠标滚轮操作，控制视图的缩放，
 * 向前滚动缩小（拉近），向后滚动放大（拉远），
 * 限制缩放范围在合理区间内
 */
void MeshViewerWidget::wheelEvent(QWheelEvent *event)
{
    float zoomFactor = 1.0f - event->angleDelta().y() / 1200.0f;
    zoom *= zoomFactor;
    zoom = qMax(0.1f, qMin(zoom, 100.0f));

    // 更新视图矩阵的Z分量
    QMatrix4x4 currentTranslation;
    currentTranslation.setToIdentity();

    // 提取当前的X,Y平移
    float currentX = viewMatrix(0, 3);
    float currentY = viewMatrix(1, 3);

    viewMatrix.setToIdentity();
    viewMatrix.translate(currentX, currentY, -zoom);
    update();
}

/**
 * @brief MainWindow构造函数
 * @param parent 父窗口指针
 *
 * 创建主窗口界面，初始化网格查看器控件，
 * 设置中央窗口布局，创建菜单栏和工具栏，
 * 配置窗口标题、大小和状态栏
 */
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

    // Set status bar message    statusBar()->showMessage("Ready");
}

/**
 * @brief 创建菜单栏和工具栏动作
 *
 * 创建应用程序的所有菜单项和对应的动作，包括：
 * - 文件菜单：打开文件、退出
 * - 处理菜单：网格简化
 * - 切换菜单：渲染模式切换、自动适应视图
 * 设置快捷键和连接信号槽
 */
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

    QAction *autoFitAction = new QAction("Auto Fit View", this);
    connect(autoFitAction, &QAction::triggered, this, &MainWindow::autoFitView);
    ToggleMenu->addAction(autoFitAction);
}

/**
 * @brief 创建菜单栏
 *
 * 菜单栏的创建已在createActions()函数中完成，
 * 此函数保留用于可能的额外菜单配置
 */
void MainWindow::createMenus()
{ // Menus are already created in createActions()
}

/**
 * @brief 打开文件对话框并加载网格
 *
 * 显示文件选择对话框，支持多种网格文件格式，
 * 用户选择文件后尝试加载网格模型，
 * 显示加载进度和结果信息在状态栏
 */
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

/**
 * @brief 加载默认模型
 *
 * 从应用程序资源中加载默认的恐龙模型(Dino.ply)，
 * 将资源文件复制到临时目录后加载，
 * 用于应用程序启动时的默认显示
 */
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

/**
 * @brief 切换渲染模式
 *
 * 在实体渲染和线框渲染模式之间切换，
 * 调用网格查看器的切换函数并在状态栏显示切换信息
 */
void MainWindow::toggleRenderMode()
{
    // Toggle the render mode between Solid and Wireframe
    if (meshViewer)
    {
        meshViewer->toggleRenderMode();
    }
    statusBar()->showMessage("Render mode toggled", 2000);
}

/**
 * @brief 自动适应视图
 *
 * 自动调整视图以适应当前加载的网格模型，
 * 确保模型完整显示在视口中并占据合适比例，
 * 如果没有加载模型则显示提示信息
 */
void MainWindow::autoFitView()
{
    if (meshViewer && meshViewer->meshLoaded)
    {
        meshViewer->autoFitView();
        statusBar()->showMessage("View auto-fitted to model", 2000);
    }
    else
    {
        QMessageBox::information(this, "Info", "Please load a mesh first!");
    }
}

/**
 * @brief 网格简化功能
 *
 * 显示网格简化参数设置对话框，让用户设置目标顶点数和误差阈值，
 * 执行网格简化算法，显示简化前后的统计信息，
 * 包括顶点数、面片数和简化比例
 */
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

        QString resultMessage = QString("简化完成！\n原始：%1 顶点，%2 面片\n简化后：%3 顶点，%4 面片\n简化率：%5 %")
                                    .arg(currentVertexCount)
                                    .arg(currentFaceCount)
                                    .arg(finalVertexCount)
                                    .arg(finalFaceCount)
                                    .arg(100.0 * (currentVertexCount - finalVertexCount) / currentVertexCount);

        QMessageBox::information(this, "简化完成", resultMessage);
        statusBar()->showMessage("网格简化完成", 3000);
    }
}

/**
 * @brief 执行网格简化
 *
 * MeshViewerWidget中的网格简化执行函数，
 * 调用网格简化器执行简化算法，
 * 简化完成后更新OpenGL缓冲区并刷新显示
 */
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

/**
 * @brief 程序主入口函数
 * @param argc 命令行参数个数
 * @param argv 命令行参数数组
 * @return 程序退出代码
 *
 * 创建Qt应用程序实例，初始化主窗口，
 * 加载默认模型并显示窗口，启动事件循环
 */
int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    MainWindow mainWindow;
    mainWindow.show();
    mainWindow.loadDefaultModel();
    return app.exec();
}