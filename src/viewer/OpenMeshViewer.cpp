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
namespace std {
    template<>
    struct hash<OpenMesh::Vec3i> {
        size_t operator()(const OpenMesh::Vec3i& key) const {
            return hash<int>()(key[0]) ^ hash<int>()(key[1]) ^ hash<int>()(key[2]);
        }
    };
}

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

    int original_vertices = mesh.n_vertices();
    int target_min_vertices = static_cast<int>(original_vertices * 0.9);  // 至少保留 90%点

    float epsilon = 0.01f * model_size;
    float max_epsilon = 0.1f * model_size; // 防止无限放大
    float step = 0.005f * model_size;

    Mesh backup = mesh; // 备份原始网格

    bool success = false;

    while (epsilon <= max_epsilon)
    {
        mesh = backup; // 每次尝试新的 epsilon 前还原
        qDebug() << "[Remesh] Trying epsilon =" << epsilon;

        if (vertexClustering(epsilon))
        {
            int new_vertices = mesh.n_vertices();

            if (new_vertices >= target_min_vertices)
            {
                success = true;
                qDebug() << "[Remesh] Success with vertex count:" << new_vertices;
                break;
            }
            else
            {
                qDebug() << "[Remesh] Too simplified (" << new_vertices << " vertices)";
            }
        }
        else
        {
            qDebug() << "[Remesh] vertexClustering failed for epsilon =" << epsilon;
        }

        epsilon += step;
    }

    if (success)
    {
        mesh.request_face_normals();
        mesh.request_vertex_normals();
        mesh.update_normals();
        updateMeshBuffers();
        update();

        int final_vertices = mesh.n_vertices();
        int final_faces = mesh.n_faces();
        float vertex_reduction = 100.0f * (1.0f - static_cast<float>(final_vertices) / original_vertices);

        qDebug() << "[Remesh] Final vertex count:" << final_vertices
            << "(" << 100 - vertex_reduction << "% retained)";
    }
    else
    {
        QMessageBox::warning(this, "Remesh", "Failed to simplify while retaining 90% vertices");
        mesh = backup;
        updateMeshBuffers();
        update();
    }
}


bool MeshViewerWidget::vertexClustering(float epsilon)
{
    if (!meshLoaded || epsilon <= 0.0f)
        return false;

    // Request necessary mesh properties
    mesh.request_vertex_status();
    mesh.request_edge_status();
    mesh.request_face_status();

    // Calculate bounding box
    OpenMesh::Vec3f bb_min(FLT_MAX, FLT_MAX, FLT_MAX);
    OpenMesh::Vec3f bb_max(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    for (auto v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); ++v_it)
    {
        OpenMesh::Vec3f p = mesh.point(*v_it);
        for (int i = 0; i < 3; ++i) {
            bb_min[i] = std::min(bb_min[i], p[i]);
            bb_max[i] = std::max(bb_max[i], p[i]);
        }
    }

    // 计算边界盒对角线长度，并用于检查异常值
    float diag_length = (bb_max - bb_min).norm();
    float bound_check = diag_length * 0.5f; // 用于检测异常顶点的阈值
    
    // Calculate cell dimensions based on epsilon
    OpenMesh::Vec3f cellSize(epsilon, epsilon, epsilon);
    
    // Map to store clusters: cell index -> quadric error metric (QEM) data
    struct CellData {
        Eigen::Matrix4f Q;           // Quadric error matrix
        OpenMesh::Vec3f position;    // Representative position
        std::vector<Mesh::VertexHandle> vertices; // Vertices in this cell
        OpenMesh::Vec3f centroid;    // 计算质心作为备选位置
        bool use_centroid;           // 是否使用质心代替QEM位置
    };
    
    std::unordered_map<OpenMesh::Vec3i, CellData> clusters;
    
    // Initialize the quadric error matrices (QEM) for each vertex
    std::vector<Eigen::Matrix4f> vertex_quadrics(mesh.n_vertices(), Eigen::Matrix4f::Zero());
    
    // Step 1: 计算每个顶点的二次误差矩阵
    for (auto f_it = mesh.faces_begin(); f_it != mesh.faces_end(); ++f_it) {
        // 获取面的法向量
        OpenMesh::Vec3f normal = mesh.normal(*f_it);
        float a = normal[0];
        float b = normal[1];
        float c = normal[2];
        
        // 获取面上的一个点以计算d
        OpenMesh::Vec3f point = mesh.point(*mesh.fv_iter(*f_it));
        float d = -(a * point[0] + b * point[1] + c * point[2]);
        
        // 创建平面方程参数 [a, b, c, d]
        Eigen::Vector4f plane(a, b, c, d);
        
        // 创建二次误差矩阵 Q = plane * plane^T
        Eigen::Matrix4f Q = plane * plane.transpose();
        
        // 将此二次误差矩阵添加到此面的所有顶点
        for (auto fv_it = mesh.fv_iter(*f_it); fv_it.is_valid(); ++fv_it) {
            vertex_quadrics[fv_it->idx()] += Q;
        }
    }

    // Step 2: 将每个顶点分配到一个单元格
    for (auto v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); ++v_it) {
        OpenMesh::Vec3f p = mesh.point(*v_it);
        
        // 计算单元格索引
        OpenMesh::Vec3i cellIdx;
        for (int i = 0; i < 3; ++i) {
            cellIdx[i] = static_cast<int>((p[i] - bb_min[i]) / cellSize[i]);
        }
        
        // 将顶点添加到其单元格
        clusters[cellIdx].vertices.push_back(*v_it);
        
        // 更新单元格的质心信息
        if (clusters[cellIdx].vertices.size() == 1) {
            // 初始化质心为第一个顶点
            clusters[cellIdx].centroid = p;
            clusters[cellIdx].Q = vertex_quadrics[v_it->idx()];
            clusters[cellIdx].use_centroid = false;
        } else {
            // 累积顶点位置以计算质心
            const auto& oldCentroid = clusters[cellIdx].centroid;
            const size_t n = clusters[cellIdx].vertices.size();
            clusters[cellIdx].centroid = oldCentroid * ((n-1.0f)/n) + p * (1.0f/n);
            
            // 累积二次误差矩阵
            clusters[cellIdx].Q += vertex_quadrics[v_it->idx()];
        }
    }
    
    // Step 3: 为每个单元格计算最优代表顶点位置
    for (auto& cluster_pair : clusters) {
        CellData& cell = cluster_pair.second;
        
        // 提取子矩阵 A 和向量 b
        Eigen::Matrix3f A;
        Eigen::Vector3f b;
        
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                A(i, j) = cell.Q(i, j);
            }
            b(i) = -cell.Q(i, 3);
        }
        
        // 添加正则化项以确保矩阵可逆
        float reg_factor = 1e-6f * A.norm();
        A += Eigen::Matrix3f::Identity() * reg_factor;
        
        // 求解线性系统 Ax = b 以找到最优位置
        Eigen::Vector3f optimal_pos;
        
        // 检查矩阵是否可逆和条件数（数值稳定性）
        bool matrix_ok = false;
        if (A.determinant() != 0) {
            // 使用更稳定的分解方法
            Eigen::LDLT<Eigen::Matrix3f> ldlt(A);
            if (ldlt.isPositiveDefinite()) {
                optimal_pos = ldlt.solve(b);
                matrix_ok = ldlt.info() == Eigen::Success;
            }
        }
        
        if (!matrix_ok) {
            // 如果矩阵不可逆或不是正定的，使用最小二乘求解
            Eigen::JacobiSVD<Eigen::Matrix3f> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
            if (svd.rank() > 0) {
                optimal_pos = svd.solve(b);
                matrix_ok = true;
            }
        }
        
        // 如果矩阵求解成功，检查生成的位置是否合理（不超出模型边界太多）
        if (matrix_ok) {
            OpenMesh::Vec3f candidate(optimal_pos[0], optimal_pos[1], optimal_pos[2]);
            
            // 检查计算出的点是否偏离太远（异常值检测）
            bool is_outlier = false;
            
            // 检查点是否偏离边界盒太远
            for (int i = 0; i < 3; ++i) {
                if (candidate[i] < bb_min[i] - bound_check || candidate[i] > bb_max[i] + bound_check) {
                    is_outlier = true;
                    break;
                }
            }
            
            // 检查点到单元格质心的距离是否过大
            if (!is_outlier) {
                float dist_to_centroid = (candidate - cell.centroid).norm();
                if (dist_to_centroid > diag_length * 0.1f) {  // 如果距离质心太远
                    is_outlier = true;
                }
            }
            
            if (!is_outlier) {
                cell.position = candidate;
                cell.use_centroid = false;
            } else {
                // 使用质心作为备选方案
                cell.position = cell.centroid;
                cell.use_centroid = true;
            }
        } else {
            // 如果矩阵求解失败，使用质心
            cell.position = cell.centroid;
            cell.use_centroid = true;
        }
    }
    
    // 日志输出
    int centroid_used = 0;
    for (const auto& cluster_pair : clusters) {
        if (cluster_pair.second.use_centroid) {
            centroid_used++;
        }
    }
    qDebug() << "[Vertex Clustering] Using centroid for" << centroid_used << "out of" << clusters.size() << "clusters";
    
    // Step 4: 创建新网格
    Mesh new_mesh;
    
    // 映射：原始顶点句柄 -> 新顶点句柄
    std::map<Mesh::VertexHandle, Mesh::VertexHandle> vertex_map;
    
    // 为每个单元格创建一个新顶点
    for (auto& cluster_pair : clusters) {
        const auto& cell = cluster_pair.second;
        
        // 将代表顶点添加到新网格
        Mesh::VertexHandle new_vh = new_mesh.add_vertex(cell.position);
        
        // 将所有原始顶点映射到此新顶点
        for (const auto& vh : cell.vertices) {
            vertex_map[vh] = new_vh;
        }
    }
    
    // 将面添加到新网格
    for (auto f_it = mesh.faces_begin(); f_it != mesh.faces_end(); ++f_it) {
        std::vector<Mesh::VertexHandle> face_vhs;
        
        // 收集此面映射后的顶点
        for (auto fv_it = mesh.fv_iter(*f_it); fv_it.is_valid(); ++fv_it) {
            face_vhs.push_back(vertex_map[*fv_it]);
        }
        
        // 检查面是否退化（顶点映射到相同位置）
        if (face_vhs.size() == 3 && face_vhs[0] != face_vhs[1] && face_vhs[1] != face_vhs[2] && face_vhs[2] != face_vhs[0]) {
            new_mesh.add_face(face_vhs);
        }
    }
    
    // Step 5: 替换原始网格为新网格
    mesh = new_mesh;
    
    // 为新网格计算法线
    mesh.request_face_normals();
    mesh.request_vertex_normals();
    mesh.update_normals();
    
    // 清理
    mesh.release_vertex_status();
    mesh.release_edge_status();
    mesh.release_face_status();
    
    // 计算一些统计数据
    qDebug() << "[Vertex Clustering] Original vertices:" << vertex_map.size() 
             << " Clusters:" << clusters.size()
             << " New faces:" << mesh.n_faces();
    
    return true;
}

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    MainWindow mainWindow;
    mainWindow.show();
	mainWindow.loadDefaultModel();
    return app.exec();
}