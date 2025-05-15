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

    // Calculate bounding box to determine model size
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
    
    // Store original mesh statistics
    int original_vertices = mesh.n_vertices();
    int original_faces = mesh.n_faces();
    
    // Set epsilon value for vertex clustering (approximately 1% of model size)
    float epsilon = 0.01f * model_size;
    
    // Execute vertex clustering algorithm
    qDebug() << "[Remesh] Starting vertex clustering with epsilon =" << epsilon;
    bool success = vertexClustering(epsilon);
    
    if (success) {
        // Calculate reduction statistics
        int new_vertices = mesh.n_vertices();
        int new_faces = mesh.n_faces();
        float vertex_reduction = 100.0f * (1.0f - static_cast<float>(new_vertices) / original_vertices);
        float face_reduction = 100.0f * (1.0f - static_cast<float>(new_faces) / original_faces);
        
        qDebug() << "[Remesh] Vertex clustering completed:";
        qDebug() << "  - Original: " << original_vertices << " vertices, " << original_faces << " faces";
        qDebug() << "  - New:      " << new_vertices << " vertices, " << new_faces << " faces";
        qDebug() << "  - Reduction: " << vertex_reduction << "% vertices, " << face_reduction << "% faces";
        
        // Update visualization
        makeCurrent();
        updateMeshBuffers();
        doneCurrent();
        update();
    } else {
        qDebug() << "[Remesh] Vertex clustering failed";
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

    // Calculate cell dimensions based on epsilon
    OpenMesh::Vec3f cellSize(epsilon, epsilon, epsilon);
    
    // Map to store clusters: cell index -> quadric error metric (QEM) data
    struct CellData {
        Eigen::Matrix4f Q;           // Quadric error matrix
        OpenMesh::Vec3f position;    // Representative position
        std::vector<Mesh::VertexHandle> vertices; // Vertices in this cell
    };
    
    std::unordered_map<OpenMesh::Vec3i, CellData> clusters;
    
    // Initialize the quadric error matrices (QEM) for each vertex
    std::vector<Eigen::Matrix4f> vertex_quadrics(mesh.n_vertices());
    
    // Step 1: Compute quadric error matrices for each vertex based on its incident faces
    for (auto f_it = mesh.faces_begin(); f_it != mesh.faces_end(); ++f_it) {
        // Get face normal
        OpenMesh::Vec3f normal = mesh.normal(*f_it);
        float a = normal[0];
        float b = normal[1];
        float c = normal[2];
        
        // Get a point on the face to calculate d
        OpenMesh::Vec3f point = mesh.point(*mesh.fv_iter(*f_it));
        float d = -(a * point[0] + b * point[1] + c * point[2]);
        
        // Create plane equation parameters [a, b, c, d]
        Eigen::Vector4f plane(a, b, c, d);
        
        // Create quadric error matrix Q = plane * plane^T
        Eigen::Matrix4f Q = plane * plane.transpose();
        
        // Add this quadric to all vertices of this face
        for (auto fv_it = mesh.fv_iter(*f_it); fv_it.is_valid(); ++fv_it) {
            vertex_quadrics[fv_it->idx()] += Q;
        }
    }

    // Step 2: Assign each vertex to a cluster
    for (auto v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); ++v_it) {
        OpenMesh::Vec3f p = mesh.point(*v_it);
        
        // Calculate cell indices
        OpenMesh::Vec3i cellIdx;
        for (int i = 0; i < 3; ++i) {
            cellIdx[i] = static_cast<int>((p[i] - bb_min[i]) / cellSize[i]);
        }
        
        // Add this vertex to its cell
        clusters[cellIdx].vertices.push_back(*v_it);
        
        // Add this vertex's quadric to the cell
        if (clusters[cellIdx].vertices.size() == 1) {
            // First vertex in this cell
            clusters[cellIdx].Q = vertex_quadrics[v_it->idx()];
        } else {
            // Additional vertex for this cell
            clusters[cellIdx].Q += vertex_quadrics[v_it->idx()];
        }
    }
    
    // Step 3: Compute the optimal representative position for each cluster
    for (auto& cluster_pair : clusters) {
        CellData& cell = cluster_pair.second;
        
        // Extract the 3x3 submatrix A and 3x1 vector b from Q
        Eigen::Matrix3f A;
        Eigen::Vector3f b;
        
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                A(i, j) = cell.Q(i, j);
            }
            b(i) = -cell.Q(i, 3);
        }
        
        // Add small regularization to make A invertible
        A += Eigen::Matrix3f::Identity() * 1e-6f;
        
        // Solve the linear system Ax = b to find the optimal position
        Eigen::Vector3f optimal_pos;
        
        // Check if matrix is invertible
        if (A.determinant() != 0) {
            optimal_pos = A.ldlt().solve(b);
        } else {
            // If not invertible, use the centroid of vertices as fallback
            optimal_pos = Eigen::Vector3f::Zero();
            for (const auto& vh : cell.vertices) {
                OpenMesh::Vec3f p = mesh.point(vh);
                optimal_pos += Eigen::Vector3f(p[0], p[1], p[2]);
            }
            if (!cell.vertices.empty()) {
                optimal_pos /= cell.vertices.size();
            }
        }
        
        // Store the optimal position
        cell.position = OpenMesh::Vec3f(optimal_pos[0], optimal_pos[1], optimal_pos[2]);
    }
    
    // Step 4: Create a new mesh
    Mesh new_mesh;
    
    // Maps original vertex handles to new vertex handles
    std::map<Mesh::VertexHandle, Mesh::VertexHandle> vertex_map;
    
    // Create a new vertex for each cluster
    for (auto& cluster_pair : clusters) {
        const auto& cell = cluster_pair.second;
        
        // Add the representative vertex to the new mesh
        Mesh::VertexHandle new_vh = new_mesh.add_vertex(cell.position);
        
        // Map all original vertices to this new vertex
        for (const auto& vh : cell.vertices) {
            vertex_map[vh] = new_vh;
        }
    }
    
    // Add faces to the new mesh
    for (auto f_it = mesh.faces_begin(); f_it != mesh.faces_end(); ++f_it) {
        std::vector<Mesh::VertexHandle> face_vhs;
        
        // Collect the mapped vertices for this face
        for (auto fv_it = mesh.fv_iter(*f_it); fv_it.is_valid(); ++fv_it) {
            face_vhs.push_back(vertex_map[*fv_it]);
        }
        
        // Check if the face is degenerate (vertices mapped to the same position)
        if (face_vhs[0] != face_vhs[1] && face_vhs[1] != face_vhs[2] && face_vhs[2] != face_vhs[0]) {
            new_mesh.add_face(face_vhs);
        }
    }
    
    // Step 5: Replace the original mesh with the new one
    mesh = new_mesh;
    
    // Compute normals for the new mesh
    mesh.request_face_normals();
    mesh.request_vertex_normals();
    mesh.update_normals();
    
    // Clean up
    mesh.release_vertex_status();
    mesh.release_edge_status();
    mesh.release_face_status();
    
    // Calculate some statistics
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