#include "MeshDecimation.h"
#include <iostream>
#include <vector>
#include <algorithm>

MeshDecimation::MeshDecimation()
    : targetVertexCount_(100) // 默认目标顶点数量
      ,
      maxAllowedError_(1e-3) // 默认最大允许误差
      ,
      isInitialized_(false) // 初始化状态为false
{
    // 初始化完成
    std::cout << "MeshDecimation 实例创建，默认参数：目标顶点数="
              << targetVertexCount_ << "，最大误差=" << maxAllowedError_ << std::endl;
}

MeshDecimation::~MeshDecimation()
{
    // 清理数据结构
    vertexQMatrices_.clear();
    edgeInfoMap_.clear();
    while (!edgeQueue_.empty())
    {
        edgeQueue_.pop();
    }
}

void MeshDecimation::performDecimation(Mesh &mesh)
{
    std::cout << "开始网格简化，当前顶点数: " << mesh.n_vertices() 
              << "，目标顶点数: " << targetVertexCount_ << std::endl;

    // 检查是否需要简化
    if (mesh.n_vertices() <= targetVertexCount_) {
        std::cout << "当前顶点数已满足目标，无需简化" << std::endl;
        return;
    }

    // QEM算法主要步骤：
    
    // 1. 初始化：计算所有顶点的Q矩阵
    initializeQMatrices(mesh);
    
    // 2. 计算所有边的最优收缩目标顶点和成本
    computeOptimalContractionTargets(mesh);
    
    // 3. 构建基于成本的优先队列
    buildPriorityQueue(mesh);
    
    // 4. 主循环：执行边收缩直到满足停止条件
    performEdgeCollapses(mesh);
    
    std::cout << "网格简化完成，最终顶点数: " << mesh.n_vertices() << std::endl;
    isInitialized_ = true;
}

void MeshDecimation::setTargetVertexCount(int targetCount)
{
    if (targetCount > 0) {
        targetVertexCount_ = targetCount;
        std::cout << "目标顶点数设置为: " << targetVertexCount_ << std::endl;
    } else {
        std::cerr << "错误：目标顶点数必须为正数" << std::endl;
    }
}

void MeshDecimation::setMaxError(double maxError)
{
    if (maxError > 0.0) {
        maxAllowedError_ = maxError;
        std::cout << "最大允许误差设置为: " << maxAllowedError_ << std::endl;
    } else {
        std::cerr << "错误：最大允许误差必须为正数" << std::endl;
    }
}

void MeshDecimation::initializeQMatrices(Mesh &mesh)
{
    std::cout << "初始化Q矩阵..." << std::endl;
    
    // 清空之前的数据
    vertexQMatrices_.clear();
    
    // 为每个顶点计算Q矩阵
    for (auto vh : mesh.vertices()) {
        vertexQMatrices_[vh] = computeVertexQMatrix(mesh, vh);
    }
    
    std::cout << "Q矩阵初始化完成，处理顶点数: " << vertexQMatrices_.size() << std::endl;
}

void MeshDecimation::computeOptimalContractionTargets(Mesh &mesh)
{
    std::cout << "计算边收缩目标..." << std::endl;
    
    // 清空之前的边信息
    edgeInfoMap_.clear();
    
    // 为每条边计算收缩成本和目标位置
    for (auto eh : mesh.edges()) {
        computeEdgeCostAndTarget(mesh, eh);
    }
    
    std::cout << "边收缩目标计算完成，处理边数: " << edgeInfoMap_.size() << std::endl;
}

void MeshDecimation::buildPriorityQueue(Mesh &mesh)
{
    std::cout << "构建优先队列..." << std::endl;
    
    // 清空优先队列
    while (!edgeQueue_.empty()) {
        edgeQueue_.pop();
    }
    
    // 将所有边信息加入优先队列
    for (const auto& pair : edgeInfoMap_) {
        edgeQueue_.push(pair.second);
    }
    
    std::cout << "优先队列构建完成，边数: " << edgeQueue_.size() << std::endl;
}

void MeshDecimation::performEdgeCollapses(Mesh &mesh)
{
    std::cout << "开始边收缩过程..." << std::endl;
    
    int collapseCount = 0;
    int totalVertices = mesh.n_vertices();
    
    // 主循环：当顶点数大于目标数且队列非空时继续
    while (mesh.n_vertices() > targetVertexCount_ && !edgeQueue_.empty()) {
        
        // 获取成本最小的边
        EdgeInfo minEdgeInfo = edgeQueue_.top();
        edgeQueue_.pop();
        
        // 检查边是否仍然有效（可能已被之前的操作删除）
        if (edgeInfoMap_.find(minEdgeInfo.edge) == edgeInfoMap_.end()) {
            continue;
        }
        
        // 检查误差是否超过阈值
        if (minEdgeInfo.cost > maxAllowedError_) {
            std::cout << "达到最大允许误差阈值，停止简化" << std::endl;
            break;
        }
        
        // 执行边收缩
        if (collapseEdge(mesh, minEdgeInfo.edge, minEdgeInfo.targetVertex)) {
            collapseCount++;
            
            // 每收缩100条边报告一次进度
            if (collapseCount % 100 == 0) {
                std::cout << "已收缩 " << collapseCount << " 条边，当前顶点数: " 
                         << mesh.n_vertices() << std::endl;
            }
        }
    }
    
    // 清理删除的几何元素
    mesh.garbage_collection();
    
    std::cout << "边收缩完成，总共收缩了 " << collapseCount << " 条边" << std::endl;
    std::cout << "简化前顶点数: " << totalVertices << "，简化后顶点数: " << mesh.n_vertices() << std::endl;
}

Eigen::Matrix4d MeshDecimation::computeVertexQMatrix(Mesh &mesh, Mesh::VertexHandle vertex)
{
    Eigen::Matrix4d Q = Eigen::Matrix4d::Zero();
    
    // 获取顶点位置
    auto point = mesh.point(vertex);
    Eigen::Vector3d vertexPos(point[0], point[1], point[2]);
    
    // 遍历顶点周围的所有面，累加基础二次型
    for (auto vf_it = mesh.vf_iter(vertex); vf_it.is_valid(); ++vf_it) {
        auto fh = *vf_it;
        
        // 计算面的法向量
        auto normal = mesh.calc_face_normal(fh);
        Eigen::Vector3d faceNormal(normal[0], normal[1], normal[2]);
        faceNormal.normalize();
        
        // 计算该面的基础二次型并累加到Q矩阵
        Q += computeFundamentalQuadric(faceNormal, vertexPos);
    }
    
    return Q;
}

void MeshDecimation::computeEdgeCostAndTarget(Mesh &mesh, Mesh::EdgeHandle edge)
{
    // 获取边的两个顶点
    auto heh = mesh.halfedge_handle(edge, 0);
    auto v1 = mesh.from_vertex_handle(heh);
    auto v2 = mesh.to_vertex_handle(heh);
    
    // 获取顶点位置
    auto p1 = mesh.point(v1);
    auto p2 = mesh.point(v2);
    Eigen::Vector3d pos1(p1[0], p1[1], p1[2]);
    Eigen::Vector3d pos2(p2[0], p2[1], p2[2]);
    
    // 获取两个顶点的Q矩阵并相加
    Eigen::Matrix4d Q1 = vertexQMatrices_[v1];
    Eigen::Matrix4d Q2 = vertexQMatrices_[v2];
    Eigen::Matrix4d Qsum = addQMatrices(Q1, Q2);
    
    // 计算最优收缩位置
    Eigen::Vector3d targetPos = solveOptimalPosition(Qsum, pos1, pos2);
    
    // 计算该位置的二次误差作为边的成本
    double cost = computeQuadricError(Qsum, targetPos);
    
    // 存储边信息
    EdgeInfo info;
    info.edge = edge;
    info.cost = cost;
    info.targetVertex = targetPos;
    
    edgeInfoMap_[edge] = info;
}

bool MeshDecimation::collapseEdge(Mesh &mesh, Mesh::EdgeHandle edge, const Eigen::Vector3d &targetPos)
{
    // 检查边是否可以被收缩
    if (!mesh.is_collapse_ok(mesh.halfedge_handle(edge, 0))) {
        return false;
    }
    
    // 获取边的两个顶点
    auto heh = mesh.halfedge_handle(edge, 0);
    auto v1 = mesh.from_vertex_handle(heh);
    auto v2 = mesh.to_vertex_handle(heh);
    
    // 将目标位置设置给保留的顶点（v2）
    Mesh::Point newPoint(targetPos.x(), targetPos.y(), targetPos.z());
    mesh.set_point(v2, newPoint);
    
    // 更新保留顶点的Q矩阵（两个顶点Q矩阵的和）
    Eigen::Matrix4d Q1 = vertexQMatrices_[v1];
    Eigen::Matrix4d Q2 = vertexQMatrices_[v2];
    vertexQMatrices_[v2] = addQMatrices(Q1, Q2);
    
    // 收集将被影响的顶点（用于后续更新边成本）
    std::vector<Mesh::VertexHandle> affectedVertices;
    for (auto vv_it = mesh.vv_iter(v1); vv_it.is_valid(); ++vv_it) {
        affectedVertices.push_back(*vv_it);
    }
    for (auto vv_it = mesh.vv_iter(v2); vv_it.is_valid(); ++vv_it) {
        affectedVertices.push_back(*vv_it);
    }
    
    // 移除与即将删除顶点相关的边信息
    for (auto ve_it = mesh.ve_iter(v1); ve_it.is_valid(); ++ve_it) {
        edgeInfoMap_.erase(*ve_it);
    }
    
    // 执行边收缩操作
    mesh.collapse(heh);
    
    // 移除已删除顶点的Q矩阵
    vertexQMatrices_.erase(v1);
    
    // 更新受影响边的成本
    updateAffectedEdgeCosts(mesh, v2);
    
    return true;
}

void MeshDecimation::updateAffectedEdgeCosts(Mesh &mesh, Mesh::VertexHandle remainingVertex)
{
    // 更新与保留顶点相邻的所有边的成本
    for (auto ve_it = mesh.ve_iter(remainingVertex); ve_it.is_valid(); ++ve_it) {
        auto edgeHandle = *ve_it;
        
        // 重新计算该边的成本和目标位置
        computeEdgeCostAndTarget(mesh, edgeHandle);
        
        // 将更新后的边信息加入优先队列
        if (edgeInfoMap_.find(edgeHandle) != edgeInfoMap_.end()) {
            edgeQueue_.push(edgeInfoMap_[edgeHandle]);
        }
    }
}

Eigen::Matrix4d MeshDecimation::computeFundamentalQuadric(const Eigen::Vector3d &normal, const Eigen::Vector3d &point)
{
    // 计算平面方程的系数：ax + by + cz + d = 0
    // 其中 (a,b,c) = normal, d = -normal·point
    double a = normal.x();
    double b = normal.y(); 
    double c = normal.z();
    double d = -normal.dot(point);
    
    // 构建基础二次型矩阵 K_p = pp^T，其中 p = [a, b, c, d]^T
    Eigen::Vector4d p(a, b, c, d);
    Eigen::Matrix4d K = p * p.transpose();
    
    return K;
}

Eigen::Matrix4d MeshDecimation::addQMatrices(const Eigen::Matrix4d &q1, const Eigen::Matrix4d &q2)
{
    // Q矩阵相加：简单的矩阵加法操作
    return q1 + q2;
}

Eigen::Vector3d MeshDecimation::solveOptimalPosition(const Eigen::Matrix4d &qMatrix,
                                                     const Eigen::Vector3d &v1,
                                                     const Eigen::Vector3d &v2)
{
    // 1. 尝试求解线性系统 Q'v = -q (其中Q'是Q的3x3子矩阵, q是Q的前3行第4列)

    // 提取3x3子矩阵和右侧向量
    Eigen::Matrix3d Q3x3 = qMatrix.block<3, 3>(0, 0);
    Eigen::Vector3d q3x1 = qMatrix.block<3, 1>(0, 3);

    // 检查矩阵是否可逆（使用更稳定的判断方法）
    Eigen::FullPivLU<Eigen::Matrix3d> lu(Q3x3);
    if (lu.isInvertible())
    {
        // 系统可解，返回最优解
        return -Q3x3.inverse() * q3x1;
    }
    else
    {
        // 2. 系统不可解，在三个候选位置中选择误差最小的：v1, v2, (v1+v2)/2
        Eigen::Vector3d midPoint = (v1 + v2) * 0.5;

        double error1 = computeQuadricError(qMatrix, v1);
        double error2 = computeQuadricError(qMatrix, v2);
        double errorMid = computeQuadricError(qMatrix, midPoint);

        // 返回误差最小的位置
        if (error1 <= error2 && error1 <= errorMid)
            return v1;
        else if (error2 <= errorMid)
            return v2;
        else
            return midPoint;
    }
}

double MeshDecimation::computeQuadricError(const Eigen::Matrix4d &qMatrix, const Eigen::Vector3d &position)
{
    // 计算二次误差：v^T * Q * v，其中v = [x, y, z, 1]^T
    // 这个值表示将顶点放置在给定位置时的几何误差

    Eigen::Vector4d v(position.x(), position.y(), position.z(), 1.0);
    double error = v.transpose() * qMatrix * v;

    // 确保误差为非负值（由于数值精度问题可能出现微小负值）
    return std::max(0.0, error);
}
