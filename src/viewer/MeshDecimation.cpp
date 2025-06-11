#include "MeshDecimation.h"
#include <iostream>

MeshDecimation::MeshDecimation()
    : targetVertexCount_(100) // 默认目标顶点数量
      ,
      maxAllowedError_(1e-3) // 默认最大允许误差
      ,
      isInitialized_(false) // 初始化状态为false
{
    // TODO: 可以在这里添加其他初始化参数
}

MeshDecimation::~MeshDecimation()
{
    // TODO: 如果需要，进行清理工作
}

void MeshDecimation::performDecimation(Mesh &mesh)
{
    std::cout << "开始执行QEM网格简化算法..." << std::endl;

    // QEM算法主要步骤：

    // 1. 初始化：计算所有边的Q矩阵
    initializeQMatrices(mesh);

    // 2. 计算所有边的最优收缩目标顶点
    computeOptimalContractionTargets(mesh);

    // 3. 构建基于成本的优先队列
    buildPriorityQueue(mesh);

    // 4. 主循环：当顶点数 > 目标数量 且 最小成本 < 阈值时继续收缩
    performEdgeCollapses(mesh);

    std::cout << "QEM网格简化完成!" << std::endl;
}

void MeshDecimation::setTargetVertexCount(int targetCount)
{
    targetVertexCount_ = targetCount;
}

void MeshDecimation::setMaxError(double maxError)
{
    maxAllowedError_ = maxError;
}

void MeshDecimation::initializeQMatrices(Mesh &mesh)
{
    std::cout << "初始化Q矩阵..." << std::endl;

    // TODO: 实现Q矩阵初始化
    // 1. 遍历所有顶点
    // 2. 对每个顶点，计算其相邻面的基础二次型
    // 3. 将相邻面的基础二次型累加得到顶点的Q矩阵

    for (auto v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); ++v_it)
    {
        Mesh::VertexHandle vertex = *v_it;

        // 计算顶点的Q矩阵
        Eigen::Matrix4d qMatrix = computeVertexQMatrix(mesh, vertex);
        vertexQMatrices_[vertex] = qMatrix;
    }

    isInitialized_ = true;
}

void MeshDecimation::computeOptimalContractionTargets(Mesh &mesh)
{
    std::cout << "计算最优收缩目标..." << std::endl;

    // TODO: 实现最优收缩目标计算
    // 遍历所有边，计算每条边的收缩成本和目标位置

    for (auto e_it = mesh.edges_begin(); e_it != mesh.edges_end(); ++e_it)
    {
        Mesh::EdgeHandle edge = *e_it;
        computeEdgeCostAndTarget(mesh, edge);
    }
}

void MeshDecimation::buildPriorityQueue(Mesh &mesh)
{
    std::cout << "构建优先队列..." << std::endl;

    // TODO: 实现优先队列构建
    // 将所有边按照成本放入优先队列，最小成本在顶部

    // 清空队列
    while (!edgeQueue_.empty())
    {
        edgeQueue_.pop();
    }

    // 将所有边加入优先队列
    for (const auto &pair : edgeInfoMap_)
    {
        edgeQueue_.push(pair.second);
    }
}

void MeshDecimation::performEdgeCollapses(Mesh &mesh)
{
    std::cout << "执行边收缩..." << std::endl;

    int currentVertexCount = mesh.n_vertices();

    // TODO: 实现主循环
    // 当顶点数 > 目标数量 且 最小成本 < 阈值时继续

    while (currentVertexCount > targetVertexCount_ && !edgeQueue_.empty())
    {
        // 取出成本最小的边
        EdgeInfo edgeInfo = edgeQueue_.top();
        edgeQueue_.pop();

        // 检查成本是否超过阈值
        if (edgeInfo.cost > maxAllowedError_)
        {
            std::cout << "达到误差阈值，停止简化" << std::endl;
            break;
        }

        // 执行边收缩
        if (collapseEdge(mesh, edgeInfo.edge, edgeInfo.targetVertex))
        {
            currentVertexCount--;

            // 更新受影响边的成本
            // TODO: 获取保留的顶点并更新相关边
        }
    }

    std::cout << "简化完成，当前顶点数: " << currentVertexCount << std::endl;
}

Eigen::Matrix4d MeshDecimation::computeVertexQMatrix(Mesh &mesh, Mesh::VertexHandle vertex)
{
    // TODO: 实现顶点Q矩阵计算
    // 1. 初始化零矩阵
    // 2. 遍历顶点的相邻面
    // 3. 对每个面计算基础二次型并累加

    Eigen::Matrix4d qMatrix = Eigen::Matrix4d::Zero();

    // 遍历顶点的相邻面
    for (auto vf_it = mesh.vf_iter(vertex); vf_it.is_valid(); ++vf_it)
    {
        Mesh::FaceHandle face = *vf_it;

        // 计算面的法向量和一个点
        Mesh::Normal normal = mesh.normal(face);
        Mesh::Point point = mesh.point(vertex); // 使用当前顶点作为面上的点

        // 转换为Eigen向量
        Eigen::Vector3d normalVec(normal[0], normal[1], normal[2]);
        Eigen::Vector3d pointVec(point[0], point[1], point[2]);

        // 计算并累加基础二次型
        Eigen::Matrix4d faceQuadric = computeFundamentalQuadric(normalVec, pointVec);
        qMatrix += faceQuadric;
    }

    return qMatrix;
}

void MeshDecimation::computeEdgeCostAndTarget(Mesh &mesh, Mesh::EdgeHandle edge)
{
    // TODO: 实现边收缩成本和目标位置计算
    // 1. 获取边的两个顶点
    // 2. 计算组合Q矩阵 (Q1 + Q2)
    // 3. 求解最优位置
    // 4. 计算在该位置的误差成本

    auto halfedge = mesh.halfedge_handle(edge, 0);
    auto v1 = mesh.from_vertex_handle(halfedge);
    auto v2 = mesh.to_vertex_handle(halfedge);

    // 获取两个顶点的Q矩阵
    Eigen::Matrix4d q1 = vertexQMatrices_[v1];
    Eigen::Matrix4d q2 = vertexQMatrices_[v2];
    Eigen::Matrix4d qCombined = addQMatrices(q1, q2);

    // 计算最优位置
    Eigen::Vector3d pos1(mesh.point(v1)[0], mesh.point(v1)[1], mesh.point(v1)[2]);
    Eigen::Vector3d pos2(mesh.point(v2)[0], mesh.point(v2)[1], mesh.point(v2)[2]);

    Eigen::Vector3d optimalPos = solveOptimalPosition(qCombined, pos1, pos2);

    // 计算误差成本
    double cost = computeQuadricError(qCombined, optimalPos);

    // 存储边信息
    EdgeInfo edgeInfo;
    edgeInfo.edge = edge;
    edgeInfo.cost = cost;
    edgeInfo.targetVertex = optimalPos;

    edgeInfoMap_[edge] = edgeInfo;
}

bool MeshDecimation::collapseEdge(Mesh &mesh, Mesh::EdgeHandle edge, const Eigen::Vector3d &targetPos)
{
    // TODO: 实现边收缩操作
    // 1. 检查边是否仍然有效
    // 2. 获取边的两个顶点
    // 3. 将一个顶点移动到目标位置，删除另一个顶点
    // 4. 更新网格拓扑
    // 5. 更新相关顶点的Q矩阵

    if (!mesh.is_valid_handle(edge))
        return false;

    // 暂时返回false，等待具体实现
    return false;
}

void MeshDecimation::updateAffectedEdgeCosts(Mesh &mesh, Mesh::VertexHandle remainingVertex)
{
    // TODO: 实现受影响边成本更新
    // 1. 遍历保留顶点的相邻边
    // 2. 重新计算这些边的成本和目标位置
    // 3. 更新优先队列

    for (auto ve_it = mesh.ve_iter(remainingVertex); ve_it.is_valid(); ++ve_it)
    {
        Mesh::EdgeHandle edge = *ve_it;
        computeEdgeCostAndTarget(mesh, edge);
    }
}

Eigen::Matrix4d MeshDecimation::computeFundamentalQuadric(const Eigen::Vector3d &normal, const Eigen::Vector3d &point)
{
    // TODO: 实现基础二次型矩阵计算
    // 根据平面方程 ax + by + cz + d = 0 构造4x4矩阵

    // 计算平面方程系数
    double a = normal.x();
    double b = normal.y();
    double c = normal.z();
    double d = -(normal.dot(point)); // d = -(ax0 + by0 + cz0)

    // 构造基础二次型矩阵
    Eigen::Matrix4d K;
    K << a * a, a * b, a * c, a * d,
        a * b, b * b, b * c, b * d,
        a * c, b * c, c * c, c * d,
        a * d, b * d, c * d, d * d;

    return K;
}

Eigen::Matrix4d MeshDecimation::addQMatrices(const Eigen::Matrix4d &q1, const Eigen::Matrix4d &q2)
{
    // TODO: 实现Q矩阵相加
    return q1 + q2;
}

Eigen::Vector3d MeshDecimation::solveOptimalPosition(const Eigen::Matrix4d &qMatrix,
                                                     const Eigen::Vector3d &v1,
                                                     const Eigen::Vector3d &v2)
{
    // TODO: 实现最优位置求解
    // 1. 尝试求解线性系统 Q'v = 0 (其中Q'是Q的3x3子矩阵)
    // 2. 如果系统不可解，则在三个候选位置中选择误差最小的：v1, v2, (v1+v2)/2

    // 提取3x3子矩阵
    Eigen::Matrix3d Q3x3 = qMatrix.block<3, 3>(0, 0);
    Eigen::Vector3d q3x1 = qMatrix.block<3, 1>(0, 3);

    // 尝试求解线性系统
    if (Q3x3.determinant() != 0)
    {
        return -Q3x3.inverse() * q3x1;
    }
    else
    {
        // 系统不可解，选择三个候选位置中误差最小的
        Eigen::Vector3d midPoint = (v1 + v2) * 0.5;

        double error1 = computeQuadricError(qMatrix, v1);
        double error2 = computeQuadricError(qMatrix, v2);
        double errorMid = computeQuadricError(qMatrix, midPoint);

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
    // TODO: 实现二次误差计算
    // 计算 v^T * Q * v，其中v = [x, y, z, 1]^T

    Eigen::Vector4d v(position.x(), position.y(), position.z(), 1.0);
    return v.transpose() * qMatrix * v;
}
