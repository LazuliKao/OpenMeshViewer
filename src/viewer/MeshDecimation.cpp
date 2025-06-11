#include "MeshDecimation.h"
#include <iostream>

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
    std::cout << "开始执行QEM网格简化算法..." << std::endl;

    // 验证输入参数
    if (targetVertexCount_ <= 0)
    {
        std::cout << "错误：目标顶点数必须大于0" << std::endl;
        return;
    }

    int originalVertexCount = mesh.n_vertices();
    if (originalVertexCount <= targetVertexCount_)
    {
        std::cout << "网格已经满足目标顶点数要求，无需简化" << std::endl;
        return;
    }

    if (mesh.n_faces() == 0)
    {
        std::cout << "错误：输入网格没有面片" << std::endl;
        return;
    }

    std::cout << "原始网格：" << originalVertexCount << " 顶点，"
              << mesh.n_faces() << " 面片，目标：" << targetVertexCount_ << " 顶点" << std::endl;

    // QEM算法主要步骤：

    // 1. 初始化：计算所有顶点的Q矩阵
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
    if (targetCount <= 0)
    {
        std::cout << "警告：目标顶点数必须大于0，设置为默认值100" << std::endl;
        targetVertexCount_ = 100;
    }
    else
    {
        targetVertexCount_ = targetCount;
        std::cout << "目标顶点数设置为：" << targetVertexCount_ << std::endl;
    }
}

void MeshDecimation::setMaxError(double maxError)
{
    if (maxError < 0)
    {
        std::cout << "警告：最大误差不能为负数，设置为默认值1e-3" << std::endl;
        maxAllowedError_ = 1e-3;
    }
    else
    {
        maxAllowedError_ = maxError;
        std::cout << "最大允许误差设置为：" << maxAllowedError_ << std::endl;
    }
}

void MeshDecimation::initializeQMatrices(Mesh &mesh)
{
    std::cout << "初始化Q矩阵..." << std::endl;

    // 确保网格有正确计算的法向量
    mesh.request_face_normals();
    mesh.request_vertex_normals();
    mesh.update_normals();

    // 清空之前的Q矩阵
    vertexQMatrices_.clear();

    // 1. 遍历所有顶点
    for (auto v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); ++v_it)
    {
        Mesh::VertexHandle vertex = *v_it;

        // 2. 对每个顶点，计算其相邻面的基础二次型
        // 3. 将相邻面的基础二次型累加得到顶点的Q矩阵
        Eigen::Matrix4d qMatrix = computeVertexQMatrix(mesh, vertex);
        vertexQMatrices_[vertex] = qMatrix;
    }

    isInitialized_ = true;
    std::cout << "完成 " << vertexQMatrices_.size() << " 个顶点的Q矩阵计算" << std::endl;
}

void MeshDecimation::computeOptimalContractionTargets(Mesh &mesh)
{
    std::cout << "计算最优收缩目标..." << std::endl;

    // 清空之前的边信息
    edgeInfoMap_.clear();

    // 遍历所有边，计算每条边的收缩成本和目标位置
    int edgeCount = 0;
    for (auto e_it = mesh.edges_begin(); e_it != mesh.edges_end(); ++e_it)
    {
        Mesh::EdgeHandle edge = *e_it;
        computeEdgeCostAndTarget(mesh, edge);
        edgeCount++;
    }

    std::cout << "完成 " << edgeCount << " 条边的成本计算" << std::endl;
}

void MeshDecimation::buildPriorityQueue(Mesh &mesh)
{
    std::cout << "构建优先队列..." << std::endl;

    // 清空队列
    while (!edgeQueue_.empty())
    {
        edgeQueue_.pop();
    }

    // 将所有边按照成本放入优先队列，最小成本在顶部
    for (const auto &pair : edgeInfoMap_)
    {
        edgeQueue_.push(pair.second);
    }

    std::cout << "优先队列构建完成，包含 " << edgeQueue_.size() << " 条边" << std::endl;
}

void MeshDecimation::performEdgeCollapses(Mesh &mesh)
{
    std::cout << "执行边收缩..." << std::endl;
    int currentVertexCount = mesh.n_vertices();
    int originalVertexCount = currentVertexCount; // 主循环：当顶点数 > 目标数量 且 最小成本 < 阈值时继续
    while (currentVertexCount > targetVertexCount_ && !edgeQueue_.empty())
    {
        // 取出成本最小的边
        EdgeInfo edgeInfo = edgeQueue_.top();
        edgeQueue_.pop();

        // 检查边是否仍然有效（可能在之前的操作中被删除）
        if (!mesh.is_valid_handle(edgeInfo.edge))
            continue;

        // 检查成本是否超过阈值
        if (edgeInfo.cost > maxAllowedError_)
        {
            std::cout << "达到误差阈值 " << maxAllowedError_ << "，停止简化" << std::endl;
            break;
        }

        // 获取边的顶点以便后续更新，增加验证
        auto halfedge = mesh.halfedge_handle(edgeInfo.edge, 0);
        if (!mesh.is_valid_handle(halfedge))
            continue;

        auto remainingVertex = mesh.from_vertex_handle(halfedge);
        if (!mesh.is_valid_handle(remainingVertex))
            continue;

        // 执行边收缩
        if (collapseEdge(mesh, edgeInfo.edge, edgeInfo.targetVertex))
        {
            currentVertexCount--;

            // 更新受影响边的成本
            updateAffectedEdgeCosts(mesh, remainingVertex);

            // 定期输出进度
            if ((originalVertexCount - currentVertexCount) % 100 == 0)
            {
                std::cout << "已简化 " << (originalVertexCount - currentVertexCount)
                          << " 个顶点，当前顶点数: " << currentVertexCount << std::endl;
            }
        }
    }

    // 执行垃圾回收以清理被删除的元素
    mesh.garbage_collection();

    std::cout << "简化完成，从 " << originalVertexCount << " 顶点简化到 "
              << mesh.n_vertices() << " 顶点" << std::endl;
}

Eigen::Matrix4d MeshDecimation::computeVertexQMatrix(Mesh &mesh, Mesh::VertexHandle vertex)
{
    // 1. 初始化零矩阵
    Eigen::Matrix4d qMatrix = Eigen::Matrix4d::Zero();

    // 2. 遍历顶点的相邻面
    for (auto vf_it = mesh.vf_iter(vertex); vf_it.is_valid(); ++vf_it)
    {
        Mesh::FaceHandle face = *vf_it;

        // 确保面的法向量是正确计算的
        mesh.update_normal(face);

        // 计算面的法向量
        Mesh::Normal normal = mesh.normal(face);

        // 获取面上的一个顶点作为参考点
        auto fv_it = mesh.fv_iter(face);
        Mesh::Point point = mesh.point(*fv_it);

        // 转换为Eigen向量
        Eigen::Vector3d normalVec(normal[0], normal[1], normal[2]);
        normalVec.normalize(); // 确保法向量是单位向量

        Eigen::Vector3d pointVec(point[0], point[1], point[2]);

        // 3. 计算并累加基础二次型
        Eigen::Matrix4d faceQuadric = computeFundamentalQuadric(normalVec, pointVec);
        qMatrix += faceQuadric;
    }

    // 4. 处理边界边：如果顶点在边界上，为边界边添加额外的约束
    for (auto ve_it = mesh.ve_iter(vertex); ve_it.is_valid(); ++ve_it)
    {
        Mesh::EdgeHandle edge = *ve_it;
        if (mesh.is_boundary(edge))
        {
            // 为边界边添加额外的二次型约束以保持边界形状
            auto halfedge = mesh.halfedge_handle(edge, 0);
            auto v1 = mesh.from_vertex_handle(halfedge);
            auto v2 = mesh.to_vertex_handle(halfedge);

            Mesh::Point p1 = mesh.point(v1);
            Mesh::Point p2 = mesh.point(v2);

            // 计算边的方向向量
            Eigen::Vector3d edgeDir(p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]);
            edgeDir.normalize();

            // 为边界添加一个垂直于边的平面约束（权重较大以保持边界）
            Eigen::Vector3d boundaryNormal = edgeDir.cross(Eigen::Vector3d(0, 0, 1));
            if (boundaryNormal.norm() < 0.1)
            {
                boundaryNormal = edgeDir.cross(Eigen::Vector3d(0, 1, 0));
            }
            boundaryNormal.normalize();

            Eigen::Vector3d boundaryPoint(p1[0], p1[1], p1[2]);
            Eigen::Matrix4d boundaryQuadric = computeFundamentalQuadric(boundaryNormal, boundaryPoint);

            // 给边界约束更大的权重
            qMatrix += 1000.0 * boundaryQuadric;
        }
    }

    return qMatrix;
}

void MeshDecimation::computeEdgeCostAndTarget(Mesh &mesh, Mesh::EdgeHandle edge)
{
    // 0. 首先检查边本身是否有效
    if (!mesh.is_valid_handle(edge))
        return;

    // 1. 获取边的两个顶点
    auto halfedge = mesh.halfedge_handle(edge, 0);

    // 检查半边是否有效
    if (!mesh.is_valid_handle(halfedge))
        return;

    auto v1 = mesh.from_vertex_handle(halfedge);
    auto v2 = mesh.to_vertex_handle(halfedge);

    // 确保顶点有效
    if (!mesh.is_valid_handle(v1) || !mesh.is_valid_handle(v2))
        return;

    // 2. 获取两个顶点的Q矩阵并计算组合Q矩阵 (Q1 + Q2)
    auto it1 = vertexQMatrices_.find(v1);
    auto it2 = vertexQMatrices_.find(v2);

    if (it1 == vertexQMatrices_.end() || it2 == vertexQMatrices_.end())
        return;

    Eigen::Matrix4d q1 = it1->second;
    Eigen::Matrix4d q2 = it2->second;
    Eigen::Matrix4d qCombined = addQMatrices(q1, q2);

    // 3. 计算最优位置
    Eigen::Vector3d pos1(mesh.point(v1)[0], mesh.point(v1)[1], mesh.point(v1)[2]);
    Eigen::Vector3d pos2(mesh.point(v2)[0], mesh.point(v2)[1], mesh.point(v2)[2]);

    Eigen::Vector3d optimalPos = solveOptimalPosition(qCombined, pos1, pos2);

    // 4. 计算在该位置的误差成本
    double cost = computeQuadricError(qCombined, optimalPos);

    // 确保成本为非负值
    if (cost < 0)
        cost = 0;

    // 5. 存储边信息
    EdgeInfo edgeInfo;
    edgeInfo.edge = edge;
    edgeInfo.cost = cost;
    edgeInfo.targetVertex = optimalPos;

    edgeInfoMap_[edge] = edgeInfo;
}

bool MeshDecimation::collapseEdge(Mesh &mesh, Mesh::EdgeHandle edge, const Eigen::Vector3d &targetPos)
{
    // 1. 检查边是否仍然有效
    if (!mesh.is_valid_handle(edge))
        return false;

    // 2. 获取边的两个顶点
    auto halfedge = mesh.halfedge_handle(edge, 0);
    auto v1 = mesh.from_vertex_handle(halfedge);
    auto v2 = mesh.to_vertex_handle(halfedge);

    // 确保顶点有效
    if (!mesh.is_valid_handle(v1) || !mesh.is_valid_handle(v2))
        return false;

    // 3. 检查边收缩是否可行（防止拓扑错误）
    if (!mesh.is_collapse_ok(halfedge))
        return false;

    // 4. 将第一个顶点移动到目标位置
    Mesh::Point newPos(targetPos.x(), targetPos.y(), targetPos.z());
    mesh.set_point(v1, newPos);

    // 5. 更新保留顶点的Q矩阵
    // 将两个顶点的Q矩阵相加作为保留顶点的新Q矩阵
    auto it1 = vertexQMatrices_.find(v1);
    auto it2 = vertexQMatrices_.find(v2);

    if (it1 != vertexQMatrices_.end() && it2 != vertexQMatrices_.end())
    {
        Eigen::Matrix4d newQMatrix = addQMatrices(it1->second, it2->second);
        vertexQMatrices_[v1] = newQMatrix;

        // 移除被删除顶点的Q矩阵
        vertexQMatrices_.erase(it2);
    } // 6. 执行边收缩操作（这会删除半边、边和一个顶点）
    mesh.collapse(halfedge);

    // 7. 立即清理无效的边信息，避免后续访问无效句柄
    auto edgeInfoIt = edgeInfoMap_.begin();
    while (edgeInfoIt != edgeInfoMap_.end())
    {
        // 检查边是否仍然有效（收缩操作可能删除了一些边）
        if (!mesh.is_valid_handle(edgeInfoIt->first))
        {
            edgeInfoIt = edgeInfoMap_.erase(edgeInfoIt);
        }
        else
        {
            ++edgeInfoIt;
        }
    }

    return true;
}

void MeshDecimation::updateAffectedEdgeCosts(Mesh &mesh, Mesh::VertexHandle remainingVertex)
{
    // 1. 首先清理无效的边信息
    auto edgeInfoIt = edgeInfoMap_.begin();
    while (edgeInfoIt != edgeInfoMap_.end())
    {
        if (!mesh.is_valid_handle(edgeInfoIt->first))
        {
            edgeInfoIt = edgeInfoMap_.erase(edgeInfoIt);
        }
        else
        {
            ++edgeInfoIt;
        }
    }

    // 2. 收集保留顶点的相邻边
    std::vector<Mesh::EdgeHandle> adjacentEdges;
    for (auto ve_it = mesh.ve_iter(remainingVertex); ve_it.is_valid(); ++ve_it)
    {
        Mesh::EdgeHandle edge = *ve_it;
        if (mesh.is_valid_handle(edge))
        {
            adjacentEdges.push_back(edge);
        }
    }

    // 3. 重新计算相邻边的成本和目标位置
    for (const auto &edge : adjacentEdges)
    {
        if (mesh.is_valid_handle(edge))
        {
            computeEdgeCostAndTarget(mesh, edge);
        }
    }

    // 4. 重建优先队列以反映更新的成本
    // 清空当前队列
    while (!edgeQueue_.empty())
    {
        edgeQueue_.pop();
    }

    // 重新添加所有有效边到队列
    for (const auto &pair : edgeInfoMap_)
    {
        // 再次验证句柄有效性，因为在计算过程中可能有更多变化
        if (mesh.is_valid_handle(pair.first))
        {
            edgeQueue_.push(pair.second);
        }
    }
}

Eigen::Matrix4d MeshDecimation::computeFundamentalQuadric(const Eigen::Vector3d &normal, const Eigen::Vector3d &point)
{
    // 根据平面方程 ax + by + cz + d = 0 构造4x4基础二次型矩阵
    // 其中法向量 (a, b, c) = normal，d = -(ax0 + by0 + cz0)

    // 计算平面方程系数
    double a = normal.x();
    double b = normal.y();
    double c = normal.z();
    double d = -(normal.dot(point)); // d = -(ax0 + by0 + cz0)

    // 构造基础二次型矩阵 K = pp^T，其中 p = [a, b, c, d]^T
    Eigen::Matrix4d K;
    K << a * a, a * b, a * c, a * d,
        a * b, b * b, b * c, b * d,
        a * c, b * c, c * c, c * d,
        a * d, b * d, c * d, d * d;

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
