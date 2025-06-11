#include "MeshDecimation.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <limits>
#include <cmath>

MeshDecimation::MeshDecimation()
    : targetVertexCount_(100), maxError_(0.001)
{
}

void MeshDecimation::setTargetVertexCount(int count)
{
    targetVertexCount_ = count;
}

void MeshDecimation::setMaxError(double error)
{
    maxError_ = error;
}

void MeshDecimation::performDecimation(Mesh &mesh)
{
    if (!mesh.n_vertices())
    {
        std::cout << "空网格，无需简化" << std::endl;
        return;
    }
    std::cout << "开始QEM简化..." << std::endl;
    std::cout << "初始顶点数: " << mesh.n_vertices() << std::endl;
    std::cout << "目标顶点数: " << targetVertexCount_ << std::endl;

    // 请求必要的属性
    mesh.request_vertex_status();
    mesh.request_edge_status();
    mesh.request_face_status();
    mesh.request_face_normals();
    mesh.update_normals();

    // 为所有顶点初始化二次误差矩阵
    initializeQuadrics(mesh);

    // 计算初始边的代价
    computeEdgeCosts(mesh);

    // 主简化循环
    int collapseCount = 0;
    while (mesh.n_vertices() > targetVertexCount_ && !edgeQueue_.empty())
    {
        EdgeCollapse bestCollapse = edgeQueue_.top();
        edgeQueue_.pop();        // 检查此边是否仍然有效
        if (!mesh.is_valid_handle(bestCollapse.edge) ||
            mesh.status(bestCollapse.edge).deleted() ||
            validEdges_.find(bestCollapse.edge) == validEdges_.end())
        {
            continue;
        }

        // 检查代价是否超过阈值
        if (bestCollapse.cost > maxError_)
        {
            std::cout << "达到误差阈值: " << bestCollapse.cost << std::endl;
            break;
        }

        // 执行边折叠
        if (collapseEdge(mesh, bestCollapse))
        {
            collapseCount++;
            if (collapseCount % 100 == 0)
            {
                std::cout << "已折叠 " << collapseCount << " 条边，当前顶点数: " << mesh.n_vertices() << std::endl;
            }
        }
    }

    // 清理已删除的元素
    mesh.garbage_collection();

    std::cout << "简化完成！" << std::endl;
    std::cout << "最终顶点数: " << mesh.n_vertices() << std::endl;
    std::cout << "折叠边数: " << collapseCount << std::endl;
}

void MeshDecimation::initializeQuadrics(Mesh &mesh)
{
    vertexQuadrics_.clear();

    // 将所有顶点的二次误差矩阵初始化为零
    for (auto v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); ++v_it)
    {
        vertexQuadrics_[*v_it] = Eigen::Matrix4d::Zero();
    }

    // 从相邻面累积二次误差矩阵
    for (auto f_it = mesh.faces_begin(); f_it != mesh.faces_end(); ++f_it)
    {
        if (mesh.status(*f_it).deleted())
            continue;
        Eigen::Matrix4d faceQuadric = computeFaceQuadric(mesh, *f_it);

        // 将面的二次误差矩阵添加到面的所有顶点
        for (auto fv_it = mesh.fv_iter(*f_it); fv_it.is_valid(); ++fv_it)
        {
            vertexQuadrics_[*fv_it] += faceQuadric;
        }
    }
}

Eigen::Matrix4d MeshDecimation::computeFaceQuadric(Mesh &mesh, Mesh::FaceHandle face)
{
    // 获取面法向量和平面上的一个点
    Mesh::Normal normal = mesh.normal(face);

    // 获取面的第一个顶点来计算平面方程
    auto fv_it = mesh.fv_iter(face);
    Mesh::Point point = mesh.point(*fv_it);

    // 标准化法向量（把它变成长度为1的向量）
    double length = normal.norm();
    if (length < 1e-10)
    {
        return Eigen::Matrix4d::Zero();
    }
    normal /= length;

    // 计算平面方程: ax + by + cz + d = 0
    double a = normal[0];
    double b = normal[1];
    double c = normal[2];
    double d = -(a * point[0] + b * point[1] + c * point[2]);

    // 创建二次误差矩阵
    Eigen::Matrix4d quadric;
    quadric << a * a, a * b, a * c, a * d,
        a * b, b * b, b * c, b * d,
        a * c, b * c, c * c, c * d,
        a * d, b * d, c * d, d * d;

    return quadric;
}

void MeshDecimation::computeEdgeCosts(Mesh &mesh)
{
    // 清除之前的数据
    while (!edgeQueue_.empty())
        edgeQueue_.pop();
    validEdges_.clear();

    // 计算每条边的代价
    for (auto e_it = mesh.edges_begin(); e_it != mesh.edges_end(); ++e_it)
    {
        if (mesh.status(*e_it).deleted())
            continue;

        Eigen::Vector3d optimalPos;
        double cost = computeEdgeCost(mesh, *e_it, optimalPos);

        if (cost < std::numeric_limits<double>::infinity())
        {
            EdgeCollapse collapse;
            collapse.edge = *e_it;
            collapse.cost = cost;
            collapse.optimalPosition = optimalPos;

            edgeQueue_.push(collapse);
            validEdges_.insert(*e_it);
        }
    }
}

double MeshDecimation::computeEdgeCost(Mesh &mesh, Mesh::EdgeHandle edge, Eigen::Vector3d &optimalPos)
{
    auto heh = mesh.halfedge_handle(edge, 0);
    auto v1 = mesh.from_vertex_handle(heh);
    auto v2 = mesh.to_vertex_handle(heh);

    // 检查折叠是否有效
    if (!isValidCollapse(mesh, edge))
    {
        return std::numeric_limits<double>::infinity();
    }

    // 获取两个顶点的二次误差矩阵
    auto it1 = vertexQuadrics_.find(v1);
    auto it2 = vertexQuadrics_.find(v2);

    if (it1 == vertexQuadrics_.end() || it2 == vertexQuadrics_.end())
    {
        return std::numeric_limits<double>::infinity();
    }

    // 合并二次误差矩阵
    Eigen::Matrix4d combinedQuadric = it1->second + it2->second;

    // 尝试找到最优位置
    optimalPos = computeOptimalPosition(combinedQuadric);

    // 如果最优位置计算失败，尝试使用中点
    if (optimalPos.hasNaN())
    {
        auto p1 = mesh.point(v1);
        auto p2 = mesh.point(v2);
        optimalPos = Eigen::Vector3d((p1[0] + p2[0]) * 0.5,
                                     (p1[1] + p2[1]) * 0.5, (p1[2] + p2[2]) * 0.5);
    }

    // 计算最优位置处的误差
    Eigen::Vector4d pos(optimalPos[0], optimalPos[1], optimalPos[2], 1.0);
    double error = pos.transpose() * combinedQuadric * pos;

    return std::max(0.0, error);
}

Eigen::Vector3d MeshDecimation::computeOptimalPosition(const Eigen::Matrix4d &quadric)
{
    // 提取上3x3矩阵
    Eigen::Matrix3d A = quadric.block<3, 3>(0, 0);
    Eigen::Vector3d b = -quadric.block<3, 1>(0, 3);

    // 求解 Ax = b 得到最优位置
    Eigen::FullPivLU<Eigen::Matrix3d> lu(A);
    if (lu.isInvertible())
    {
        return lu.solve(b);
    }

    // 如果不可逆，返回NaN表示失败
    return Eigen::Vector3d(std::numeric_limits<double>::quiet_NaN(),
                           std::numeric_limits<double>::quiet_NaN(),
                           std::numeric_limits<double>::quiet_NaN());
}

bool MeshDecimation::isValidCollapse(Mesh& mesh, Mesh::EdgeHandle edge)
{
    auto heh = mesh.halfedge_handle(edge, 0);
    auto v1 = mesh.from_vertex_handle(heh);
    auto v2 = mesh.to_vertex_handle(heh);

    // 检查两个顶点是否是边界顶点
    bool v1_boundary = mesh.is_boundary(v1);
    bool v2_boundary = mesh.is_boundary(v2);
    bool edge_boundary = mesh.is_boundary(edge);

    // 如果两个顶点都是边界顶点但边不是边界边，则不允许折叠
    if (v1_boundary && v2_boundary && !edge_boundary)
    {
        return false;
    }

    // 通过法线检查折叠是否会导致面翻转
    std::vector<Mesh::FaceHandle> faces_to_check;

    // 收集不会被删除的 v1 周围的面，用于后续检查
    for (auto vf_it = mesh.vf_iter(v1); vf_it.is_valid(); ++vf_it)
    {
        bool will_be_deleted = false;
        for (auto fv_it = mesh.fv_iter(*vf_it); fv_it.is_valid(); ++fv_it)
        {
            if (*fv_it == v2)
            {
                will_be_deleted = true;
                break;
            }
        }
        if (!will_be_deleted)
        {
            faces_to_check.push_back(*vf_it);
        }
    }

    // 此处可添加更多拓扑合法性检查
    return true;
}

bool MeshDecimation::collapseEdge(Mesh& mesh, const EdgeCollapse& collapse)
{
    auto heh = mesh.halfedge_handle(collapse.edge, 0);
    auto v1 = mesh.from_vertex_handle(heh);
    auto v2 = mesh.to_vertex_handle(heh);

    // 设置新顶点位置（v1 是保留的顶点）
    mesh.set_point(v1, Mesh::Point(collapse.optimalPosition[0],
        collapse.optimalPosition[1],
        collapse.optimalPosition[2]));

    // 更新保留顶点的 Quadric 误差矩阵
    auto it1 = vertexQuadrics_.find(v1);
    auto it2 = vertexQuadrics_.find(v2);

    if (it1 != vertexQuadrics_.end() && it2 != vertexQuadrics_.end())
    {
        it1->second += it2->second;
        vertexQuadrics_.erase(it2);
    }

    // 收集邻居顶点，以便后续更新代价
    std::set<Mesh::VertexHandle> neighbors;
    for (auto vv_it = mesh.vv_iter(v1); vv_it.is_valid(); ++vv_it)
    {
        neighbors.insert(*vv_it);
    }
    for (auto vv_it = mesh.vv_iter(v2); vv_it.is_valid(); ++vv_it)
    {
        neighbors.insert(*vv_it);
    }

    // 从合法边集合中移除当前折叠边
    validEdges_.erase(collapse.edge);

    // 从合法边集合中移除与 v2 相邻的边
    for (auto ve_it = mesh.ve_iter(v2); ve_it.is_valid(); ++ve_it)
    {
        validEdges_.erase(*ve_it);
    }

    // 执行实际的边折叠操作
    if (!mesh.is_collapse_ok(heh))
    {
        return false;
    }

    mesh.collapse(heh);

    // 更新保留顶点周围边的折叠代价
    updateEdgeCosts(mesh, v1);

    return true;
}


void MeshDecimation::updateEdgeCosts(Mesh& mesh, Mesh::VertexHandle vertex)
{
    // 更新与该顶点相邻的所有边的折叠代价
    for (auto ve_it = mesh.ve_iter(vertex); ve_it.is_valid(); ++ve_it)
    {
        if (mesh.status(*ve_it).deleted())
            continue;

        Eigen::Vector3d optimalPos;
        double cost = computeEdgeCost(mesh, *ve_it, optimalPos);

        // 如果代价是有效值（非无穷），则加入队列
        if (cost < std::numeric_limits<double>::infinity())
        {
            EdgeCollapse collapse;
            collapse.edge = *ve_it;
            collapse.cost = cost;
            collapse.optimalPosition = optimalPos;

            edgeQueue_.push(collapse);       // 推入优先队列
            validEdges_.insert(*ve_it);      // 标记该边为合法边
        }
    }
}
