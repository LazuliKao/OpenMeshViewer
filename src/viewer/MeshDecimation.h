#pragma once

#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
#include <Eigen/Dense>
#include <queue>
#include <vector>
#include <map>
#include <set>

// 网格类型前向声明
typedef OpenMesh::TriMesh_ArrayKernelT<> Mesh;

// 用于优先队列的边折叠结构
struct EdgeCollapse
{
    Mesh::EdgeHandle edge;
    double cost;
    Eigen::Vector3d optimalPosition;

    bool operator>(const EdgeCollapse &other) const
    {
        return cost > other.cost; // 用于最小堆
    }
};

class MeshDecimation
{
public:
    // 构造函数
    MeshDecimation();

    // 设置简化参数
    void setTargetVertexCount(int count);
    void setMaxError(double error);

    // 主要简化函数
    void performDecimation(Mesh &mesh);

private:
    // QEM算法组件
    void initializeQuadrics(Mesh &mesh);
    void computeEdgeCosts(Mesh &mesh);
    bool collapseEdge(Mesh &mesh, const EdgeCollapse &collapse);
    void updateEdgeCosts(Mesh &mesh, Mesh::VertexHandle vertex); // 实用函数
    Eigen::Matrix4d computeFaceQuadric(Mesh &mesh, Mesh::FaceHandle face);
    double computeEdgeCost(Mesh &mesh, Mesh::EdgeHandle edge, Eigen::Vector3d &optimalPos);
    Eigen::Vector3d computeOptimalPosition(const Eigen::Matrix4d &quadric);
    bool isValidCollapse(Mesh &mesh, Mesh::EdgeHandle edge);

    // 成员变量
    int targetVertexCount_;
    double maxError_;

    // QEM数据结构
    // 顶点二次误差矩阵映射表，存储每个顶点的4x4二次误差矩阵用于计算边折叠代价
    std::map<Mesh::VertexHandle, Eigen::Matrix4d> vertexQuadrics_;
    // 边折叠优先队列，按照折叠代价从小到大排序，确保每次选择代价最小的边进行折叠
    std::priority_queue<EdgeCollapse, std::vector<EdgeCollapse>, std::greater<EdgeCollapse>> edgeQueue_;
    // 有效边集合，存储当前网格中可以进行折叠操作的边句柄
    std::set<Mesh::EdgeHandle> validEdges_;
};
