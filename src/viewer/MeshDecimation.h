#pragma once

#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
#include <Eigen/Dense>
#include <queue>
#include <vector>
#include <map>

// Define the mesh type (same as in OpenMeshViewer.h)
typedef OpenMesh::TriMesh_ArrayKernelT<> Mesh;

// QEM算法中的边结构，包含误差成本和目标顶点位置
struct EdgeInfo
{
    Mesh::EdgeHandle edge;        // 边句柄
    double cost;                  // 收缩误差成本
    Eigen::Vector3d targetVertex; // 最优收缩目标顶点位置

    // 用于优先队列的比较函数（最小堆）
    bool operator>(const EdgeInfo &other) const
    {
        return cost > other.cost;
    }
};

class MeshDecimation
{
public:
    MeshDecimation();
    ~MeshDecimation();

    // 对给定网格执行简化操作
    void performDecimation(Mesh &mesh);

    // 设置简化参数
    void setTargetVertexCount(int targetCount); // 设置目标顶点数量
    void setMaxError(double maxError);          // 设置最大允许误差

private:
    // QEM算法主要步骤函数

    // 初始化阶段：计算所有边的Q矩阵
    void initializeQMatrices(Mesh &mesh);

    // 计算所有边的最优收缩目标顶点
    void computeOptimalContractionTargets(Mesh &mesh);

    // 构建基于成本的优先队列
    void buildPriorityQueue(Mesh &mesh);

    // 主循环：执行边收缩直到满足条件
    void performEdgeCollapses(Mesh &mesh);

    // 辅助函数

    // 计算单个顶点的Q矩阵（基于相邻面的基础二次型）
    Eigen::Matrix4d computeVertexQMatrix(Mesh &mesh, Mesh::VertexHandle vertex);

    // 计算单个边的收缩成本和最优目标位置
    void computeEdgeCostAndTarget(Mesh &mesh, Mesh::EdgeHandle edge);

    // 执行单个边的收缩操作
    bool collapseEdge(Mesh &mesh, Mesh::EdgeHandle edge, const Eigen::Vector3d &targetPos);

    // 更新受影响边的成本
    void updateAffectedEdgeCosts(Mesh &mesh, Mesh::VertexHandle remainingVertex);

    // 从面片计算基础二次型矩阵
    Eigen::Matrix4d computeFundamentalQuadric(const Eigen::Vector3d &normal, const Eigen::Vector3d &point);

    // 计算两个Q矩阵的和
    Eigen::Matrix4d addQMatrices(const Eigen::Matrix4d &q1, const Eigen::Matrix4d &q2);

    // 解决最优收缩位置（通过求解线性系统）
    Eigen::Vector3d solveOptimalPosition(const Eigen::Matrix4d &qMatrix,
                                         const Eigen::Vector3d &v1,
                                         const Eigen::Vector3d &v2);

    // 计算给定位置的二次误差
    double computeQuadricError(const Eigen::Matrix4d &qMatrix, const Eigen::Vector3d &position);

private:
    // 算法参数
    int targetVertexCount_;  // 目标顶点数量
    double maxAllowedError_; // 最大允许误差阈值

    // 数据结构
    std::map<Mesh::VertexHandle, Eigen::Matrix4d> vertexQMatrices_;                          // 每个顶点的Q矩阵
    std::map<Mesh::EdgeHandle, EdgeInfo> edgeInfoMap_;                                       // 边信息映射
    std::priority_queue<EdgeInfo, std::vector<EdgeInfo>, std::greater<EdgeInfo>> edgeQueue_; // 边的优先队列

    // 内部状态
    bool isInitialized_; // 是否已初始化
};
