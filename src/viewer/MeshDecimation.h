#pragma once

#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
#include <Eigen/Dense>
#include <queue>
#include <vector>
#include <map>
#include <set>

// Forward declaration of Mesh type
typedef OpenMesh::TriMesh_ArrayKernelT<> Mesh;

// Edge collapse structure for priority queue
struct EdgeCollapse
{
    Mesh::EdgeHandle edge;
    double cost;
    Eigen::Vector3d optimalPosition;

    bool operator>(const EdgeCollapse &other) const
    {
        return cost > other.cost; // For min-heap
    }
};

class MeshDecimation
{
public:
    // Constructor
    MeshDecimation();

    // Set decimation parameters
    void setTargetVertexCount(int count);
    void setMaxError(double error);

    // Main decimation function
    void performDecimation(Mesh &mesh);

private:
    // QEM algorithm components
    void initializeQuadrics(Mesh &mesh);
    void computeEdgeCosts(Mesh &mesh);
    bool collapseEdge(Mesh &mesh, const EdgeCollapse &collapse);
    void updateEdgeCosts(Mesh &mesh, Mesh::VertexHandle vertex);    // Utility functions
    Eigen::Matrix4d computeFaceQuadric(Mesh &mesh, Mesh::FaceHandle face);
    double computeEdgeCost(Mesh &mesh, Mesh::EdgeHandle edge, Eigen::Vector3d &optimalPos);
    Eigen::Vector3d computeOptimalPosition(const Eigen::Matrix4d &quadric);
    bool isValidCollapse(Mesh &mesh, Mesh::EdgeHandle edge);

    // Member variables
    int targetVertexCount_;
    double maxError_;

    // QEM data structures
    std::map<Mesh::VertexHandle, Eigen::Matrix4d> vertexQuadrics_;
    std::priority_queue<EdgeCollapse, std::vector<EdgeCollapse>, std::greater<EdgeCollapse>> edgeQueue_;
    std::set<Mesh::EdgeHandle> validEdges_;
};
