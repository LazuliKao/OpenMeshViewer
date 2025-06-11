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
        std::cout << "Empty mesh, nothing to decimate" << std::endl;
        return;
    }

    std::cout << "Starting QEM decimation..." << std::endl;
    std::cout << "Initial vertices: " << mesh.n_vertices() << std::endl;
    std::cout << "Target vertices: " << targetVertexCount_ << std::endl;

    // Request necessary properties
    mesh.request_vertex_status();
    mesh.request_edge_status();
    mesh.request_face_status();
    mesh.request_face_normals();
    mesh.update_normals();

    // Initialize quadrics for all vertices
    initializeQuadrics(mesh);

    // Compute initial edge costs
    computeEdgeCosts(mesh);

    // Main decimation loop
    int collapseCount = 0;
    while (mesh.n_vertices() > targetVertexCount_ && !edgeQueue_.empty())
    {
        EdgeCollapse bestCollapse = edgeQueue_.top();
        edgeQueue_.pop();

        // Check if this edge is still valid
        if (!mesh.is_valid_handle(bestCollapse.edge) ||
            mesh.status(bestCollapse.edge).deleted() ||
            validEdges_.find(bestCollapse.edge) == validEdges_.end())
        {
            continue;
        }

        // Check if cost exceeds threshold
        if (bestCollapse.cost > maxError_)
        {
            std::cout << "Reached error threshold: " << bestCollapse.cost << std::endl;
            break;
        }

        // Perform edge collapse
        if (collapseEdge(mesh, bestCollapse))
        {
            collapseCount++;
            if (collapseCount % 100 == 0)
            {
                std::cout << "Collapsed " << collapseCount << " edges, vertices: " << mesh.n_vertices() << std::endl;
            }
        }
    }

    // Clean up deleted elements
    mesh.garbage_collection();

    std::cout << "Decimation complete!" << std::endl;
    std::cout << "Final vertices: " << mesh.n_vertices() << std::endl;
    std::cout << "Collapsed edges: " << collapseCount << std::endl;
}

void MeshDecimation::initializeQuadrics(Mesh &mesh)
{
    vertexQuadrics_.clear();

    // Initialize all vertex quadrics to zero
    for (auto v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); ++v_it)
    {
        vertexQuadrics_[*v_it] = Eigen::Matrix4d::Zero();
    }

    // Accumulate quadrics from adjacent faces
    for (auto f_it = mesh.faces_begin(); f_it != mesh.faces_end(); ++f_it)
    {
        if (mesh.status(*f_it).deleted())
            continue;

        Eigen::Matrix4d faceQuadric = computeFaceQuadric(mesh, *f_it);

        // Add face quadric to all vertices of the face
        for (auto fv_it = mesh.fv_iter(*f_it); fv_it.is_valid(); ++fv_it)
        {
            vertexQuadrics_[*fv_it] += faceQuadric;
        }
    }
}

Eigen::Matrix4d MeshDecimation::computeFaceQuadric(Mesh &mesh, Mesh::FaceHandle face)
{
    // Get face normal and a point on the plane
    Mesh::Normal normal = mesh.normal(face);

    // Get first vertex of face to compute plane equation
    auto fv_it = mesh.fv_iter(face);
    Mesh::Point point = mesh.point(*fv_it);

    // Normalize normal vector
    double length = normal.norm();
    if (length < 1e-10)
    {
        return Eigen::Matrix4d::Zero();
    }
    normal /= length;

    // Compute plane equation: ax + by + cz + d = 0
    double a = normal[0];
    double b = normal[1];
    double c = normal[2];
    double d = -(a * point[0] + b * point[1] + c * point[2]);

    // Create quadric matrix
    Eigen::Matrix4d quadric;
    quadric << a * a, a * b, a * c, a * d,
        a * b, b * b, b * c, b * d,
        a * c, b * c, c * c, c * d,
        a * d, b * d, c * d, d * d;

    return quadric;
}

void MeshDecimation::computeEdgeCosts(Mesh &mesh)
{
    // Clear previous data
    while (!edgeQueue_.empty())
        edgeQueue_.pop();
    validEdges_.clear();

    // Compute cost for each edge
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

    // Check if collapse is valid
    if (!isValidCollapse(mesh, edge))
    {
        return std::numeric_limits<double>::infinity();
    }

    // Get quadrics for both vertices
    auto it1 = vertexQuadrics_.find(v1);
    auto it2 = vertexQuadrics_.find(v2);

    if (it1 == vertexQuadrics_.end() || it2 == vertexQuadrics_.end())
    {
        return std::numeric_limits<double>::infinity();
    }

    // Combine quadrics
    Eigen::Matrix4d combinedQuadric = it1->second + it2->second;

    // Try to find optimal position
    optimalPos = computeOptimalPosition(combinedQuadric);

    // If optimal position computation failed, try midpoint
    if (optimalPos.hasNaN())
    {
        auto p1 = mesh.point(v1);
        auto p2 = mesh.point(v2);
        optimalPos = Eigen::Vector3d((p1[0] + p2[0]) * 0.5,
                                     (p1[1] + p2[1]) * 0.5,
                                     (p1[2] + p2[2]) * 0.5);
    }

    // Compute error at optimal position
    Eigen::Vector4d pos(optimalPos[0], optimalPos[1], optimalPos[2], 1.0);
    double error = pos.transpose() * combinedQuadric * pos;

    return std::max(0.0, error);
}

Eigen::Vector3d MeshDecimation::computeOptimalPosition(const Eigen::Matrix4d &quadric)
{
    // Extract upper 3x3 matrix
    Eigen::Matrix3d A = quadric.block<3, 3>(0, 0);
    Eigen::Vector3d b = -quadric.block<3, 1>(0, 3);

    // Solve Ax = b for optimal position
    Eigen::FullPivLU<Eigen::Matrix3d> lu(A);
    if (lu.isInvertible())
    {
        return lu.solve(b);
    }

    // If not invertible, return NaN to indicate failure
    return Eigen::Vector3d(std::numeric_limits<double>::quiet_NaN(),
                           std::numeric_limits<double>::quiet_NaN(),
                           std::numeric_limits<double>::quiet_NaN());
}

bool MeshDecimation::isValidCollapse(Mesh &mesh, Mesh::EdgeHandle edge)
{
    auto heh = mesh.halfedge_handle(edge, 0);
    auto v1 = mesh.from_vertex_handle(heh);
    auto v2 = mesh.to_vertex_handle(heh);

    // Check if vertices are boundary vertices
    bool v1_boundary = mesh.is_boundary(v1);
    bool v2_boundary = mesh.is_boundary(v2);
    bool edge_boundary = mesh.is_boundary(edge);

    // Don't collapse if both vertices are boundary but edge is not
    if (v1_boundary && v2_boundary && !edge_boundary)
    {
        return false;
    }

    // Check for face flip by examining normals
    std::vector<Mesh::FaceHandle> faces_to_check;

    // Collect faces around v1 that won't be deleted
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

    // Additional topological checks could be added here
    return true;
}

bool MeshDecimation::collapseEdge(Mesh &mesh, const EdgeCollapse &collapse)
{
    auto heh = mesh.halfedge_handle(collapse.edge, 0);
    auto v1 = mesh.from_vertex_handle(heh);
    auto v2 = mesh.to_vertex_handle(heh);

    // Set new position for v1 (the vertex that will remain)
    mesh.set_point(v1, Mesh::Point(collapse.optimalPosition[0],
                                   collapse.optimalPosition[1],
                                   collapse.optimalPosition[2]));

    // Update quadric for remaining vertex
    auto it1 = vertexQuadrics_.find(v1);
    auto it2 = vertexQuadrics_.find(v2);

    if (it1 != vertexQuadrics_.end() && it2 != vertexQuadrics_.end())
    {
        it1->second += it2->second;
        vertexQuadrics_.erase(it2);
    }

    // Collect neighboring vertices for cost updates
    std::set<Mesh::VertexHandle> neighbors;
    for (auto vv_it = mesh.vv_iter(v1); vv_it.is_valid(); ++vv_it)
    {
        neighbors.insert(*vv_it);
    }
    for (auto vv_it = mesh.vv_iter(v2); vv_it.is_valid(); ++vv_it)
    {
        neighbors.insert(*vv_it);
    }

    // Remove collapsed edge from valid edges
    validEdges_.erase(collapse.edge);

    // Remove edges incident to v2 from valid edges
    for (auto ve_it = mesh.ve_iter(v2); ve_it.is_valid(); ++ve_it)
    {
        validEdges_.erase(*ve_it);
    }

    // Perform the actual collapse
    if (!mesh.is_collapse_ok(heh))
    {
        return false;
    }

    mesh.collapse(heh);

    // Update costs for neighboring edges
    updateEdgeCosts(mesh, v1);

    return true;
}

void MeshDecimation::updateEdgeCosts(Mesh &mesh, Mesh::VertexHandle vertex)
{
    // Update costs for all edges incident to the vertex
    for (auto ve_it = mesh.ve_iter(vertex); ve_it.is_valid(); ++ve_it)
    {
        if (mesh.status(*ve_it).deleted())
            continue;

        Eigen::Vector3d optimalPos;
        double cost = computeEdgeCost(mesh, *ve_it, optimalPos);

        if (cost < std::numeric_limits<double>::infinity())
        {
            EdgeCollapse collapse;
            collapse.edge = *ve_it;
            collapse.cost = cost;
            collapse.optimalPosition = optimalPos;

            edgeQueue_.push(collapse);
            validEdges_.insert(*ve_it);
        }
    }
}
