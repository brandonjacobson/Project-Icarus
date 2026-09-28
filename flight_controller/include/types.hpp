#pragma once
#include <Eigen/Dense>
#include <cmath>

struct Quaternion {
    double w, x, y, z;
    
    Quaternion multiply(const Quaternion& other) const {
        Quaternion result;
        result.w = w * other.w - x*other.x - y*other.y - z*other.z;
        result.x = w * other.x + x*other.w + y*other.z - z*other.y;
        result.y = w * other.y - x*other.z + y*other.w + z*other.x;
        result.z = w * other.z + x*other.y - y*other.x + z*other.w;
        return result;
    }

    void normalize() {
        double norm = std::sqrt(w*w + x*x + y*y + z*z);
        w /= norm;
        x /= norm;
        y /= norm;
        z /= norm;
    }

    Eigen::Matrix3d toRotationMatrix() const {
        Eigen::Matrix3d R;
        R(0,0) = 1-2*(y*y + z*z);
        R(0,1) = 2*(x*y - w*z);
        R(0,2) = 2*(x*z + w*y);
        R(1,0) = 2*(x*y + w*z);
        R(1,1) = 1-2*(x*x +z*z);
        R(1,2) = 2*(y*z - w*x);
        R(2,0) = 2*(x*z-w*y);
        R(2,1) = 2*(y*z+w*x);
        R(2,2) = 1-2*(x*x + y*y);
        return R;
    }

    Quaternion operator+(const Quaternion& other) const {
        return Quaternion{w + other.w, x + other.x, y + other.y, z+other.z};
    }    
};

inline Quaternion operator*(double n, const Quaternion& q){
    return Quaternion{q.w*n, q.x*n, q.y*n, q.z*n};
};

struct State {
    Eigen::Vector3d position;
    Eigen::Vector3d velocity; 
    Quaternion q_bw; // quaternion from body to world frame
    Eigen::Vector3d omega; // p q r
};

struct Params {
    double m_kg = 0.91;
    
    Eigen::Matrix3d J = Eigen::Vector3d(0.01, 0.01, 0.02).asDiagonal();

    double g = 9.8;
    double arm_m = 0.165;
    double c_yaw_m = 0.016;
};