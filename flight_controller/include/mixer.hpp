#pragma once
#include <Eigen/Dense>
#include <cmath>
#include "types.hpp"

struct Mixer {
    Eigen::Matrix4d M;
    Eigen::Matrix4d Minv;

    explicit Mixer(const Params& p){
        double arm_xy = p.arm_m / std::sqrt(2);
        M << 1, 1, 1, 1,
            -arm_xy, arm_xy, arm_xy, -arm_xy,
            -arm_xy, -arm_xy, arm_xy, arm_xy,
            p.c_yaw_m, -p.c_yaw_m, p.c_yaw_m, -p.c_yaw_m;
        Minv = M.inverse();

    }
    
    Eigen::Vector4d wrench(const Eigen::Vector4d& F) const {
        Eigen::Vector4d wrench = M*F;
        return wrench;
    }

    Eigen::Vector4d thrusts(const Eigen::Vector4d& w) const {
        Eigen::Vector4d thrusts = Minv*w;
        return thrusts;
    }
};