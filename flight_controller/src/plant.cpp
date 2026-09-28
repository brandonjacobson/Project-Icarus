#include "plant.hpp"
#include <iostream>

Plant::Plant(const Params& p) : p_(p), mixer_(p) {

}

State Plant::f(const State& x, const Eigen::Vector4d& F) const {
    Eigen::Vector4d wrench = mixer_.wrench(F);
    Eigen::Matrix3d R = x.q_bw.toRotationMatrix();

    double T = wrench(0);
    Eigen::Vector3d tau = wrench.tail<3>();

    State dx{};
    dx.position = x.velocity;
    dx.velocity = (T / p_.m_kg) * R.col(2) - Eigen::Vector3d(0, 0, p_.g);
    Quaternion q;
    q.w = 0;
    q.x = x.omega(0); // p
    q.y = x.omega(1); // q
    q.z = x.omega(2); // r
    dx.q_bw = 0.5*(x.q_bw.multiply(q));
    dx.omega(0) = (tau(0) - x.omega(1)*x.omega(2)*(p_.J(2,2)-p_.J(1,1))) * 1 / p_.J(0,0);
    dx.omega(1) = (tau(1) - x.omega(0)*x.omega(2)*(p_.J(0,0)-p_.J(2,2))) * 1 / p_.J(1,1);
    dx.omega(2) = (tau(2) - x.omega(0)*x.omega(1)*(p_.J(1,1)-p_.J(0,0))) * 1 / p_.J(2,2);
    return dx;
}

State helper_step(const State& a, double s, const State& b) {
    State result;
    result.position = a.position + s*b.position;
    result.velocity = a.velocity + s*b.velocity;
    result.q_bw = a.q_bw + s*b.q_bw;
    result.omega = a.omega + s*b.omega;
    return result;
}

State Plant::step(const State& x, const Eigen::Vector4d& F_motors_N, double dt) const {
    State k1 = f(x, F_motors_N);
    State k2 = f(helper_step(x, dt/2, k1), F_motors_N);
    State k3 = f(helper_step(x, dt/2, k2), F_motors_N);
    State k4 = f(helper_step(x, dt, k3), F_motors_N);
    State x_new = helper_step(x, dt/6, helper_step(helper_step(helper_step(k4, 2, k3), 2, k2), 1, k1));
    x_new.q_bw.normalize();
    return x_new;
}