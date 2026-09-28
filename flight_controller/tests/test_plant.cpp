#include <iostream>
#include "plant.hpp"

int main(){
    Params p;
    Plant plant(p);
    Quaternion q;
    q.w = 1;
    q.x = 0;
    q.y = 0;
    q.z = 0;
    Eigen::Vector3d omega(0, 0, 1);
    State x{Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), q, omega};
    Eigen::Vector4d F(0, 0, 0, 0);
    double dt = 0.001;
    State dx = plant.f(x, F);
    for (int i = 0; i < 1571; i++){
        x = plant.step(x, F, dt);
    }
    std::cout << "\n" << "x" << "\n";
    std::cout << x.position.transpose() << "\n";
    std::cout << x.velocity.transpose() << "\n";
    std::cout << "Quaterion: " << x.q_bw.w << " " << x.q_bw.x << " " << x.q_bw.y << " " << x.q_bw.z << "\n";
}