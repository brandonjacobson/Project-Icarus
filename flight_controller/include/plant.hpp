#pragma once
#include "types.hpp"
#include "mixer.hpp"

class Plant {
    public:
        explicit Plant(const Params& p);
        State f(const State& x, const Eigen::Vector4d& F_motors_N) const;
        State step(const State& x, const Eigen::Vector4d& F_motors_N, double dt) const;
    private:
        Params p_;
        Mixer mixer_;
};