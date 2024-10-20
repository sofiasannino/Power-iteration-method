#include "power_iteration.h"

namespace eigenvalue {

    using linear_algebra::operator*;
    using linear_algebra::operator-;

    double power_iteration::solve(const linear_algebra::square_matrix& A, const std::vector<double>& x0) const
    {
        //
        std::vector<double> xk=x0;
        double vk=linear_algebra::scalar(xk, A*xk);
        double residual=0;
        double increment=0;
        std::size_t t=0;
        bool conv=false;
        //
        while(!conv && t<max_it) {
            xk=A*xk; //z(k+1)
            linear_algebra::normalize(xk); //x(k+1)
            double vk1=linear_algebra::scalar(xk, A*xk); //v(k+1)
            residual=linear_algebra::norm(A*xk-vk1*xk);  //residual at k+1
            increment=std::abs(vk-vk1)/std::abs(vk); //increment at k+1
            conv=eigenvalue::power_iteration::converged(residual, increment); //check convergence as defined in (5) in the assignment
            t++;
            vk=vk1;

        }
      return vk;
    }

    bool power_iteration::converged(const double& residual, const double& increment) const
    {
        bool conv;

        switch(termination) {
            case(RESIDUAL):
                conv = residual < tolerance;
                break;
            case(INCREMENT):
                conv = increment < tolerance;
                break;
            case(BOTH):
                conv = residual < tolerance && increment < tolerance;
                break;
            default:
                conv = false;
        }
        return conv;
    }

} // eigenvalue