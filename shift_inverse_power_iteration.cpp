#include "shift_inverse_power_iteration.h"

namespace eigenvalue {

    using linear_algebra::operator*;
    using linear_algebra::operator-;

    double shift_inverse_power_iteration::solve(const linear_algebra::square_matrix& A, const double& mu, const std::vector<double>& x0) const
    {
        linear_algebra::square_matrix B=A;
        //compute A-mu*eye(n,n)
        for(std::size_t i=0; i<A.size(); i++) {
            B(i,i)=A(i,i)-mu;
        }
        const eigenvalue::inverse_power_iteration pi(10000, 1e-6, BOTH); //I had to initialize pi in this way because I cannot
        //have access to the private members of inverse_power_iteration class if I don't modify code outside these functions
        const double lambda1=pi.solve(B, x0);
        return mu+lambda1; //eigenvalue of matrix A closest to mu
    }

} // eigenvalue