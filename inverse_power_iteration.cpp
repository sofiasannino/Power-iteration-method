#include "inverse_power_iteration.h"

#include "power_iteration.h"

namespace eigenvalue {

    using linear_algebra::operator*;
    using linear_algebra::operator-;

    double inverse_power_iteration::solve(const linear_algebra::square_matrix& A, const std::vector<double>& x0) const
    {
        //initializing L and U for the LU factorization
        linear_algebra::square_matrix L;
        linear_algebra::square_matrix U;
        linear_algebra::lu(A,  L,  U);

        //calculating A^(-1)
        linear_algebra::square_matrix A1=A;
        for(std::size_t i=0; i<A.size(); i++) {
            //creating i column of the identity matrix
            std::vector<double> ei(A.size(), 0.0);
            ei[i]=1;

            std::vector<double> yi=linear_algebra::forwardsolve(L, ei ); //L*yi=ei;
            std::vector<double> xi = linear_algebra::backsolve(U, yi); //U*xi=yi;

            for(std::size_t k=0; k<A.size(); k++) {
                A1(k, i)=xi[k];
            }
        }
        const eigenvalue::power_iteration pi(max_it, tolerance,termination);
        const double lambda1=pi.solve(A1, x0);
        return 1.0/lambda1;
    }
    //is the following function useless?
    bool inverse_power_iteration::converged(const double& residual, const double& increment) const
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