#ifndef AMREX_MLEBABECCECLAP_3D_K_H_
#define AMREX_MLEBABECCECLAP_3D_K_H_
#include <AMReX_Config.H>

#include <AMReX_EB_LeastSquares_3D_K.H>

namespace amrex {

// constexpr Real eps = 1.0e-6;
// 
// inline Real getFaceStateUpwind ( Real const &edgeVel, Real const &leftState, Real const &rightState ) {
//    Real edstate = 0.0;
//    if ( edgeVel > eps ) { 
//       edstate = leftState;
//    }   
//    if ( edgeVel < -eps ) { 
//       edstate = rightState;
//    }   
//    if ( fabs(edgeVel) <= eps ) { 
//       edstate =  0.5 * ( leftState + rightState );
//    }   
//    return edstate;
// }
// 
// inline Real getCellUpwindCoeff ( Real const &leftedgeVel, Real const &rightedgeVel ) {
// 
//    Real coeff = 0.0;
//    if ( leftedgeVel*rightedgeVel < 0.0 ) { // Case where velocity change sign, use centered + art. visc
//       Real maxvel = std::max(fabs(leftedgeVel),fabs(rightedgeVel));
//          coeff += 0.5*(rightedgeVel-leftedgeVel); // centered part
//          coeff += maxvel; // art. visc.
//    } else { // same sign velocity, use upwind
//       if ( leftedgeVel < -eps ) 
//          coeff -= leftedgeVel;
//       else if ( fabs(leftedgeVel) <= eps )
//          coeff -= 0.5*leftedgeVel;
//       if ( rightedgeVel > eps )
//          coeff += rightedgeVel;
//       else if ( fabs(rightedgeVel) <= eps )
//          coeff += 0.5*rightedgeVel;
//    }
// 
//    return coeff;
// }
// 
// inline Real getLeftCellUpwindCoeff ( Real const &leftedgeVel, Real const &rightedgeVel ) {
// 
//    Real coeff = 0.0;
//    if ( leftedgeVel*rightedgeVel < 0.0 ) {
//       Real maxvel = std::max(fabs(leftedgeVel),fabs(rightedgeVel));
//       coeff -= 0.5*leftedgeVel;
//       coeff -= 0.5*maxvel;
//    } else {
//       if ( leftedgeVel > eps ) 
//          coeff -= leftedgeVel;
//       else if ( fabs(leftedgeVel) < eps )
//          coeff -= 0.5*leftedgeVel;
//    }
//    return coeff;
// }
// 
// inline Real getRightCellUpwindCoeff ( Real const &leftedgeVel, Real const &rightedgeVel ) {
// 
//    Real coeff = 0.0;
//    if ( leftedgeVel*rightedgeVel < 0.0 ) {
//       Real maxvel = std::max(fabs(leftedgeVel),fabs(rightedgeVel));
//       coeff += 0.5*rightedgeVel;
//       coeff += 0.5*maxvel;
//    } else {
//       if ( rightedgeVel < -eps )
//          coeff += rightedgeVel;
//       else if ( fabs(rightedgeVel) < eps )
//          coeff += 0.5*rightedgeVel;
//    }
//    return coeff;
// }
// 
// inline Real getCellUpwindCoeff ( Real const &leftedgeVel, Real const &rightedgeVel,
//                                  Real const &relaxleft  , Real const &relaxright ) {
// 
//    Real coeff = 0.0;
//    if ( leftedgeVel*relaxleft*rightedgeVel*relaxright < 0.0 ) { // Case where velocity change sign, use centered + art. visc
//       Real maxvel = std::max(fabs(leftedgeVel*relaxleft),fabs(rightedgeVel*relaxright));
//          coeff += 0.5*(rightedgeVel*relaxright-leftedgeVel*relaxleft); // centered part
//          coeff += maxvel; // art. visc.
//    } else { // same sign velocity, use upwind
//       if ( leftedgeVel < -eps ) 
//          coeff -= leftedgeVel*relaxleft;
//       else if ( fabs(leftedgeVel) < eps )
//          coeff -= 0.5*leftedgeVel*relaxleft;
//       if ( rightedgeVel > eps )
//          coeff += rightedgeVel*relaxright;
//       else if ( fabs(rightedgeVel) < eps )
//          coeff += 0.5*rightedgeVel*relaxright;
//    }
// 
//    return coeff;
// }
// 
// inline Real getFaceUpwindFlux ( Real const &edgeVel, Real const &leftState, Real const &rightState ) {
// 
//    Real Flux = 0.0;
//    if ( edgeVel > eps )
//       Flux = edgeVel * leftState;
//    else if ( edgeVel < -eps )
//       Flux = edgeVel * rightState;
//    else
//       Flux = edgeVel * 0.5 * ( leftState + rightState );
//    return Flux;
// }

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void mlebabecceclap_adotx_centroid (Box const& box, Array4<Real> const& y,
                        Array4<Real const> const& x, Array4<Real const> const& a,
                        Array4<Real const> const& bX, Array4<Real const> const& bY,
                        Array4<Real const> const& bZ,
                        Array4<Real const> const& cX, Array4<Real const> const& cY,
                        Array4<Real const> const& cZ,
                        Array4<EBCellFlag const> const& flag,
                        Array4<Real const> const& vfrc, Array4<Real const> const& apx,
                        Array4<Real const> const& apy, Array4<Real const> const& apz,
                        Array4<Real const> const& fcx, Array4<Real const> const& fcy,
                        Array4<Real const> const& fcz,
                        Array4<Real const> const& ccent, Array4<Real const> const& ba,
                        Array4<Real const> const& bcent, Array4<Real const> const& beb,
                        Array4<Real const> const& phieb,
                        const int& domlo_x, const int& domlo_y, const int& domlo_z,
                        const int& domhi_x, const int& domhi_y, const int& domhi_z,
                        const bool& on_x_face, const bool& on_y_face, const bool& on_z_face,
                        bool is_eb_dirichlet, bool is_eb_inhomog,
                        GpuArray<Real,AMREX_SPACEDIM> const& dxinv,
                        Real alpha, Real beta, Real eta, int ncomp) noexcept
{
    Real dhx = beta*dxinv[0]*dxinv[0];
    Real dhy = beta*dxinv[1]*dxinv[1];
    Real dhz = beta*dxinv[2]*dxinv[2];
    Real dhxc = eta*dxinv[0];
    Real dhyc = eta*dxinv[1];
    Real dhzc = eta*dxinv[2];

    amrex::Abort("phi_on_centroid is still a WIP");

    amrex::Loop(box, ncomp, [=] (int i, int j, int k, int n) noexcept
    {
        if (flag(i,j,k).isCovered())
        {
            y(i,j,k,n) = 0.0;
        }
        else if (flag(i,j,k).isRegular() &&
                 ((flag(i-1,j  ,k  ).isRegular() && flag(i+1,j  ,k  ).isRegular() &&
                  flag(i  ,j-1,k  ).isRegular() && flag(i  ,j+1,k  ).isRegular() &&
                  flag(i  ,j  ,k-1).isRegular() && flag(i  ,j  ,k+1).isRegular()) ))
        {
            y(i,j,k,n) = alpha*a(i,j,k)*x(i,j,k,n)
                - dhx  * (bX(i+1,j,k,n)*(x(i+1,j,k,n) - x(i  ,j,k,n))
                         -bX(i  ,j,k,n)*(x(i  ,j,k,n) - x(i-1,j,k,n)))
                - dhy  * (bY(i,j+1,k,n)*(x(i,j+1,k,n) - x(i,j  ,k,n))
                         -bY(i,j  ,k,n)*(x(i,j  ,k,n) - x(i,j-1,k,n)))
                - dhz  * (bZ(i,j,k+1,n)*(x(i,j,k+1,n) - x(i,j,k  ,n))
                         -bZ(i,j,k  ,n)*(x(i,j,k  ,n) - x(i,j,k-1,n)))
                - dhxc * (cX(i+1,j,k,n)*getFaceStateUpwind(cX(i+1,j,k,n), x(i  ,j,k,n), x(i+1,j,k,n))
                        - cX(i  ,j,k,n)*getFaceStateUpwind(cX(i  ,j,k,n), x(i-1,j,k,n), x(i  ,j,k,n)))
                - dhyc * (cY(i,j+1,k,n)*getFaceStateUpwind(cY(i,j+1,k,n), x(i,j  ,k,n), x(i,j+1,k,n))
                        - cY(i,j  ,k,n)*getFaceStateUpwind(cY(i,j  ,k,n), x(i,j-1,k,n), x(i,j  ,k,n)))
                - dhzc * (cZ(i,j,k+1,n)*getFaceStateUpwind(cZ(i,j,k+1,n), x(i,j,k  ,n), x(i,j,k+1,n))
                        - cZ(i,j,k  ,n)*getFaceStateUpwind(cZ(i,j,k  ,n), x(i,j,k-1,n), x(i,j,k  ,n)));
        }
        else
        {
            Real kappa = vfrc(i,j,k);
            Real apxm = apx(i,j,k);
            Real apxp = apx(i+1,j,k);
            Real apym = apy(i,j,k);
            Real apyp = apy(i,j+1,k);
            Real apzm = apz(i,j,k);
            Real apzp = apz(i,j,k+1);

            // First get EB-aware slope that doesn't know about extdir
            bool needs_bdry_stencil = (i <= domlo_x) || (i >= domhi_x) ||
                                      (j <= domlo_y) || (j >= domhi_y) ||
                                      (k <= domlo_z) || (k >= domhi_z);

            Real fxm = bX(i,j,k,n)*(x(i,j,k,n) - x(i-1,j,k,n));
            Real fxmc = cX(i  ,j,k,n)*getFaceStateUpwind(cX(i  ,j,k,n), x(i-1,j,k,n), x(i  ,j,k,n));
            if ( (apxm != 0.0) && (vfrc(i,j,k) != 1.0 || vfrc(i-1,j,k) != 1.0 || vfrc(i+1,j,k) != 1.0) ) {
                Real yloc_on_xface = fcx(i,j,k,0);
                Real zloc_on_xface = fcx(i,j,k,1);

                if(needs_bdry_stencil) {
                  fxm  = grad_x_of_phi_on_centroids_extdir(i,j,k,n,x,phieb,flag,ccent,bcent,vfrc,
                                                          yloc_on_xface,zloc_on_xface,
                                                          is_eb_dirichlet,is_eb_inhomog,
                                                          on_x_face,domlo_x,domhi_x,
                                                          on_y_face,domlo_y,domhi_y,
                                                          on_z_face,domlo_z,domhi_z);
                } else {
                  fxm = grad_x_of_phi_on_centroids(i,j,k,n,x,phieb,flag,ccent,bcent,
                                                 yloc_on_xface,zloc_on_xface,is_eb_dirichlet,is_eb_inhomog);
                }

                fxm  *= bX(i,j,k,n);
            }

            Real fxp  = bX(i+1,j,k,n)*(x(i+1,j,k,n) - x(i,j,k,n));
            Real fxpc = cX(i+1,j,k,n)*getFaceStateUpwind(cX(i+1,j,k,n), x(i  ,j,k,n), x(i+1,j,k,n));
            if ( (apxp != 0.0) && (vfrc(i,j,k) != 1.0 || vfrc(i+1,j,k) != 1.0 || vfrc(i-1,j,k) != 1.0) ) {
                Real yloc_on_xface = fcx(i+1,j,k,0);
                Real zloc_on_xface = fcx(i+1,j,k,1);

                if(needs_bdry_stencil) {
                  fxp = grad_x_of_phi_on_centroids_extdir(i+1,j,k,n,x,phieb,flag,ccent,bcent,vfrc,
                                                          yloc_on_xface,zloc_on_xface,
                                                          is_eb_dirichlet,is_eb_inhomog,
                                                          on_x_face,domlo_x,domhi_x,
                                                          on_y_face,domlo_y,domhi_y,
                                                          on_z_face,domlo_z,domhi_z);
                } else {
                  fxp = grad_x_of_phi_on_centroids(i+1,j,k,n,x,phieb,flag,ccent,bcent,
                                                 yloc_on_xface,zloc_on_xface,is_eb_dirichlet,is_eb_inhomog);
                }
                
                fxp  *= bX(i+1,j,k,n);
            }

            Real fym = bY(i,j,k,n)*(x(i,j,k,n) - x(i,j-1,k,n));
            Real fymc = cY(i,j  ,k,n)*getFaceStateUpwind(cY(i,j  ,k,n), x(i,j-1,k,n), x(i,j  ,k,n));
            if ( (apym != 0.0) && (vfrc(i,j,k) != 1.0 || vfrc(i,j-1,k) != 1.0 || vfrc(i,j+1,k) != 1.0) ) {
                Real xloc_on_yface = fcy(i,j,k,0);
                Real zloc_on_yface = fcy(i,j,k,1);

                if(needs_bdry_stencil) {
                  fym = grad_y_of_phi_on_centroids_extdir(i,j,k,n,x,phieb,flag,ccent,bcent,vfrc,
                                                          xloc_on_yface,zloc_on_yface,
                                                          is_eb_dirichlet,is_eb_inhomog,
                                                          on_x_face,domlo_x,domhi_x,
                                                          on_y_face,domlo_y,domhi_y,
                                                          on_z_face,domlo_z,domhi_z);
                } else {
                  fym = grad_y_of_phi_on_centroids(i,j,k,n,x,phieb,flag,ccent,bcent,
                                                 xloc_on_yface,zloc_on_yface,is_eb_dirichlet,is_eb_inhomog);
                }
  
                fym  *= bY(i,j,k,n);
            }

            Real fyp = bY(i,j+1,k,n)*(x(i,j+1,k,n) - x(i,j,k,n));
            Real fypc = cY(i,j+1,k,n)*getFaceStateUpwind(cY(i,j+1,k,n), x(i,j  ,k,n), x(i,j+1,k,n));
            if ( (apyp != 0.0) && (vfrc(i,j,k) != 1.0 || vfrc(i,j+1,k) != 1.0 || vfrc(i,j-1,k) != 1.0) ) {
                Real xloc_on_yface = fcy(i,j+1,k,0);
                Real zloc_on_yface = fcy(i,j+1,k,1);

                if(needs_bdry_stencil) {
                  fyp = grad_y_of_phi_on_centroids_extdir(i,j+1,k,n,x,phieb,flag,ccent,bcent,vfrc,
                                                          xloc_on_yface,zloc_on_yface,
                                                          is_eb_dirichlet,is_eb_inhomog,
                                                          on_x_face,domlo_x,domhi_x,
                                                          on_y_face,domlo_y,domhi_y,
                                                          on_z_face,domlo_z,domhi_z);
                } else {
                  fyp = grad_y_of_phi_on_centroids(i,j+1,k,n,x,phieb,flag,ccent,bcent,
                                                 xloc_on_yface,zloc_on_yface,is_eb_dirichlet,is_eb_inhomog);
                }

                fyp  *= bY(i,j+1,k,n);
            }

            Real fzm = bZ(i,j,k,n)*(x(i,j,k,n) - x(i,j,k-1,n));
            Real fzmc = cZ(i,j,k  ,n)*getFaceStateUpwind(cZ(i,j,k  ,n), x(i,j,k-1,n), x(i,j,k  ,n));
            if ( (apzm != 0.0) && (vfrc(i,j,k) != 1.0 || vfrc(i,j,k-1) != 1.0 || vfrc(i,j,k+1) != 1.0) ) {
                Real xloc_on_zface = fcz(i,j,k,0);
                Real yloc_on_zface = fcz(i,j,k,1);

                if(needs_bdry_stencil) {
                  fzm = grad_z_of_phi_on_centroids_extdir(i,j,k,n,x,phieb,flag,ccent,bcent,vfrc,
                                                          xloc_on_zface,yloc_on_zface,
                                                          is_eb_dirichlet,is_eb_inhomog,
                                                          on_x_face,domlo_x,domhi_x,
                                                          on_y_face,domlo_y,domhi_y,
                                                          on_z_face,domlo_z,domhi_z);
                } else {
                  fzm = grad_z_of_phi_on_centroids(i,j,k,n,x,phieb,flag,ccent,bcent,
                                                 xloc_on_zface,yloc_on_zface,is_eb_dirichlet,is_eb_inhomog);
                }

                fzm  *= bZ(i,j,k,n);
            }

            Real fzp = bZ(i,j,k+1,n)*(x(i,j,k+1,n) - x(i,j,k,n));
            Real fzpc = cZ(i,j,k+1,n)*getFaceStateUpwind(cZ(i,j,k+1,n), x(i,j,k  ,n), x(i,j,k+1,n));
            if ( (apzp != 0.0) && (vfrc(i,j,k) != 1.0 || vfrc(i,j,k+1) != 1.0  || vfrc(i,j,k-1) != 1.0) ) {
                Real xloc_on_zface = fcz(i,j,k+1,0);
                Real yloc_on_zface = fcz(i,j,k+1,1);

                if(needs_bdry_stencil) {
                  fzp = grad_z_of_phi_on_centroids_extdir(i,j,k+1,n,x,phieb,flag,ccent,bcent,vfrc,
                                                          xloc_on_zface,yloc_on_zface,
                                                          is_eb_dirichlet,is_eb_inhomog,
                                                          on_x_face,domlo_x,domhi_x,
                                                          on_y_face,domlo_y,domhi_y,
                                                          on_z_face,domlo_z,domhi_z);
                } else {
                  fzp = grad_z_of_phi_on_centroids(i,j,k+1,n,x,phieb,flag,ccent,bcent,
                                                 xloc_on_zface,yloc_on_zface,is_eb_dirichlet,is_eb_inhomog);
                }

                fzp  *= bZ(i,j,k+1,n);
            }

            Real feb = 0.0;
            if (is_eb_dirichlet && flag(i,j,k).isSingleValued()) {
                Real dapx = apxm-apxp;
                Real dapy = apym-apyp;
                Real dapz = apzm-apzp;
                Real anorm = std::sqrt(dapx*dapx+dapy*dapy+dapz*dapz);
                Real anorminv = 1.0/anorm;
                Real anrmx = dapx * anorminv;
                Real anrmy = dapy * anorminv;
                Real anrmz = dapz * anorminv;

                feb = grad_eb_of_phi_on_centroids_extdir(i,j,k,n,x,phieb,flag,ccent,bcent,vfrc,
                                                         anrmx,anrmy,anrmz,is_eb_inhomog,
                                                         on_x_face,domlo_x,domhi_x,
                                                         on_y_face,domlo_y,domhi_y,
                                                         on_z_face,domlo_z,domhi_z);
                feb *= ba(i,j,k) * beb(i,j,k,n);
            }

            y(i,j,k,n) = alpha*a(i,j,k)*x(i,j,k,n) + (1.0/kappa) *
                (dhx*(apxm*fxm - apxp*fxp) +
                 dhy*(apym*fym - apyp*fyp) +
                 dhz*(apzm*fzm - apzp*fzp) +
                 dhxc*(apxm*fxmc - apxp*fxpc) +
                 dhyc*(apym*fymc - apyp*fypc) +
                 dhzc*(apzm*fzmc - apzp*fzpc) - dhx*feb);
        }
    });
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void mlebabecceclap_adotx (Box const& box, Array4<Real> const& y,
                        Array4<Real const> const& x, Array4<Real const> const& a,
                        Array4<Real const> const& bX, Array4<Real const> const& bY,
                        Array4<Real const> const& bZ, Array4<Real const> const& cX,
                        Array4<Real const> const& cY, Array4<Real const> const& cZ,
                        Array4<const int> const& ccm,
                        Array4<EBCellFlag const> const& flag,
                        Array4<Real const> const& vfrc, Array4<Real const> const& apx,
                        Array4<Real const> const& apy, Array4<Real const> const& apz,
                        Array4<Real const> const& fcx, Array4<Real const> const& fcy,
                        Array4<Real const> const& fcz, Array4<Real const> const& ba,
                        Array4<Real const> const& bc, Array4<Real const> const& beb,
                        bool is_dirichlet, Array4<Real const> const& phieb,
                        bool is_inhomog, GpuArray<Real,AMREX_SPACEDIM> const& dxinv,
                        Real alpha, Real beta, Real eta, int ncomp,
                        bool beta_on_centroid, bool phi_on_centroid) noexcept
{
    Real dhx = beta*dxinv[0]*dxinv[0];
    Real dhy = beta*dxinv[1]*dxinv[1];
    Real dhz = beta*dxinv[2]*dxinv[2];
    Real dhxc = eta*dxinv[0];
    Real dhyc = eta*dxinv[1];
    Real dhzc = eta*dxinv[2];
  
    bool beta_on_center = !(beta_on_centroid);
    bool  phi_on_center = !( phi_on_centroid);

    amrex::Loop(box, ncomp, [=] (int i, int j, int k, int n) noexcept
    {
        if (flag(i,j,k).isCovered())
        {
            y(i,j,k,n) = 0.0;
        }
        else if (flag(i,j,k).isRegular())
        {
            y(i,j,k,n) = alpha*a(i,j,k)*x(i,j,k,n)
                - dhx  * (bX(i+1,j,k,n)*(x(i+1,j,k,n) - x(i  ,j,k,n))
                         -bX(i  ,j,k,n)*(x(i  ,j,k,n) - x(i-1,j,k,n)))
                - dhy  * (bY(i,j+1,k,n)*(x(i,j+1,k,n) - x(i,j  ,k,n))
                         -bY(i,j  ,k,n)*(x(i,j  ,k,n) - x(i,j-1,k,n)))
                - dhz  * (bZ(i,j,k+1,n)*(x(i,j,k+1,n) - x(i,j,k  ,n))
                         -bZ(i,j,k  ,n)*(x(i,j,k  ,n) - x(i,j,k-1,n)))
                - dhxc * (cX(i+1,j,k,n)*getFaceStateUpwind(cX(i+1,j,k,n), x(i  ,j,k,n), x(i+1,j,k,n))
                        - cX(i  ,j,k,n)*getFaceStateUpwind(cX(i  ,j,k,n), x(i-1,j,k,n), x(i  ,j,k,n)))
                - dhyc * (cY(i,j+1,k,n)*getFaceStateUpwind(cY(i,j+1,k,n), x(i,j  ,k,n), x(i,j+1,k,n))
                        - cY(i,j  ,k,n)*getFaceStateUpwind(cY(i,j  ,k,n), x(i,j-1,k,n), x(i,j  ,k,n)))
                - dhzc * (cZ(i,j,k+1,n)*getFaceStateUpwind(cZ(i,j,k+1,n), x(i,j,k  ,n), x(i,j,k+1,n))
                        - cZ(i,j,k  ,n)*getFaceStateUpwind(cZ(i,j,k  ,n), x(i,j,k-1,n), x(i,j,k  ,n)));
        }
        else
        {
            Real kappa = vfrc(i,j,k);
            Real apxm = apx(i,j,k);
            Real apxp = apx(i+1,j,k);
            Real apym = apy(i,j,k);
            Real apyp = apy(i,j+1,k);
            Real apzm = apz(i,j,k);
            Real apzp = apz(i,j,k+1);

            Real fxm = bX(i,j,k,n)*(x(i,j,k,n) - x(i-1,j,k,n));
            Real fxmc = cX(i  ,j,k,n)*getFaceStateUpwind(cX(i  ,j,k,n), x(i-1,j,k,n), x(i  ,j,k,n));
            if (apxm != 0.0 && apxm != 1.0) {
                int jj = j + static_cast<int>(amrex::Math::copysign(1.0, fcx(i,j,k,0)));
                int kk = k + static_cast<int>(amrex::Math::copysign(1.0, fcx(i,j,k,1)));
                Real fracy = (ccm(i-1,jj,k) || ccm(i,jj,k)) ? amrex::Math::abs(fcx(i,j,k,0)) : 0.0;
                Real fracz = (ccm(i-1,j,kk) || ccm(i,j,kk)) ? amrex::Math::abs(fcx(i,j,k,1)) : 0.0;
                if (beta_on_center && phi_on_center)
                {
                    fxm = (1.0-fracy)*(1.0-fracz)*fxm +
                        fracy*(1.0-fracz)*bX(i,jj,k ,n)*(x(i,jj,k ,n)-x(i-1,jj,k ,n)) +
                        fracz*(1.0-fracy)*bX(i,j ,kk,n)*(x(i,j ,kk,n)-x(i-1,j ,kk,n)) +
                        fracy*     fracz *bX(i,jj,kk,n)*(x(i,jj,kk,n)-x(i-1,jj,kk,n));

                    fxmc = (1.0-fracy)*(1.0-fracz)*fxmc +
                         fracy*(1.0-fracz)*cX(i,jj,k ,n)*getFaceStateUpwind(cX(i,jj,k ,n),x(i-1,jj,k ,n),x(i,jj,k ,n)) +
                         fracz*(1.0-fracy)*cX(i,j ,kk,n)*getFaceStateUpwind(cX(i,j ,kk,n),x(i-1,j ,kk,n),x(i,j ,kk,n)) +
                         fracy*     fracz *cX(i,jj,kk,n)*getFaceStateUpwind(cX(i,jj,kk,n),x(i-1,jj,kk,n),x(i,jj,kk,n));
                }
                else if (beta_on_centroid && phi_on_center)
                {
                    fxm = (1.0-fracy)*(1.0-fracz)*(x(i, j, k,n)-x(i-1, j, k,n)) +
                               fracy *(1.0-fracz)*(x(i,jj, k,n)-x(i-1,jj, k,n)) +
                               fracz *(1.0-fracy)*(x(i, j,kk,n)-x(i-1, j,kk,n)) +
                               fracy *     fracz *(x(i,jj,kk,n)-x(i-1,jj,kk,n));

                    fxmc = (1.0-fracy)*(1.0-fracz)*getFaceStateUpwind(cX(i, j, k,n),x(i-1, j, k,n),x(i, j, k,n)) +
                                fracy *(1.0-fracz)*getFaceStateUpwind(cX(i,jj, k,n),x(i-1,jj, k,n),x(i,jj, k,n)) +
                                fracz *(1.0-fracy)*getFaceStateUpwind(cX(i, j,kk,n),x(i-1, j,kk,n),x(i, j,kk,n)) +
                                fracy *     fracz *getFaceStateUpwind(cX(i,jj,kk,n),x(i-1,jj,kk,n),x(i,jj,kk,n));

                    fxm *= bX(i,j,k,n);
                    fxmc *= cX(i,j,k,n);
                }
            }

            Real fxp = bX(i+1,j,k,n)*(x(i+1,j,k,n) - x(i,j,k,n));
            Real fxpc = cX(i+1,j,k,n)*getFaceStateUpwind(cX(i+1,j,k,n), x(i  ,j,k,n), x(i+1,j,k,n));
            if (apxp != 0.0 && apxp != 1.0) {
                int jj = j + static_cast<int>(amrex::Math::copysign(1.0,fcx(i+1,j,k,0)));
                int kk = k + static_cast<int>(amrex::Math::copysign(1.0,fcx(i+1,j,k,1)));
                Real fracy = (ccm(i,jj,k) || ccm(i+1,jj,k)) ? amrex::Math::abs(fcx(i+1,j,k,0)) : 0.0;
                Real fracz = (ccm(i,j,kk) || ccm(i+1,j,kk)) ? amrex::Math::abs(fcx(i+1,j,k,1)) : 0.0;
                if (beta_on_center && phi_on_center)
                {
                    fxp = (1.0-fracy)*(1.0-fracz)*fxp +
                        fracy*(1.0-fracz)*bX(i+1,jj,k ,n)*(x(i+1,jj,k ,n)-x(i,jj,k ,n)) +
                        fracz*(1.0-fracy)*bX(i+1,j ,kk,n)*(x(i+1,j ,kk,n)-x(i,j ,kk,n)) +
                        fracy*     fracz *bX(i+1,jj,kk,n)*(x(i+1,jj,kk,n)-x(i,jj,kk,n));

                    fxpc = (1.0-fracy)*(1.0-fracz)*fxpc +
                         fracy*(1.0-fracz)*cX(i+1,jj,k ,n)*getFaceStateUpwind(cX(i+1,jj,k ,n),x(i,jj,k ,n),x(i+1,jj,k ,n)) +
                         fracz*(1.0-fracy)*cX(i+1,j ,kk,n)*getFaceStateUpwind(cX(i+1,j ,kk,n),x(i,j ,kk,n),x(i+1,j ,kk,n)) +
                         fracy*     fracz *cX(i+1,jj,kk,n)*getFaceStateUpwind(cX(i+1,jj,kk,n),x(i,jj,kk,n),x(i+1,jj,kk,n));
                }
                else if (beta_on_centroid && phi_on_center)
                {
                    fxp = (1.0-fracy)*(1.0-fracz)*(x(i+1, j, k,n)-x(i, j, k,n)) +
                               fracy *(1.0-fracz)*(x(i+1,jj, k,n)-x(i,jj, k,n)) +
                               fracz *(1.0-fracy)*(x(i+1, j,kk,n)-x(i, j,kk,n)) +
                               fracy *     fracz *(x(i+1,jj,kk,n)-x(i,jj,kk,n));

                    fxpc = (1.0-fracy)*(1.0-fracz)*getFaceStateUpwind(cX(i+1, j, k,n),x(i, j, k,n),x(i+1, j, k,n)) +
                                fracy *(1.0-fracz)*getFaceStateUpwind(cX(i+1,jj, k,n),x(i,jj, k,n),x(i+1,jj, k,n)) +
                                fracz *(1.0-fracy)*getFaceStateUpwind(cX(i+1, j,kk,n),x(i, j,kk,n),x(i+1, j,kk,n)) +
                                fracy *     fracz *getFaceStateUpwind(cX(i+1,jj,kk,n),x(i,jj,kk,n),x(i+1,jj,kk,n));

                    fxp *= bX(i+1,j,k,n);
                    fxpc *= cX(i+1,j,k,n);

                }
            }

            Real fym = bY(i,j,k,n)*(x(i,j,k,n) - x(i,j-1,k,n));
            Real fymc = cY(i,j  ,k,n)*getFaceStateUpwind(cY(i,j  ,k,n), x(i,j-1,k,n), x(i,j  ,k,n));
            if (apym != 0.0 && apym != 1.0) {
                int ii = i + static_cast<int>(amrex::Math::copysign(1.0,fcy(i,j,k,0)));
                int kk = k + static_cast<int>(amrex::Math::copysign(1.0,fcy(i,j,k,1)));
                Real fracx = (ccm(ii,j-1,k) || ccm(ii,j,k)) ? amrex::Math::abs(fcy(i,j,k,0)) : 0.0;
                Real fracz = (ccm(i,j-1,kk) || ccm(i,j,kk)) ? amrex::Math::abs(fcy(i,j,k,1)) : 0.0;
                if (beta_on_center && phi_on_center)
                {
                    fym = (1.0-fracx)*(1.0-fracz)*fym +
                        fracx*(1.0-fracz)*bY(ii,j,k ,n)*(x(ii,j,k ,n)-x(ii,j-1,k ,n)) +
                        fracz*(1.0-fracx)*bY(i ,j,kk,n)*(x(i ,j,kk,n)-x(i ,j-1,kk,n)) +
                        fracx*     fracz *bY(ii,j,kk,n)*(x(ii,j,kk,n)-x(ii,j-1,kk,n));

                    fymc = (1.0-fracx)*(1.0-fracz)*fymc +
                         fracx*(1.0-fracz)*cY(ii,j,k ,n)*getFaceStateUpwind(cY(ii,j,k ,n),x(ii,j-1,k ,n),x(ii,j,k ,n)) +
                         fracz*(1.0-fracx)*cY(i ,j,kk,n)*getFaceStateUpwind(cY(i ,j,kk,n),x(i ,j-1,kk,n),x(i ,j,kk,n)) +
                         fracx*     fracz *cY(ii,j,kk,n)*getFaceStateUpwind(cY(ii,j,kk,n),x(ii,j-1,kk,n),x(ii,j,kk,n));
                }
                else if (beta_on_centroid && phi_on_center)
                {
                    fym = (1.0-fracx)*(1.0-fracz)*(x( i,j, k,n)-x( i,j-1, k,n)) +
                               fracx *(1.0-fracz)*(x(ii,j, k,n)-x(ii,j-1, k,n)) +
                               fracz *(1.0-fracx)*(x(i ,j,kk,n)-x( i,j-1,kk,n)) +
                               fracx *     fracz *(x(ii,j,kk,n)-x(ii,j-1,kk,n));

                    fymc = (1.0-fracx)*(1.0-fracz)*getFaceStateUpwind(cY( i,j, k,n),x( i,j-1, k,n),x( i,j, k,n)) +
                                fracx *(1.0-fracz)*getFaceStateUpwind(cY(ii,j, k,n),x(ii,j-1, k,n),x(ii,j, k,n)) +
                                fracz *(1.0-fracx)*getFaceStateUpwind(cY( i,j,kk,n),x(i ,j-1,kk,n),x( i,j,kk,n)) +
                                fracx *     fracz *getFaceStateUpwind(cY(ii,j,kk,n),x(ii,j-1,kk,n),x(ii,j,kk,n));

                    fym *= bY(i,j,k,n);
                    fymc *= cY(i,j,k,n);
                }
            }

            Real fyp = bY(i,j+1,k,n)*(x(i,j+1,k,n) - x(i,j,k,n));
            Real fypc = cY(i,j+1,k,n)*getFaceStateUpwind(cY(i,j+1,k,n), x(i,j  ,k,n), x(i,j+1,k,n));
            if (apyp != 0.0 && apyp != 1.0) {
                int ii = i + static_cast<int>(amrex::Math::copysign(1.0,fcy(i,j+1,k,0)));
                int kk = k + static_cast<int>(amrex::Math::copysign(1.0,fcy(i,j+1,k,1)));
                Real fracx = (ccm(ii,j,k) || ccm(ii,j+1,k)) ? amrex::Math::abs(fcy(i,j+1,k,0)) : 0.0;
                Real fracz = (ccm(i,j,kk) || ccm(i,j+1,kk)) ? amrex::Math::abs(fcy(i,j+1,k,1)) : 0.0;
                if (beta_on_center && phi_on_center)
                {
                    fyp = (1.0-fracx)*(1.0-fracz)*fyp +
                        fracx*(1.0-fracz)*bY(ii,j+1,k ,n)*(x(ii,j+1,k ,n)-x(ii,j,k ,n)) +
                        fracz*(1.0-fracx)*bY(i ,j+1,kk,n)*(x(i ,j+1,kk,n)-x(i ,j,kk,n)) +
                        fracx*     fracz *bY(ii,j+1,kk,n)*(x(ii,j+1,kk,n)-x(ii,j,kk,n));

                    fypc = (1.0-fracx)*(1.0-fracz)*fypc +
                         fracx*(1.0-fracz)*cY(ii,j+1,k ,n)*getFaceStateUpwind(cY(ii,j+1,k ,n),x(ii,j,k ,n),x(ii,j+1,k ,n)) +
                         fracz*(1.0-fracx)*cY(i ,j+1,kk,n)*getFaceStateUpwind(cY(i ,j+1,kk,n),x(i ,j,kk,n),x(i ,j+1,kk,n)) +
                         fracx*     fracz *cY(ii,j+1,kk,n)*getFaceStateUpwind(cY(ii,j+1,kk,n),x(ii,j,kk,n),x(ii,j+1,kk,n));
                }
                else if (beta_on_centroid && phi_on_center)
                {
                    fyp = (1.0-fracx)*(1.0-fracz)*(x( i,j+1, k,n)-x( i,j, k,n)) +
                               fracx *(1.0-fracz)*(x(ii,j+1, k,n)-x(ii,j, k,n)) +
                               fracz *(1.0-fracx)*(x( i,j+1,kk,n)-x( i,j,kk,n)) +
                               fracx *     fracz *(x(ii,j+1,kk,n)-x(ii,j,kk,n));

                    fypc = (1.0-fracx)*(1.0-fracz)*getFaceStateUpwind(cY( i,j+1, k,n),x( i,j, k,n),x( i,j+1, k,n)) +
                                fracx *(1.0-fracz)*getFaceStateUpwind(cY(ii,j+1, k,n),x(ii,j, k,n),x(ii,j+1, k,n)) +
                                fracz *(1.0-fracx)*getFaceStateUpwind(cY( i,j+1,kk,n),x(i ,j,kk,n),x( i,j+1,kk,n)) +
                                fracx *     fracz *getFaceStateUpwind(cY(ii,j+1,kk,n),x(ii,j,kk,n),x(ii,j+1,kk,n));

                    fyp *= bY(i,j+1,k,n);
                    fypc *= cY(i,j+1,k,n);
                }
            }

            Real fzm = bZ(i,j,k,n)*(x(i,j,k,n) - x(i,j,k-1,n));
            Real fzmc = cZ(i,j,k  ,n)*getFaceStateUpwind(cZ(i,j,k  ,n), x(i,j,k-1,n), x(i,j,k  ,n));
            if (apzm != 0.0 && apzm != 1.0) {
                int ii = i + static_cast<int>(amrex::Math::copysign(1.0,fcz(i,j,k,0)));
                int jj = j + static_cast<int>(amrex::Math::copysign(1.0,fcz(i,j,k,1)));
                Real fracx = (ccm(ii,j,k-1) || ccm(ii,j,k)) ? amrex::Math::abs(fcz(i,j,k,0)) : 0.0;
                Real fracy = (ccm(i,jj,k-1) || ccm(i,jj,k)) ? amrex::Math::abs(fcz(i,j,k,1)) : 0.0;
                if (beta_on_center && phi_on_center)
                {
                    fzm = (1.0-fracx)*(1.0-fracy)*fzm +
                        fracx*(1.0-fracy)*bZ(ii,j ,k,n)*(x(ii,j ,k,n)-x(ii,j ,k-1,n)) +
                        fracy*(1.0-fracx)*bZ(i ,jj,k,n)*(x(i ,jj,k,n)-x(i ,jj,k-1,n)) +
                        fracx*     fracy *bZ(ii,jj,k,n)*(x(ii,jj,k,n)-x(ii,jj,k-1,n));

                    fzmc = (1.0-fracx)*(1.0-fracy)*fzmc +
                        fracx*(1.0-fracy)*cZ(ii,j ,k,n)*getFaceStateUpwind(cZ(ii,j ,k,n),x(ii,j ,k-1,n),x(ii,j ,k,n)) +
                        fracy*(1.0-fracx)*cZ(i ,jj,k,n)*getFaceStateUpwind(cZ(i ,jj,k,n),x(i ,jj,k-1,n),x(i ,jj,k,n)) +
                        fracx*     fracy *cZ(ii,jj,k,n)*getFaceStateUpwind(cZ(ii,jj,k,n),x(ii,jj,k-1,n),x(ii,jj,k,n));
                }
                else if (beta_on_centroid && phi_on_center)
                {
                    fzm = (1.0-fracx)*(1.0-fracy)*(x( i, j,k,n)-x( i, j,k-1,n)) +
                               fracx *(1.0-fracy)*(x(ii, j,k,n)-x(ii, j,k-1,n)) +
                               fracy *(1.0-fracx)*(x( i,jj,k,n)-x( i,jj,k-1,n)) +
                               fracx *     fracy *(x(ii,jj,k,n)-x(ii,jj,k-1,n));

                    fzmc = (1.0-fracx)*(1.0-fracy)*getFaceStateUpwind(cZ( i, j,k,n),x( i, j,k-1,n),x( i, j,k,n)) +
                                fracx *(1.0-fracy)*getFaceStateUpwind(cZ(ii, j,k,n),x(ii, j,k-1,n),x(ii, j,k,n)) +
                                fracy *(1.0-fracx)*getFaceStateUpwind(cZ( i,jj,k,n),x( i,jj,k-1,n),x( i,jj,k,n)) +
                                fracx *     fracy *getFaceStateUpwind(cZ(ii,jj,k,n),x(ii,jj,k-1,n),x(ii,jj,k,n));

                    fzm *= bZ(i,j,k,n);
                    fzmc *= cZ(i,j,k,n);

                }
            }

            Real fzp = bZ(i,j,k+1,n)*(x(i,j,k+1,n) - x(i,j,k,n));
            Real fzpc = cZ(i,j,k+1,n)*getFaceStateUpwind(cZ(i,j,k+1,n), x(i,j,k  ,n), x(i,j,k+1,n));
            if (apzp != 0.0 && apzp != 1.0) {
                int ii = i + static_cast<int>(amrex::Math::copysign(1.0,fcz(i,j,k+1,0)));
                int jj = j + static_cast<int>(amrex::Math::copysign(1.0,fcz(i,j,k+1,1)));
                Real fracx = (ccm(ii,j,k) || ccm(ii,j,k+1)) ? amrex::Math::abs(fcz(i,j,k+1,0)) : 0.0;
                Real fracy = (ccm(i,jj,k) || ccm(i,jj,k+1)) ? amrex::Math::abs(fcz(i,j,k+1,1)) : 0.0;
                if (beta_on_center && phi_on_center)
                {
                    fzp = (1.0-fracx)*(1.0-fracy)*fzp +
                        fracx*(1.0-fracy)*bZ(ii,j ,k+1,n)*(x(ii,j ,k+1,n)-x(ii,j ,k,n)) +
                        fracy*(1.0-fracx)*bZ(i ,jj,k+1,n)*(x(i ,jj,k+1,n)-x(i ,jj,k,n)) +
                        fracx*     fracy *bZ(ii,jj,k+1,n)*(x(ii,jj,k+1,n)-x(ii,jj,k,n));

                    fzpc = (1.0-fracx)*(1.0-fracy)*fzpc +
                        fracx*(1.0-fracy)*cZ(ii,j ,k+1,n)*getFaceStateUpwind(cZ(ii,j ,k+1,n),x(ii,j ,k,n),x(ii,j ,k+1,n)) +
                        fracy*(1.0-fracx)*cZ(i ,jj,k+1,n)*getFaceStateUpwind(cZ(i ,jj,k+1,n),x(i ,jj,k,n),x(i ,jj,k+1,n)) +
                        fracx*     fracy *cZ(ii,jj,k+1,n)*getFaceStateUpwind(cZ(ii,jj,k+1,n),x(ii,jj,k,n),x(ii,jj,k+1,n));
                }
                else if (beta_on_centroid && phi_on_center)
                {
                    fzp = (1.0-fracx)*(1.0-fracy)*(x( i, j,k+1,n)-x( i, j,k,n)) +
                               fracx *(1.0-fracy)*(x(ii, j,k+1,n)-x(ii, j,k,n)) +
                               fracy *(1.0-fracx)*(x( i,jj,k+1,n)-x( i,jj,k,n)) +
                               fracx *     fracy *(x(ii,jj,k+1,n)-x(ii,jj,k,n));

                    fzpc = (1.0-fracx)*(1.0-fracy)*getFaceStateUpwind(cZ( i, j,k+1,n),x( i, j,k,n),x( i, j,k+1,n)) +
                                fracx *(1.0-fracy)*getFaceStateUpwind(cZ(ii, j,k+1,n),x(ii, j,k,n),x(ii, j,k+1,n)) +
                                fracy *(1.0-fracx)*getFaceStateUpwind(cZ( i,jj,k+1,n),x( i,jj,k,n),x( i,jj,k+1,n)) +
                                fracx *     fracy *getFaceStateUpwind(cZ(ii,jj,k+1,n),x(ii,jj,k,n),x(ii,jj,k+1,n));

                    fzp *= bZ(i,j,k+1,n);
                    fzpc *= cZ(i,j,k+1,n);

                }
            }

            Real feb = 0.0;
            if (is_dirichlet) {
                Real dapx = apxm-apxp;
                Real dapy = apym-apyp;
                Real dapz = apzm-apzp;
                Real anorm = std::sqrt(dapx*dapx+dapy*dapy+dapz*dapz);
                Real anorminv = 1.0/anorm;
                Real anrmx = dapx * anorminv;
                Real anrmy = dapy * anorminv;
                Real anrmz = dapz * anorminv;

                Real phib = is_inhomog ? phieb(i,j,k,n) : 0.0;

                Real bctx = bc(i,j,k,0);
                Real bcty = bc(i,j,k,1);
                Real bctz = bc(i,j,k,2);
                Real dx_eb = get_dx_eb(kappa);

                Real dg = dx_eb / amrex::max(amrex::Math::abs(anrmx), amrex::Math::abs(anrmy),
                                             amrex::Math::abs(anrmz));
                Real gx = bctx - dg*anrmx;
                Real gy = bcty - dg*anrmy;
                Real gz = bctz - dg*anrmz;
                Real sx = amrex::Math::copysign(1.0,anrmx);
                Real sy = amrex::Math::copysign(1.0,anrmy);
                Real sz = amrex::Math::copysign(1.0,anrmz);
                int ii = i - static_cast<int>(sx);
                int jj = j - static_cast<int>(sy);
                int kk = k - static_cast<int>(sz);

                gx = sx*gx;
                gy = sy*gy;
                gz = sz*gz;
                Real gxy = gx*gy;
                Real gxz = gx*gz;
                Real gyz = gy*gz;
                Real gxyz = gx*gy*gz;
                Real phig = (1.0+gx+gy+gz+gxy+gxz+gyz+gxyz) * x(i ,j ,k ,n)
                    +       (-gz - gxz - gyz - gxyz)        * x(i ,j ,kk,n)
                    +       (-gy - gxy - gyz - gxyz)        * x(i ,jj,k ,n)
                    +       (gyz + gxyz)                    * x(i ,jj,kk,n)
                    +       (-gx - gxy - gxz - gxyz)        * x(ii,j ,k ,n)
                    +       (gxz + gxyz)                    * x(ii,j ,kk,n)
                    +       (gxy + gxyz)                    * x(ii,jj,k ,n)
                    +       (-gxyz)                         * x(ii,jj,kk,n);

                Real dphidn = (phib-phig)/dg;

                feb = dphidn * ba(i,j,k) * beb(i,j,k,n);
            }

            y(i,j,k,n) = alpha*a(i,j,k)*x(i,j,k,n) + (1.0/kappa) *
                (dhx*(apxm*fxm - apxp*fxp) +
                 dhy*(apym*fym - apyp*fyp) +
                 dhz*(apzm*fzm - apzp*fzp) +
                 dhxc*(apxm*fxmc - apxp*fxpc) +
                 dhyc*(apym*fymc - apyp*fypc) +
                 dhzc*(apzm*fzmc - apzp*fzpc) - dhx*feb);
        }
    });
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void mlebabecceclap_ebflux (int i, int j, int k, int n,
                         Array4<Real> const& feb,
                         Array4<Real const> const& x,
                         Array4<EBCellFlag const> const& flag,
                         Array4<Real const> const& vfrc,
                         Array4<Real const> const& apx,
                         Array4<Real const> const& apy,
                         Array4<Real const> const& apz,
                         Array4<Real const> const& bc,
                         Array4<Real const> const& beb,
                         Array4<Real const> const& phieb,
                         bool is_inhomog,
                         GpuArray<Real,AMREX_SPACEDIM> const& dxinv) noexcept
{
    Real dhx = dxinv[0];

    if (!flag(i,j,k).isSingleValued())
    {
        feb(i,j,k,n) = 0.0;
    }
    else
    {
        Real kappa = vfrc(i,j,k);
        Real apxm = apx(i,j,k);
        Real apxp = apx(i+1,j,k);
        Real apym = apy(i,j,k);
        Real apyp = apy(i,j+1,k);
        Real apzm = apz(i,j,k);
        Real apzp = apz(i,j,k+1);

        Real dapx = apxm-apxp;
        Real dapy = apym-apyp;
        Real dapz = apzm-apzp;
        Real anorm = std::sqrt(dapx*dapx+dapy*dapy+dapz*dapz);
        Real anorminv = 1.0/anorm;
        Real anrmx = dapx * anorminv;
        Real anrmy = dapy * anorminv;
        Real anrmz = dapz * anorminv;

        Real phib = is_inhomog ? phieb(i,j,k,n) : 0.0;

        Real bctx = bc(i,j,k,0);
        Real bcty = bc(i,j,k,1);
        Real bctz = bc(i,j,k,2);
        Real dx_eb = get_dx_eb(kappa);

        Real dg = dx_eb / amrex::max(amrex::Math::abs(anrmx), amrex::Math::abs(anrmy), amrex::Math::abs(anrmz));
        Real gx = bctx - dg*anrmx;
        Real gy = bcty - dg*anrmy;
        Real gz = bctz - dg*anrmz;
        Real sx = amrex::Math::copysign(1.0,anrmx);
        Real sy = amrex::Math::copysign(1.0,anrmy);
        Real sz = amrex::Math::copysign(1.0,anrmz);
        int ii = i - static_cast<int>(sx);
        int jj = j - static_cast<int>(sy);
        int kk = k - static_cast<int>(sz);

        gx = sx*gx;
        gy = sy*gy;
        gz = sz*gz;
        Real gxy = gx*gy;
        Real gxz = gx*gz;
        Real gyz = gy*gz;
        Real gxyz = gx*gy*gz;
        Real phig = (1.0+gx+gy+gz+gxy+gxz+gyz+gxyz) * x(i ,j ,k ,n)
            +       (-gz - gxz - gyz - gxyz)        * x(i ,j ,kk,n)
            +       (-gy - gxy - gyz - gxyz)        * x(i ,jj,k ,n)
            +       (gyz + gxyz)                    * x(i ,jj,kk,n)
            +       (-gx - gxy - gxz - gxyz)        * x(ii,j ,k ,n)
            +       (gxz + gxyz)                    * x(ii,j ,kk,n)
            +       (gxy + gxyz)                    * x(ii,jj,k ,n)
            +       (-gxyz)                         * x(ii,jj,kk,n);

        Real dphidn = dhx*(phib-phig)/dg;
        feb(i,j,k,n) = -beb(i,j,k,n) * dphidn;
    }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void mlebabecceclap_gsrb (Box const& box,
                       Array4<Real> const& phi, Array4<Real const> const& rhs,
                       Real alpha, Array4<Real const> const& a,
                       Real dhx, Real dhy, Real dhz,
                       Real dhxc, Real dhyc, Real dhzc,
                       Array4<Real const> const& bX, Array4<Real const> const& bY,
                       Array4<Real const> const& bZ,
                       Array4<Real const> const& cX, Array4<Real const> const& cY,
                       Array4<Real const> const& cZ,
                       Array4<int const> const& m0, Array4<int const> const& m2,
                       Array4<int const> const& m4,
                       Array4<int const> const& m1, Array4<int const> const& m3,
                       Array4<int const> const& m5,
                       Array4<Real const> const& f0, Array4<Real const> const& f2,
                       Array4<Real const> const& f4,
                       Array4<Real const> const& f1, Array4<Real const> const& f3,
                       Array4<Real const> const& f5,
                       Array4<const int> const& ccm, Array4<EBCellFlag const> const& flag,
                       Array4<Real const> const& vfrc,
                       Array4<Real const> const& apx, Array4<Real const> const& apy,
                       Array4<Real const> const& apz,
                       Array4<Real const> const& fcx, Array4<Real const> const& fcy,
                       Array4<Real const> const& fcz,
                       Array4<Real const> const& ba, Array4<Real const> const& bc,
                       Array4<Real const> const& beb,
                       bool is_dirichlet, bool beta_on_centroid, bool phi_on_centroid,
                       Box const& vbox, int redblack, int ncomp) noexcept
{
    // constexpr Real omega = 1.15;
    constexpr Real omega = 0.7;

    const auto vlo = amrex::lbound(vbox);
    const auto vhi = amrex::ubound(vbox);

    // exit(1);
//    amrex::Loop(box, ncomp, [=] (int i, int j, int k, int n) noexcept
    // amrex::Loop here causes gcc 8 to crash.
    const auto lo = amrex::lbound(box);
    const auto hi = amrex::ubound(box);
    for (int n = 0; n < ncomp; ++n) {
    for (int k = lo.z; k <= hi.z; ++k) {
    for (int j = lo.y; j <= hi.y; ++j) {
    for (int i = lo.x; i <= hi.x; ++i)
    {
        if ((i+j+k+redblack) % 2 == 0)
        {
            if (flag(i,j,k).isCovered())
            {
                phi(i,j,k,n) = 0.0;
            }
            else
            {
                Real cf0 = (i == vlo.x && m0(vlo.x-1,j,k) > 0)
                    ? f0(vlo.x,j,k,n) : 0.0;
                Real cf1 = (j == vlo.y && m1(i,vlo.y-1,k) > 0)
                    ? f1(i,vlo.y,k,n) : 0.0;
                Real cf2 = (k == vlo.z && m2(i,j,vlo.z-1) > 0)
                    ? f2(i,j,vlo.z,n) : 0.0;
                Real cf3 = (i == vhi.x && m3(vhi.x+1,j,k) > 0)
                    ? f3(vhi.x,j,k,n) : 0.0;
                Real cf4 = (j == vhi.y && m4(i,vhi.y+1,k) > 0)
                    ? f4(i,vhi.y,k,n) : 0.0;
                Real cf5 = (k == vhi.z && m5(i,j,vhi.z+1) > 0)
                    ? f5(i,j,vhi.z,n) : 0.0;

                if (flag(i,j,k).isRegular())
                {
                    // getCellUpwindCoeff values always positive for same sign edge coeffs
                    Real gamma = alpha*a(i,j,k)
                        + dhx*(bX(i+1,j,k,n) + bX(i,j,k,n))
                        + dhy*(bY(i,j+1,k,n) + bY(i,j,k,n))
                        + dhz*(bZ(i,j,k+1,n) + bZ(i,j,k,n))
                        - dhxc*getCellUpwindCoeff(cX(i,j,k,n), cX(i+1,j,k,n))
                        - dhyc*getCellUpwindCoeff(cY(i,j,k,n), cY(i,j+1,k,n))
                        - dhzc*getCellUpwindCoeff(cZ(i,j,k,n), cZ(i,j,k+1,n));

                    // Right and Left Upwind coeff values negative or zero
                    Real rho = dhx*(bX(i+1,j  ,k  ,n)*phi(i+1,j  ,k  ,n) +
                                    bX(i  ,j  ,k  ,n)*phi(i-1,j  ,k  ,n))
                        +      dhy*(bY(i  ,j+1,k  ,n)*phi(i  ,j+1,k  ,n) +
                                    bY(i  ,j  ,k  ,n)*phi(i  ,j-1,k  ,n))
                        +      dhz*(bZ(i  ,j  ,k+1,n)*phi(i  ,j  ,k+1,n) +
                                    bZ(i  ,j  ,k  ,n)*phi(i  ,j  ,k-1,n))
                        +      dhxc*( getLeftCellUpwindCoeff( cX(i  ,j  ,k,n),cX(i+1,j  ,k,  n))*phi(i-1,j  ,k  ,n)
                                    + getRightCellUpwindCoeff(cX(i  ,j  ,k,n),cX(i+1,j  ,k,  n))*phi(i+1,j  ,k  ,n))
                        +      dhyc*( getLeftCellUpwindCoeff( cY(i  ,j  ,k,n),cY(i  ,j+1,k,  n))*phi(i  ,j-1,k  ,n)
                                    + getRightCellUpwindCoeff(cY(i  ,j  ,k,n),cY(i  ,j+1,k,  n))*phi(i  ,j+1,k  ,n))
                        +      dhzc*( getLeftCellUpwindCoeff( cZ(i  ,j  ,k,n),cZ(i  ,j  ,k+1,n))*phi(i  ,j  ,k-1,n)
                                    + getRightCellUpwindCoeff(cZ(i  ,j  ,k,n),cZ(i  ,j  ,k+1,n))*phi(i  ,j  ,k+1,n));

                    // TODO: add c coef terms to delta..? Lucas's impl doesnt do this
                    // Boundary stuff ... may not need to update
                    Real delta = dhx*(bX(i,j,k,n)*cf0 + bX(i+1,j,k,n)*cf3)
                        +        dhy*(bY(i,j,k,n)*cf1 + bY(i,j+1,k,n)*cf4)
                        +        dhz*(bZ(i,j,k,n)*cf2 + bZ(i,j,k+1,n)*cf5);

                    Real res = rhs(i,j,k,n) - (gamma*phi(i,j,k,n) - rho);
                    phi(i,j,k,n) += omega*res/(gamma-delta);
                }
                else
                {
                    Real kappa = vfrc(i,j,k);
                    Real apxm = apx(i,j,k);
                    Real apxp = apx(i+1,j,k);
                    Real apym = apy(i,j,k);
                    Real apyp = apy(i,j+1,k);
                    Real apzm = apz(i,j,k);
                    Real apzp = apz(i,j,k+1);

                    Real fxm = -bX(i,j,k,n)*phi(i-1,j,k,n);
                    Real oxm = -bX(i,j,k,n)*cf0;
                    Real sxm =  bX(i,j,k,n);
                    Real fxmc = -getLeftCellUpwindCoeff(cX(i,j,k,n),cX(i+1,j,k,n))*phi(i-1,j  ,k  ,n);
                    Real sxmc = getCellUpwindCoeffLeft(cX(i,j,k,n), cX(i+1,j,k,n));
                    if (apxm != 0.0 && apxm != 1.0) {
                        int jj = j + static_cast<int>(amrex::Math::copysign(1.0, fcx(i,j,k,0)));
                        int kk = k + static_cast<int>(amrex::Math::copysign(1.0, fcx(i,j,k,1)));
                        Real fracy = (ccm(i-1,jj,k) || ccm(i,jj,k))
                            ? amrex::Math::abs(fcx(i,j,k,0)) : 0.0;
                        Real fracz = (ccm(i-1,j,kk) || ccm(i,j,kk))
                            ? amrex::Math::abs(fcx(i,j,k,1)) : 0.0;
                        if (!beta_on_centroid && !phi_on_centroid)
                        {
                            fxm = (1.0-fracy)*(1.0-fracz)*fxm
                                 +     fracy *(1.0-fracz)*bX(i,jj,k ,n)*(phi(i,jj,k ,n)-phi(i-1,jj,k ,n))
                                 +(1.0-fracy)*     fracz *bX(i,j ,kk,n)*(phi(i,j ,kk,n)-phi(i-1,j ,kk,n))
                                 +     fracy *     fracz *bX(i,jj,kk,n)*(phi(i,jj,kk,n)-phi(i-1,jj,kk,n));

                            // TODO: should we be subtracting each component?
                            fxmc = (1.0-fracy)*(1.0-fracz)*fxmc
                                 +     fracy *(1.0-fracz)*getLeftCellUpwindCoeff(cX(i,jj,k,n) ,cX(i+1,jj,k,n) )*getFaceStateUpwind(cX(i,jj,k ,n),phi(i-1,jj,k ,n),phi(i,jj,k ,n))
                                 +(1.0-fracy)*     fracz *getLeftCellUpwindCoeff(cX(i,j,kk,n) ,cX(i+1,j,kk,n) )*getFaceStateUpwind(cX(i,j ,kk,n),phi(i-1,j ,kk,n),phi(i,j ,kk,n))
                                 +     fracy *     fracz *getLeftCellUpwindCoeff(cX(i,jj,kk,n),cX(i+1,jj,kk,n))*getFaceStateUpwind(cX(i,jj,kk,n),phi(i-1,jj,kk,n),phi(i,jj,kk,n));
                        }
                        else if (beta_on_centroid && !phi_on_centroid)
                        {
                            fxm = (1.0-fracy)*(1.0-fracz)*(              -phi(i-1, j, k,n))
                                 +     fracy *(1.0-fracz)*(phi(i,jj,k ,n)-phi(i-1,jj, k,n))
                                 +(1.0-fracy)*     fracz *(phi(i,j ,kk,n)-phi(i-1, j,kk,n))
                                 +     fracy *     fracz *(phi(i,jj,kk,n)-phi(i-1,jj,kk,n));
                            fxm *= bX(i,j,k,n);

                            fxmc = (1.0-fracy)*(1.0-fracz)*(              -phi(i-1, j, k,n))
                                 +     fracy *(1.0-fracz)*getFaceStateUpwind(cX(i,jj,k ,n),phi(i-1,jj, k,n),phi(i,jj,k ,n))
                                 +(1.0-fracy)*     fracz *getFaceStateUpwind(cX(i,j ,kk,n),phi(i-1, j,kk,n),phi(i,j ,kk,n))
                                 +     fracy *     fracz *getFaceStateUpwind(cX(i,jj,kk,n),phi(i-1,jj,kk,n),phi(i,jj,kk,n));
                            fxmc *= getLeftCellUpwindCoeff(cX(i,j,k,n) ,cX(i+1,j,k,n) );
                        }
                        oxm = 0.0;
                        sxm = (1.0-fracy)*(1.0-fracz)*sxm;
                        sxmc = (1.0-fracy)*(1.0-fracz)*sxmc;
                    }

                    Real fxp =  bX(i+1,j,k,n)*phi(i+1,j,k,n);
                    Real oxp =  bX(i+1,j,k,n)*cf3;
                    Real sxp = -bX(i+1,j,k,n);
                    Real fxpc =  getRightCellUpwindCoeff(cX(i,j,k,n), cX(i+1,j,k,n))*phi(i-1,j  ,k  ,n);
                    Real sxpc = -getCellUpwindCoeffRight(cX(i,j,k,n), cX(i+1,j,k,n));
                    if (apxp != 0.0 && apxp != 1.0) {
                        int jj = j + static_cast<int>(amrex::Math::copysign(1.0,fcx(i+1,j,k,0)));
                        int kk = k + static_cast<int>(amrex::Math::copysign(1.0,fcx(i+1,j,k,1)));
                        Real fracy = (ccm(i,jj,k) || ccm(i+1,jj,k))
                            ? amrex::Math::abs(fcx(i+1,j,k,0)) : 0.0;
                        Real fracz = (ccm(i,j,kk) || ccm(i+1,j,kk))
                            ? amrex::Math::abs(fcx(i+1,j,k,1)) : 0.0;
                        if (!beta_on_centroid && !phi_on_centroid)
                        {
                            fxp = (1.0-fracy)*(1.0-fracz)*fxp
                                  +    fracy *(1.0-fracz)*bX(i+1,jj,k ,n)*(phi(i+1,jj,k ,n)-phi(i,jj,k ,n))
                                 +(1.0-fracy)*     fracz *bX(i+1,j ,kk,n)*(phi(i+1,j ,kk,n)-phi(i,j ,kk,n))
                                 +     fracy *     fracz *bX(i+1,jj,kk,n)*(phi(i+1,jj,kk,n)-phi(i,jj,kk,n));

                            fxpc = (1.0-fracy)*(1.0-fracz)*fxpc
                                  +    fracy *(1.0-fracz)*getRightCellUpwindCoeff(cX(i,jj,k,n) ,cX(i+1,jj,k,n) )*getFaceStateUpwind(cX(i+1,jj,k ,n),phi(i,jj,k ,n),phi(i+1,jj,k ,n))
                                 +(1.0-fracy)*     fracz *getRightCellUpwindCoeff(cX(i,j,kk,n) ,cX(i+1,j,kk,n) )*getFaceStateUpwind(cX(i+1,j ,kk,n),phi(i,j ,kk,n),phi(i+1,j ,kk,n))
                                 +     fracy *     fracz *getRightCellUpwindCoeff(cX(i,jj,kk,n),cX(i+1,jj,kk,n))*getFaceStateUpwind(cX(i+1,jj,kk,n),phi(i,jj,kk,n),phi(i+1,jj,kk,n));
                        }
                        else if (beta_on_centroid && !phi_on_centroid)
                        {
                            fxp = (1.0-fracy)*(1.0-fracz)*(phi(i+1, j, k,n)               ) +
                                       fracy *(1.0-fracz)*(phi(i+1,jj, k,n)-phi(i,jj, k,n)) +
                                       fracz *(1.0-fracy)*(phi(i+1, j,kk,n)-phi(i, j,kk,n)) +
                                       fracy *     fracz *(phi(i+1,jj,kk,n)-phi(i,jj,kk,n));
                            fxp *= bX(i+1,j,k,n);

                            fxpc = (1.0-fracy)*(1.0-fracz)*(phi(i+1, j, k,n)               ) +
                                        fracy *(1.0-fracz)*getFaceStateUpwind(cX(i+1,jj, k,n),phi(i,jj, k,n),phi(i+1,jj, k,n)) +
                                        fracz *(1.0-fracy)*getFaceStateUpwind(cX(i+1, j,kk,n),phi(i, j,kk,n),phi(i+1, j,kk,n)) +
                                        fracy *     fracz *getFaceStateUpwind(cX(i+1,jj,kk,n),phi(i,jj,kk,n),phi(i+1,jj,kk,n));
                            fxpc *= getRightCellUpwindCoeff(cX(i,j,k,n) ,cX(i+1,j,k,n) );
                        }

                        oxp = 0.0;
                        sxp = (1.0-fracy)*(1.0-fracz)*sxp;
                        sxpc = (1.0-fracy)*(1.0-fracz)*sxpc;
                    }

                    Real fym = -bY(i,j,k,n)*phi(i,j-1,k,n);
                    Real oym = -bY(i,j,k,n)*cf1;
                    Real sym =  bY(i,j,k,n);
                    Real fymc = -getLeftCellUpwindCoeff(cY(i,j,k,n), cY(i,j+1,k,n))*phi(i,j-1,k,n);
                    Real symc =  getCellUpwindCoeffLeft(cY(i,j,k,n), cY(i,j+1,k,n));
                    if (apym != 0.0 && apym != 1.0) {
                        int ii = i + static_cast<int>(amrex::Math::copysign(1.0,fcy(i,j,k,0)));
                        int kk = k + static_cast<int>(amrex::Math::copysign(1.0,fcy(i,j,k,1)));
                        Real fracx = (ccm(ii,j-1,k) || ccm(ii,j,k))
                            ? amrex::Math::abs(fcy(i,j,k,0)) : 0.0;
                        Real fracz = (ccm(i,j-1,kk) || ccm(i,j,kk))
                            ? amrex::Math::abs(fcy(i,j,k,1)) : 0.0;
                        if (!beta_on_centroid && !phi_on_centroid)
                        {
                            fym = (1.0-fracx)*(1.0-fracz)*fym
                                +      fracx *(1.0-fracz)*bY(ii,j,k ,n)*(phi(ii,j,k ,n)-phi(ii,j-1,k ,n))
                                + (1.0-fracx)*     fracz *bY(i ,j,kk,n)*(phi(i ,j,kk,n)-phi(i ,j-1,kk,n))
                                +      fracx *     fracz *bY(ii,j,kk,n)*(phi(ii,j,kk,n)-phi(ii,j-1,kk,n));

                            fymc = (1.0-fracx)*(1.0-fracz)*fymc
                                +      fracx *(1.0-fracz)*getLeftCellUpwindCoeff(cY(ii,j,k ,n),cY(ii,j+1,k ,n))*getFaceStateUpwind(cY(ii,j,k ,n),phi(ii,j-1,k ,n),phi(ii,j,k ,n))
                                + (1.0-fracx)*     fracz *getLeftCellUpwindCoeff(cY(i ,j,kk,n),cY(i ,j+1,kk,n))*getFaceStateUpwind(cY(i ,j,kk,n),phi(i ,j-1,kk,n),phi(i ,j,kk,n))
                                +      fracx *     fracz *getLeftCellUpwindCoeff(cY(ii,j,kk,n),cY(ii,j+1,kk,n))*getFaceStateUpwind(cY(ii,j,kk,n),phi(ii,j-1,kk,n),phi(ii,j,kk,n));
                        }
                        else if (beta_on_centroid && !phi_on_centroid)
                        {
                            fym = (1.0-fracx)*(1.0-fracz)*(              -phi( i,j-1, k,n))
                                +      fracx *(1.0-fracz)*(phi(ii,j,k ,n)-phi(ii,j-1, k,n))
                                + (1.0-fracx)*     fracz *(phi(i ,j,kk,n)-phi( i,j-1,kk,n))
                                +      fracx *     fracz *(phi(ii,j,kk,n)-phi(ii,j-1,kk,n));
                            fym *= bY(i,j,k,n);

                            fymc = (1.0-fracx)*(1.0-fracz)*(              -phi( i,j-1, k,n))
                                +      fracx *(1.0-fracz)*getFaceStateUpwind(cY(ii,j,k ,n),phi(ii,j-1, k,n),phi(ii,j,k ,n))
                                + (1.0-fracx)*     fracz *getFaceStateUpwind(cY(i ,j,kk,n),phi( i,j-1,kk,n),phi(i ,j,kk,n))
                                +      fracx *     fracz *getFaceStateUpwind(cY(ii,j,kk,n),phi(ii,j-1,kk,n),phi(ii,j,kk,n));
                            fymc *= getLeftCellUpwindCoeff(cY(i,j,k,n) ,cY(i,j+1,k,n) );
                        }
                        oym = 0.0;
                        sym = (1.0-fracx)*(1.0-fracz)*sym;
                        symc = (1.0-fracx)*(1.0-fracz)*symc;
                    }

                    Real fyp =  bY(i,j+1,k,n)*phi(i,j+1,k,n);
                    Real oyp =  bY(i,j+1,k,n)*cf4;
                    Real syp = -bY(i,j+1,k,n);
                    Real fypc =  getRightCellUpwindCoeff(cY(i,j,k,n), cY(i,j+1,k,n))*phi(i,j+1,k,n);
                    Real sypc = -getCellUpwindCoeffRight(cY(i,j,k,n), cY(i,j+1,k,n));
                    if (apyp != 0.0 && apyp != 1.0) {
                        int ii = i + static_cast<int>(amrex::Math::copysign(1.0,fcy(i,j+1,k,0)));
                        int kk = k + static_cast<int>(amrex::Math::copysign(1.0,fcy(i,j+1,k,1)));
                        Real fracx = (ccm(ii,j,k) || ccm(ii,j+1,k))
                            ? amrex::Math::abs(fcy(i,j+1,k,0)) : 0.0;
                        Real fracz = (ccm(i,j,kk) || ccm(i,j+1,kk))
                            ? amrex::Math::abs(fcy(i,j+1,k,1)) : 0.0;
                        if (!beta_on_centroid && !phi_on_centroid)
                        {
                            fyp = (1.0-fracx)*(1.0-fracz)*fyp
                                +      fracx *(1.0-fracz)*bY(ii,j+1,k ,n)*(phi(ii,j+1,k ,n)-phi(ii,j,k ,n))
                                + (1.0-fracx)*     fracz *bY(i ,j+1,kk,n)*(phi(i ,j+1,kk,n)-phi(i ,j,kk,n))
                                +      fracx *     fracz *bY(ii,j+1,kk,n)*(phi(ii,j+1,kk,n)-phi(ii,j,kk,n));

                            fypc = (1.0-fracx)*(1.0-fracz)*fypc
                                +      fracx *(1.0-fracz)*getRightCellUpwindCoeff(cY(ii,j,k ,n),cY(ii,j+1,k ,n))*getFaceStateUpwind(cY(ii,j+1,k ,n),phi(ii,j,k ,n),phi(ii,j+1,k ,n))
                                + (1.0-fracx)*     fracz *getRightCellUpwindCoeff(cY(i ,j,kk,n),cY(i ,j+1,kk,n))*getFaceStateUpwind(cY(i ,j+1,kk,n),phi(i ,j,kk,n),phi(i ,j+1,kk,n))
                                +      fracx *     fracz *getRightCellUpwindCoeff(cY(ii,j,kk,n),cY(ii,j+1,kk,n))*getFaceStateUpwind(cY(ii,j+1,kk,n),phi(ii,j,kk,n),phi(ii,j+1,kk,n));
                        }
                        else if (beta_on_centroid && !phi_on_centroid)
                        {
                            fyp = (1.0-fracx)*(1.0-fracz)*(phi( i,j+1, k,n)               )
                                +      fracx *(1.0-fracz)*(phi(ii,j+1, k,n)-phi(ii,j, k,n))
                                + (1.0-fracx)*     fracz *(phi( i,j+1,kk,n)-phi( i,j,kk,n))
                                +      fracx *     fracz *(phi(ii,j+1,kk,n)-phi(ii,j,kk,n));
                            fyp *= bY(i,j+1,k,n);

                            fypc = (1.0-fracx)*(1.0-fracz)*(phi( i,j+1, k,n)               )
                                +      fracx *(1.0-fracz)*getFaceStateUpwind(cY(ii,j+1, k,n),phi(ii,j, k,n),phi(ii,j+1, k,n))
                                + (1.0-fracx)*     fracz *getFaceStateUpwind(cY( i,j+1,kk,n),phi( i,j,kk,n),phi( i,j+1,kk,n))
                                +      fracx *     fracz *getFaceStateUpwind(cY(ii,j+1,kk,n),phi(ii,j,kk,n),phi(ii,j+1,kk,n));
                            fypc *= getRightCellUpwindCoeff(cY(i,j,k,n) ,cY(i,j+1,k,n) );
                        }
                        oyp = 0.0;
                        sypc = (1.0-fracx)*(1.0-fracz)*sypc;
                    }

                    Real fzm = -bZ(i,j,k,n)*phi(i,j,k-1,n);
                    Real ozm = -bZ(i,j,k,n)*cf2;
                    Real szm =  bZ(i,j,k,n);
                    Real fzmc = -getLeftCellUpwindCoeff(cZ(i,j,k,n), cZ(i,j,k+1,n))*phi(i,j,k-1,n);
                    Real szmc =  getCellUpwindCoeffLeft(cZ(i,j,k,n), cZ(i,j,k+1,n));
                    if (apzm != 0.0 && apzm != 1.0) {
                        int ii = i + static_cast<int>(amrex::Math::copysign(1.0,fcz(i,j,k,0)));
                        int jj = j + static_cast<int>(amrex::Math::copysign(1.0,fcz(i,j,k,1)));
                        Real fracx = (ccm(ii,j,k-1) || ccm(ii,j,k))
                            ? amrex::Math::abs(fcz(i,j,k,0)) : 0.0;
                        Real fracy = (ccm(i,jj,k-1) || ccm(i,jj,k))
                            ? amrex::Math::abs(fcz(i,j,k,1)) : 0.0;
                        if (!beta_on_centroid && !phi_on_centroid)
                        {
                            fzm = (1.0-fracx)*(1.0-fracy)*fzm
                                 +     fracx *(1.0-fracy)*bZ(ii, j,k,n)*(phi(ii, j,k,n)-phi(ii, j,k-1,n))
                                 +(1.0-fracx)*     fracy *bZ( i,jj,k,n)*(phi( i,jj,k,n)-phi( i,jj,k-1,n))
                                 +     fracx *     fracy *bZ(ii,jj,k,n)*(phi(ii,jj,k,n)-phi(ii,jj,k-1,n));

                            fzmc = (1.0-fracx)*(1.0-fracy)*fzmc
                                 +     fracx *(1.0-fracy)*getLeftCellUpwindCoeff(cZ(ii, j,k,n),cZ(ii, j,k+1,n))*getFaceStateUpwind(cZ(ii, j,k,n),phi(ii, j,k-1,n),phi(ii, j,k,n))
                                 +(1.0-fracx)*     fracy *getLeftCellUpwindCoeff(cZ( i,jj,k,n),cZ( i,jj,k+1,n))*getFaceStateUpwind(cZ( i,jj,k,n),phi( i,jj,k-1,n),phi( i,jj,k,n))
                                 +     fracx *     fracy *getLeftCellUpwindCoeff(cZ(ii,jj,k,n),cZ(ii,jj,k+1,n))*getFaceStateUpwind(cZ(ii,jj,k,n),phi(ii,jj,k-1,n),phi(ii,jj,k,n));
                        }
                        else if (beta_on_centroid && !phi_on_centroid)
                        {
                            fzm = (1.0-fracx)*(1.0-fracy)*(              -phi( i, j,k-1,n))
                                +      fracx *(1.0-fracy)*(phi(ii, j,k,n)-phi(ii, j,k-1,n))
                                + (1.0-fracx)*     fracy *(phi( i,jj,k,n)-phi(i ,jj,k-1,n))
                                +      fracx *     fracy *(phi(ii,jj,k,n)-phi(ii,jj,k-1,n));
                            fzm *= bZ(i,j,k,n);

                            fzmc = (1.0-fracx)*(1.0-fracy)*(              -phi( i, j,k-1,n))
                                +      fracx *(1.0-fracy)*getFaceStateUpwind(cZ(ii, j,k,n),phi(ii, j,k-1,n),phi(ii, j,k,n))
                                + (1.0-fracx)*     fracy *getFaceStateUpwind(cZ( i,jj,k,n),phi(i ,jj,k-1,n),phi( i,jj,k,n))
                                +      fracx *     fracy *getFaceStateUpwind(cZ(ii,jj,k,n),phi(ii,jj,k-1,n),phi(ii,jj,k,n));
                            fzmc *= getLeftCellUpwindCoeff(cZ(i, j,k,n),cZ(i, j,k+1,n));
                        }
                        ozm = 0.0;
                        szm = (1.0-fracx)*(1.0-fracy)*szm;
                        szmc = (1.0-fracx)*(1.0-fracy)*szmc;
                    }

                    Real fzp =  bZ(i,j,k+1,n)*phi(i,j,k+1,n);
                    Real ozp =  bZ(i,j,k+1,n)*cf5;
                    Real szp = -bZ(i,j,k+1,n);
                    Real fzpc = getRightCellUpwindCoeff(cZ(i,j,k,n), cZ(i,j,k+1,n))*phi(i,j,k+1,n);
                    Real szpc = -getCellUpwindCoeffRight(cZ(i,j,k,n), cZ(i,j,k+1,n));
                    if (apzp != 0.0 && apzp != 1.0) {
                        int ii = i + static_cast<int>(amrex::Math::copysign(1.0,fcz(i,j,k+1,0)));
                        int jj = j + static_cast<int>(amrex::Math::copysign(1.0,fcz(i,j,k+1,1)));
                        Real fracx = (ccm(ii,j,k) || ccm(ii,j,k+1))
                            ? amrex::Math::abs(fcz(i,j,k+1,0)) : 0.0;
                        Real fracy = (ccm(i,jj,k) || ccm(i,jj,k+1))
                            ? amrex::Math::abs(fcz(i,j,k+1,1)) : 0.0;
                        if (!beta_on_centroid && !phi_on_centroid)
                        {
                            fzp = (1.0-fracx)*(1.0-fracy)*fzp
                                +      fracx *(1.0-fracy)*bZ(ii,j ,k+1,n)*(phi(ii,j ,k+1,n)-phi(ii,j ,k,n))
                                + (1.0-fracx)*     fracy *bZ(i ,jj,k+1,n)*(phi(i ,jj,k+1,n)-phi(i ,jj,k,n))
                                +      fracx *     fracy *bZ(ii,jj,k+1,n)*(phi(ii,jj,k+1,n)-phi(ii,jj,k,n));

                            fzpc = (1.0-fracx)*(1.0-fracy)*fzpc
                                +      fracx *(1.0-fracy)*getRightCellUpwindCoeff(cZ(ii, j,k,n),cZ(ii, j,k+1,n))*getFaceStateUpwind(cZ(ii,j ,k+1,n),phi(ii,j ,k,n),phi(ii,j ,k+1,n))
                                + (1.0-fracx)*     fracy *getRightCellUpwindCoeff(cZ( i,jj,k,n),cZ( i,jj,k+1,n))*getFaceStateUpwind(cZ(i ,jj,k+1,n),phi(i ,jj,k,n),phi(i ,jj,k+1,n))
                                +      fracx *     fracy *getRightCellUpwindCoeff(cZ(ii,jj,k,n),cZ(ii,jj,k+1,n))*getFaceStateUpwind(cZ(ii,jj,k+1,n),phi(ii,jj,k,n),phi(ii,jj,k+1,n));
                        }
                        else if (beta_on_centroid && !phi_on_centroid)
                        {
                            fzp = (1.0-fracx)*(1.0-fracy)*(phi( i, j,k+1,n)               )
                                +      fracx *(1.0-fracy)*(phi(ii, j,k+1,n)-phi(ii, j,k,n))
                                + (1.0-fracx)*     fracy *(phi( i,jj,k+1,n)-phi( i,jj,k,n))
                                +      fracx *     fracy *(phi(ii,jj,k+1,n)-phi(ii,jj,k,n));
                            fzp *= bZ(i,j,k+1,n);

                            fzpc = (1.0-fracx)*(1.0-fracy)*(phi( i, j,k+1,n)               )
                                +      fracx *(1.0-fracy)*getFaceStateUpwind(cZ(ii, j,k+1,n),phi(ii, j,k,n),phi(ii, j,k+1,n))
                                + (1.0-fracx)*     fracy *getFaceStateUpwind(cZ( i,jj,k+1,n),phi( i,jj,k,n),phi( i,jj,k+1,n))
                                +      fracx *     fracy *getFaceStateUpwind(cZ(ii,jj,k+1,n),phi(ii,jj,k,n),phi(ii,jj,k+1,n));
                            fzpc *= getRightCellUpwindCoeff(cZ(i, j,k,n),cZ(i, j,k+1,n));
                        }
                        ozp = 0.0;
                        szp = (1.0-fracx)*(1.0-fracy)*szp;
                        szpc = (1.0-fracx)*(1.0-fracy)*szpc;
                    }

                    Real vfrcinv = 1.0/kappa;
                    Real gamma = alpha*a(i,j,k) + vfrcinv *
                        (dhx*(apxm*sxm-apxp*sxp) +
                         dhy*(apym*sym-apyp*syp) +
                         dhz*(apzm*szm-apzp*szp) -
                         dhxc*(apxm*sxmc-apxp*sxpc) -                                             
                         dhyc*(apym*symc-apyp*sypc) -
                         dhzc*(apzm*szmc-apzp*szpc));

                    Real rho = -vfrcinv *
                        (dhx*(apxm*fxm-apxp*fxp) +
                         dhy*(apym*fym-apyp*fyp) +
                         dhz*(apzm*fzm-apzp*fzp) +
                         dhxc*(apxm*fxmc-apxp*fxpc) +
                         dhyc*(apym*fymc-apyp*fypc) +
                         dhzc*(apzm*fzmc-apzp*fzpc));

        

                    Real delta = -vfrcinv *
                        (dhx*(apxm*oxm-apxp*oxp) +
                         dhy*(apym*oym-apyp*oyp) +
                         dhz*(apzm*ozm-apzp*ozp));

                    if (is_dirichlet) {
                        Real dapx = apxm-apxp;
                        Real dapy = apym-apyp;
                        Real dapz = apzm-apzp;
                        Real anorm = std::sqrt(dapx*dapx+dapy*dapy+dapz*dapz);
                        Real anorminv = 1.0/anorm;
                        Real anrmx = dapx * anorminv;
                        Real anrmy = dapy * anorminv;
                        Real anrmz = dapz * anorminv;
                        Real bctx = bc(i,j,k,0);
                        Real bcty = bc(i,j,k,1);
                        Real bctz = bc(i,j,k,2);
                        Real dx_eb = get_dx_eb(vfrc(i,j,k));

                        Real dg = dx_eb / amrex::max(amrex::Math::abs(anrmx),amrex::Math::abs(anrmy),
                                                     amrex::Math::abs(anrmz));

                        Real gx = bctx - dg*anrmx;
                        Real gy = bcty - dg*anrmy;
                        Real gz = bctz - dg*anrmz;
                        Real sx = amrex::Math::copysign(1.0,anrmx);
                        Real sy = amrex::Math::copysign(1.0,anrmy);
                        Real sz = amrex::Math::copysign(1.0,anrmz);
                        int ii = i - static_cast<int>(sx);
                        int jj = j - static_cast<int>(sy);
                        int kk = k - static_cast<int>(sz);

                        gx *= sx;
                        gy *= sy;
                        gz *= sz;
                        Real gxy = gx*gy;
                        Real gxz = gx*gz;
                        Real gyz = gy*gz;
                        Real gxyz = gx*gy*gz;
                        Real phig_gamma = (1.0+gx+gy+gz+gxy+gxz+gyz+gxyz);
                        Real phig = (-gz - gxz - gyz - gxyz) * phi(i,j,kk,n)
                            + (-gy - gxy - gyz - gxyz) * phi(i,jj,k,n)
                            + (gyz + gxyz) * phi(i,jj,kk,n)
                            + (-gx - gxy - gxz - gxyz) * phi(ii,j,k,n)
                            + (gxz + gxyz) * phi(ii,j,kk,n)
                            + (gxy + gxyz) * phi(ii,jj,k,n)
                            + (-gxyz) * phi(ii,jj,kk,n);

                        Real dphidn    = (    -phig)/dg;
                        Real feb_gamma = -phig_gamma/dg * ba(i,j,k) * beb(i,j,k,n);
                        gamma += vfrcinv*(-dhx)*feb_gamma;
                        Real feb = dphidn * ba(i,j,k) * beb(i,j,k,n);
                        rho += -vfrcinv*(-dhx)*feb;
                    }

                    Real res = rhs(i,j,k,n) - (gamma*phi(i,j,k,n) - rho);
                    phi(i,j,k,n) += omega*res/(gamma-delta);
                }
            }
        }
    }}}}
//    });
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void mlebabecceclap_flux_x (Box const& box, Array4<Real> const& fx, Array4<Real const> const& apx,
                         Array4<Real const> const& fcx, Array4<Real const> const& sol,
                         Array4<Real const> const& bX, Array4<Real const> const& cX,
                         Array4<int const> const& ccm, Real dhx, Real dhxc, 
                         int face_only, int ncomp, Box const& xbox,
                         bool beta_on_centroid, bool phi_on_centroid) noexcept
{
    int lof = xbox.smallEnd(0);
    int hif = xbox.bigEnd(0);
    amrex::LoopConcurrent(box, ncomp, [=] (int i, int j, int k, int n) noexcept
    {
        if (!face_only || lof == i || hif == i) {
            if (apx(i,j,k) == 0.0) {
                fx(i,j,k,n) = 0.0;
            } else if (apx(i,j,k) == 1.0) {
                fx(i,j,k,n) = -dhx*bX(i,j,k,n)*(sol(i,j,k,n)-sol(i-1,j,k,n))
                              -dhxc*getFaceUpwindFlux(cX(i,j,k,n),sol(i-1,j,k,n),sol(i,j,k,n));

            } else {
                Real fxm = bX(i,j,k,n)*(sol(i,j,k,n) - sol(i-1,j,k,n));
                Real fxmc = getFaceUpwindFlux(cX(i,j,k,n),sol(i-1,j,k,n),sol(i,j,k,n));
                int jj = j + static_cast<int>(amrex::Math::copysign(1.0, fcx(i,j,k,0)));
                int kk = k + static_cast<int>(amrex::Math::copysign(1.0, fcx(i,j,k,1)));
                Real fracy = (ccm(i-1,jj,k) || ccm(i,jj,k)) ? amrex::Math::abs(fcx(i,j,k,0)) : 0.0;
                Real fracz = (ccm(i-1,j,kk) || ccm(i,j,kk)) ? amrex::Math::abs(fcx(i,j,k,1)) : 0.0;
                if (!beta_on_centroid && !phi_on_centroid)
                {
                    fxm = (1.0-fracy)*(1.0-fracz)*fxm +
                               fracy *(1.0-fracz)*bX(i,jj,k ,n)*(sol(i,jj,k ,n)-sol(i-1,jj,k ,n)) +
                               fracz *(1.0-fracy)*bX(i,j ,kk,n)*(sol(i,j ,kk,n)-sol(i-1,j ,kk,n)) +
                               fracy*      fracz *bX(i,jj,kk,n)*(sol(i,jj,kk,n)-sol(i-1,jj,kk,n));

                    fxmc = (1.0-fracy)*(1.0-fracz)*fxmc +
                               fracy *(1.0-fracz)*getFaceUpwindFlux(cX(i,jj,k ,n),sol(i-1,jj,k ,n),sol(i,jj,k ,n)) +
                               fracz *(1.0-fracy)*getFaceUpwindFlux(cX(i,j ,kk,n),sol(i-1,j ,kk,n),sol(i,j ,kk,n)) +
                               fracy*      fracz *getFaceUpwindFlux(cX(i,jj,kk,n),sol(i-1,jj,kk,n),sol(i,jj,kk,n));
                }
                else if (beta_on_centroid && !phi_on_centroid)
                {
                    fxm = (1.0-fracy)*(1.0-fracz)*(sol(i, j, k,n)-sol(i-1, j, k,n)) +
                               fracy *(1.0-fracz)*(sol(i,jj, k,n)-sol(i-1,jj, k,n)) +
                               fracz *(1.0-fracy)*(sol(i, j,kk,n)-sol(i-1, j,kk,n)) +
                               fracy*      fracz *(sol(i,jj,kk,n)-sol(i-1,jj,kk,n));

                    fxmc = (1.0-fracy)*(1.0-fracz)*getFaceStateUpwind(cX(i, j, k,n),sol(i-1, j, k,n),sol(i, j, k,n)) +
                                fracy *(1.0-fracz)*getFaceStateUpwind(cX(i,jj, k,n),sol(i-1,jj, k,n),sol(i,jj, k,n)) +
                                fracz *(1.0-fracy)*getFaceStateUpwind(cX(i, j,kk,n),sol(i-1, j,kk,n),sol(i, j,kk,n)) +
                                fracy *     fracz *getFaceStateUpwind(cX(i,jj,kk,n),sol(i-1,jj,kk,n),sol(i,jj,kk,n));

                    fxm *= bX(i,j,k,n);
                    fxmc *= cX(i,j,k,n);

                }
                fx(i,j,k,n) = -dhx*fxm - dhxc*fxmc;
            }
        }
    });
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void mlebabecceclap_flux_y (Box const& box, Array4<Real> const& fy, Array4<Real const> const& apy,
                         Array4<Real const> const& fcy, Array4<Real const> const& sol,
                         Array4<Real const> const& bY, Array4<Real const> const& cY,
                         Array4<int const> const& ccm, Real dhy, Real dhyc, 
                         int face_only, int ncomp, Box const& ybox,
                         bool beta_on_centroid, bool phi_on_centroid) noexcept
{
    int lof = ybox.smallEnd(1);
    int hif = ybox.bigEnd(1);
    amrex::LoopConcurrent(box, ncomp, [=] (int i, int j, int k, int n) noexcept
    {
        if (!face_only || lof == j || hif == j) {
            if (apy(i,j,k) == 0.0) {
                fy(i,j,k,n) = 0.0;
            } else if (apy(i,j,k) == 1.0) {
                fy(i,j,k,n) = -dhy*bY(i,j,k,n)*(sol(i,j,k,n)-sol(i,j-1,k,n));
                              -dhyc*getFaceUpwindFlux(cY(i,j,k,n),sol(i,j-1,k,n),sol(i,j,k,n));
            } else {
                Real fym = bY(i,j,k,n)*(sol(i,j,k,n) - sol(i,j-1,k,n));
                Real fymc = getFaceUpwindFlux(cY(i,j,k,n),sol(i,j-1,k,n),sol(i,j,k,n));
                int ii = i + static_cast<int>(amrex::Math::copysign(1.0,fcy(i,j,k,0)));
                int kk = k + static_cast<int>(amrex::Math::copysign(1.0,fcy(i,j,k,1)));
                Real fracx = (ccm(ii,j-1,k) || ccm(ii,j,k)) ? amrex::Math::abs(fcy(i,j,k,0)) : 0.0;
                Real fracz = (ccm(i,j-1,kk) || ccm(i,j,kk)) ? amrex::Math::abs(fcy(i,j,k,1)) : 0.0;
                if (!beta_on_centroid && !phi_on_centroid)
                {
                    fym = (1.0-fracx)*(1.0-fracz)*fym +
                               fracx *(1.0-fracz)*bY(ii,j,k ,n)*(sol(ii,j,k ,n)-sol(ii,j-1,k ,n)) +
                               fracz *(1.0-fracx)*bY(i ,j,kk,n)*(sol(i ,j,kk,n)-sol(i ,j-1,kk,n)) +
                               fracx *     fracz *bY(ii,j,kk,n)*(sol(ii,j,kk,n)-sol(ii,j-1,kk,n));

                    fymc = (1.0-fracx)*(1.0-fracz)*fymc +
                               fracx *(1.0-fracz)*getFaceUpwindFlux(cY(ii,j,k ,n),sol(ii,j-1,k ,n),sol(ii,j,k ,n)) +
                               fracz *(1.0-fracx)*getFaceUpwindFlux(cY(i ,j,kk,n),sol(i ,j-1,kk,n),sol(i ,j,kk,n)) +
                               fracx *     fracz *getFaceUpwindFlux(cY(ii,j,kk,n),sol(ii,j-1,kk,n),sol(ii,j,kk,n));
                }
                else if (beta_on_centroid && !phi_on_centroid)
                {
                    fym = (1.0-fracx)*(1.0-fracz)*(sol( i,j, k,n)-sol( i,j-1, k,n)) +
                               fracx *(1.0-fracz)*(sol(ii,j, k,n)-sol(ii,j-1, k,n)) +
                               fracz *(1.0-fracx)*(sol( i,j,kk,n)-sol( i,j-1,kk,n)) +
                               fracx *     fracz *(sol(ii,j,kk,n)-sol(ii,j-1,kk,n));

                    fymc = (1.0-fracx)*(1.0-fracz)*getFaceStateUpwind(cY( i,j, k,n),sol( i,j-1, k,n),sol( i,j, k,n)) +
                                fracx *(1.0-fracz)*getFaceStateUpwind(cY(ii,j, k,n),sol(ii,j-1, k,n),sol(ii,j, k,n)) +
                                fracz *(1.0-fracx)*getFaceStateUpwind(cY( i,j,kk,n),sol( i,j-1,kk,n),sol( i,j,kk,n)) +
                                fracx *     fracz *getFaceStateUpwind(cY(ii,j,kk,n),sol(ii,j-1,kk,n),sol(ii,j,kk,n));

                    fym *= bY(i,j,k,n);
                    fymc *= cY(i,j,k,n);
                }

                fy(i,j,k,n) = -dhy*fym - dhyc*fymc;
            }
        }
    });
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void mlebabecceclap_flux_z (Box const& box, Array4<Real> const& fz, Array4<Real const> const& apz,
                         Array4<Real const> const& fcz, Array4<Real const> const& sol,
                         Array4<Real const> const& bZ, Array4<Real const> const& cZ, 
                         Array4<int const> const& ccm, Real dhz, Real dhzc, 
                         int face_only, int ncomp, Box const& zbox,
                         bool beta_on_centroid, bool phi_on_centroid) noexcept
{
    int lof = zbox.smallEnd(2);
    int hif = zbox.bigEnd(2);
    amrex::LoopConcurrent(box, ncomp, [=] (int i, int j, int k, int n) noexcept
    {
        if (!face_only || lof == k || hif == k) {
            if (apz(i,j,k) == 0.0) {
                fz(i,j,k,n) = 0.0;
            } else if (apz(i,j,k) == 1.0) {
                fz(i,j,k,n) = -dhz*bZ(i,j,k,n)*(sol(i,j,k,n)-sol(i,j,k-1,n));
                              -dhzc*getFaceUpwindFlux(cZ(i,j,k,n),sol(i,j,k-1,n),sol(i,j,k,n));
            } else {
                Real fzm = bZ(i,j,k,n)*(sol(i,j,k,n) - sol(i,j,k-1,n));
                Real fzmc = getFaceUpwindFlux(cZ(i,j,k,n),sol(i,j,k-1,n),sol(i,j,k,n));
                int ii = i + static_cast<int>(amrex::Math::copysign(1.0,fcz(i,j,k,0)));
                int jj = j + static_cast<int>(amrex::Math::copysign(1.0,fcz(i,j,k,1)));
                Real fracx = (ccm(ii,j,k-1) || ccm(ii,j,k)) ? amrex::Math::abs(fcz(i,j,k,0)) : 0.0;
                Real fracy = (ccm(i,jj,k-1) || ccm(i,jj,k)) ? amrex::Math::abs(fcz(i,j,k,1)) : 0.0;
                if (!beta_on_centroid && !phi_on_centroid)
                {
                    fzm = (1.0-fracx)*(1.0-fracy)*fzm +
                        fracx*(1.0-fracy)*bZ(ii,j ,k,n)*(sol(ii,j ,k,n)-sol(ii,j ,k-1,n)) +
                        fracy*(1.0-fracx)*bZ(i ,jj,k,n)*(sol(i ,jj,k,n)-sol(i ,jj,k-1,n)) +
                        fracx*     fracy *bZ(ii,jj,k,n)*(sol(ii,jj,k,n)-sol(ii,jj,k-1,n));

                    fzmc = (1.0-fracx)*(1.0-fracy)*fzmc +
                        fracx*(1.0-fracy)*getFaceUpwindFlux(cZ(ii,j ,k,n),sol(ii,j ,k-1,n),sol(ii,j ,k,n)) +
                        fracy*(1.0-fracx)*getFaceUpwindFlux(cZ(i ,jj,k,n),sol(i ,jj,k-1,n),sol(i ,jj,k,n)) +
                        fracx*     fracy *getFaceUpwindFlux(cZ(ii,jj,k,n),sol(ii,jj,k-1,n),sol(ii,jj,k,n));
                }
                else if (beta_on_centroid && !phi_on_centroid)
                {
                    fzm = (1.0-fracx)*(1.0-fracy)*(sol( i, j,k,n)-sol( i, j,k-1,n)) +
                               fracx *(1.0-fracy)*(sol(ii, j,k,n)-sol(ii, j,k-1,n)) +
                               fracy *(1.0-fracx)*(sol( i,jj,k,n)-sol( i,jj,k-1,n)) +
                               fracx *     fracy *(sol(ii,jj,k,n)-sol(ii,jj,k-1,n));

                    fzmc = (1.0-fracx)*(1.0-fracy)*(cZ( i, j,k,n),sol( i, j,k-1,n),sol( i, j,k,n)) +
                                fracx *(1.0-fracy)*(cZ(ii, j,k,n),sol(ii, j,k-1,n),sol(ii, j,k,n)) +
                                fracy *(1.0-fracx)*(cZ( i,jj,k,n),sol( i,jj,k-1,n),sol( i,jj,k,n)) +
                                fracx *     fracy *(cZ(ii,jj,k,n),sol(ii,jj,k-1,n),sol(ii,jj,k,n));

                    fzm *= bZ(i,j,k,n);
                    fzmc *= cZ(i,j,k,n);

                }

                fz(i,j,k,n) = -dhz*fzm - dhzc*fzmc;
            }
        }
    });
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void mlebabecceclap_flux_x_0 (Box const& box, Array4<Real> const& fx, Array4<Real const> const& apx,
                           Array4<Real const> const& sol, Array4<Real const> const& bX, Array4<Real const> const& cX,
                           Real dhx, Real dhxc, int face_only, int ncomp, Box const& xbox) noexcept
{
    int lof = xbox.smallEnd(0);
    int hif = xbox.bigEnd(0);
    amrex::LoopConcurrent(box, ncomp, [=] (int i, int j, int k, int n) noexcept
    {
        if (!face_only || lof == i || hif == i) {
            if (apx(i,j,k) == 0.0) {
                fx(i,j,k,n) = 0.0;
            } else {
                fx(i,j,k,n) = -dhx*bX(i,j,k,n)*(sol(i,j,k,n)-sol(i-1,j,k,n))
                              -dhxc*getFaceUpwindFlux(cX(i,j,k,n),sol(i-1,j,k,n),sol(i,j,k,n));
            }
        }
    });
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void mlebabecceclap_flux_y_0 (Box const& box, Array4<Real> const& fy, Array4<Real const> const& apy,
                           Array4<Real const> const& sol, Array4<Real const> const& bY, Array4<Real const> const& cY,
                           Real dhy, Real dhyc, int face_only, int ncomp, Box const& ybox) noexcept
{
    int lof = ybox.smallEnd(1);
    int hif = ybox.bigEnd(1);
    amrex::LoopConcurrent(box, ncomp, [=] (int i, int j, int k, int n) noexcept
    {
        if (!face_only || lof == j || hif == j) {
            if (apy(i,j,k) == 0.0) {
                fy(i,j,k,n) = 0.0;
            } else {
                fy(i,j,k,n) = -dhy*bY(i,j,k,n)*(sol(i,j,k,n)-sol(i,j-1,k,n))
                              -dhyc*getFaceUpwindFlux(cY(i,j,k,n),sol(i,j-1,k,n),sol(i,j,k,n));
            }
        }
    });
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void mlebabecceclap_flux_z_0 (Box const& box, Array4<Real> const& fz, Array4<Real const> const& apz,
                           Array4<Real const> const& sol, Array4<Real const> const& bZ, Array4<Real const> const& cZ,
                           Real dhz, Real dhzc, int face_only, int ncomp, Box const& zbox) noexcept
{
    int lof = zbox.smallEnd(2);
    int hif = zbox.bigEnd(2);
    amrex::LoopConcurrent(box, ncomp, [=] (int i, int j, int k, int n) noexcept
    {
        if (!face_only || lof == k || hif == k) {
            if (apz(i,j,k) == 0.0) {
                fz(i,j,k,n) = 0.0;
            } else {
                fz(i,j,k,n) = -dhz*bZ(i,j,k,n)*(sol(i,j,k,n)-sol(i,j,k-1,n))
                              -dhzc*getFaceUpwindFlux(cZ(i,j,k,n),sol(i,j,k-1,n),sol(i,j,k,n));
            }
        }
    });
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void mlebabecceclap_grad_x (Box const& box, Array4<Real> const& gx, Array4<Real const> const& sol,
                         Array4<Real const> const& apx, Array4<Real const> const& fcx,
                         Array4<int const> const& ccm,
                         Real dxi, int ncomp, bool phi_on_centroid) noexcept
{
    amrex::LoopConcurrent(box, ncomp, [=] (int i, int j, int k, int n) noexcept
    {
        if (apx(i,j,k) == 0.0) {
            gx(i,j,k,n) = 0.0;
        } else if (apx(i,j,k) == 1.0) {
            gx(i,j,k,n) = dxi*(sol(i,j,k,n)-sol(i-1,j,k,n));
        } else {
            Real gxm = (sol(i,j,k,n) - sol(i-1,j,k,n));
            int jj = j + static_cast<int>(amrex::Math::copysign(1.0, fcx(i,j,k,0)));
            int kk = k + static_cast<int>(amrex::Math::copysign(1.0, fcx(i,j,k,1)));
            Real fracy = (ccm(i-1,jj,k) || ccm(i,jj,k)) ? amrex::Math::abs(fcx(i,j,k,0)) : 0.0;
            Real fracz = (ccm(i-1,j,kk) || ccm(i,j,kk)) ? amrex::Math::abs(fcx(i,j,k,1)) : 0.0;
            if (!phi_on_centroid)
            {
                gxm = (1.0-fracy)*(1.0-fracz)*gxm +
                    fracy*(1.0-fracz)*(sol(i,jj,k ,n)-sol(i-1,jj,k ,n)) +
                    fracz*(1.0-fracy)*(sol(i,j ,kk,n)-sol(i-1,j ,kk,n)) +
                    fracy*     fracz *(sol(i,jj,kk,n)-sol(i-1,jj,kk,n));
            }
            gx(i,j,k,n) = dxi*gxm;
        }
    });
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void mlebabecceclap_grad_y (Box const& box, Array4<Real> const& gy, Array4<Real const> const& sol,
                         Array4<Real const> const& apy, Array4<Real const> const& fcy,
                         Array4<int const> const& ccm,
                         Real dyi, int ncomp, bool phi_on_centroid) noexcept
{
    amrex::LoopConcurrent(box, ncomp, [=] (int i, int j, int k, int n) noexcept
    {
        if (apy(i,j,k) == 0.0) {
            gy(i,j,k,n) = 0.0;
        } else if (apy(i,j,k) == 1.0) {
            gy(i,j,k,n) = dyi*(sol(i,j,k,n)-sol(i,j-1,k,n));
        } else {
            Real gym = (sol(i,j,k,n) - sol(i,j-1,k,n));
            int ii = i + static_cast<int>(amrex::Math::copysign(1.0,fcy(i,j,k,0)));
            int kk = k + static_cast<int>(amrex::Math::copysign(1.0,fcy(i,j,k,1)));
            Real fracx = (ccm(ii,j-1,k) || ccm(ii,j,k)) ? amrex::Math::abs(fcy(i,j,k,0)) : 0.0;
            Real fracz = (ccm(i,j-1,kk) || ccm(i,j,kk)) ? amrex::Math::abs(fcy(i,j,k,1)) : 0.0;
            if (!phi_on_centroid)
            {
                gym = (1.0-fracx)*(1.0-fracz)*gym +
                    fracx*(1.0-fracz)*(sol(ii,j,k ,n)-sol(ii,j-1,k ,n)) +
                    fracz*(1.0-fracx)*(sol(i ,j,kk,n)-sol(i ,j-1,kk,n)) +
                    fracx*     fracz *(sol(ii,j,kk,n)-sol(ii,j-1,kk,n));
            }
            gy(i,j,k,n) = dyi*gym;
        }
    });
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void mlebabecceclap_grad_z (Box const& box, Array4<Real> const& gz, Array4<Real const> const& sol,
                         Array4<Real const> const& apz, Array4<Real const> const& fcz,
                         Array4<int const> const& ccm,
                         Real dzi, int ncomp, bool phi_on_centroid) noexcept
{
    amrex::LoopConcurrent(box, ncomp, [=] (int i, int j, int k, int n) noexcept
    {
        if (apz(i,j,k) == 0.0) {
            gz(i,j,k,n) = 0.0;
        } else if (apz(i,j,k) == 1.0) {
            gz(i,j,k,n) = dzi*(sol(i,j,k,n)-sol(i,j,k-1,n));
        } else {
            Real gzm = (sol(i,j,k,n) - sol(i,j,k-1,n));
            int ii = i + static_cast<int>(amrex::Math::copysign(1.0,fcz(i,j,k,0)));
            int jj = j + static_cast<int>(amrex::Math::copysign(1.0,fcz(i,j,k,1)));
            Real fracx = (ccm(ii,j,k-1) || ccm(ii,j,k)) ? amrex::Math::abs(fcz(i,j,k,0)) : 0.0;
            Real fracy = (ccm(i,jj,k-1) || ccm(i,jj,k)) ? amrex::Math::abs(fcz(i,j,k,1)) : 0.0;
            if (!phi_on_centroid)
            {
                gzm = (1.0-fracx)*(1.0-fracy)*gzm +
                    fracx*(1.0-fracy)*(sol(ii,j ,k,n)-sol(ii,j ,k-1,n)) +
                    fracy*(1.0-fracx)*(sol(i ,jj,k,n)-sol(i ,jj,k-1,n)) +
                    fracx*     fracy *(sol(ii,jj,k,n)-sol(ii,jj,k-1,n));
            }
            gz(i,j,k,n) = dzi*gzm;
        }
    });
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void mlebabecceclap_grad_x_0 (Box const& box, Array4<Real> const& gx, Array4<Real const> const& sol,
                           Array4<Real const> const& apx, Real dxi, int ncomp) noexcept
{
    amrex::LoopConcurrent(box, ncomp, [=] (int i, int j, int k, int n) noexcept
    {
        if (apx(i,j,k) == 0.0) {
            gx(i,j,k,n) = 0.0;
        } else {
            gx(i,j,k,n) = dxi*(sol(i,j,k,n)-sol(i-1,j,k,n));
        }
    });
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void mlebabecceclap_grad_y_0 (Box const& box, Array4<Real> const& gy, Array4<Real const> const& sol,
                           Array4<Real const> const& apy, Real dyi, int ncomp) noexcept
{
    amrex::LoopConcurrent(box, ncomp, [=] (int i, int j, int k, int n) noexcept
    {
        if (apy(i,j,k) == 0.0) {
            gy(i,j,k,n) = 0.0;
        } else {
            gy(i,j,k,n) = dyi*(sol(i,j,k,n)-sol(i,j-1,k,n));
        }
    });
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void mlebabecceclap_grad_z_0 (Box const& box, Array4<Real> const& gz, Array4<Real const> const& sol,
                           Array4<Real const> const& apz, Real dzi, int ncomp) noexcept
{
    amrex::LoopConcurrent(box, ncomp, [=] (int i, int j, int k, int n) noexcept
    {
        if (apz(i,j,k) == 0.0) {
            gz(i,j,k,n) = 0.0;
        } else {
            gz(i,j,k,n) = dzi*(sol(i,j,k,n)-sol(i,j,k-1,n));
        }
    });
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void mlebabecceclap_normalize (Box const& box, Array4<Real> const& phi,
                            Real alpha, Array4<Real const> const& a,
                            Real dhx, Real dhy, Real dhz,
                            Real dhxc, Real dhyc, Real dhzc,
                            Array4<Real const> const& bX, Array4<Real const> const& bY,
                            Array4<Real const> const& bZ,
                            Array4<Real const> const& cX, Array4<Real const> const& cY,
                            Array4<Real const> const& cZ,
                            Array4<const int> const& ccm, Array4<EBCellFlag const> const& flag,
                            Array4<Real const> const& vfrc,
                            Array4<Real const> const& apx, Array4<Real const> const& apy,
                            Array4<Real const> const& apz,
                            Array4<Real const> const& fcx, Array4<Real const> const& fcy,
                            Array4<Real const> const& fcz,
                            Array4<Real const> const& ba, Array4<Real const> const& bc,
                            Array4<Real const> const& beb,
                            bool is_dirichlet, bool beta_on_centroid, int ncomp) noexcept
{
    amrex::Loop(box, ncomp, [=] (int i, int j, int k, int n) noexcept
    {
        if (flag(i,j,k).isRegular())
        {
            phi(i,j,k,n) /= alpha*a(i,j,k) + dhx*(bX(i,j,k,n) + bX(i+1,j,k,n))
                                           + dhy*(bY(i,j,k,n) + bY(i,j+1,k,n))
                                           + dhz*(bZ(i,j,k,n) + bZ(i,j,k+1,n));
                                           - dhxc*getCellUpwindCoeff(cX(i,j,k,n),cX(i+1,j,k,n))
                                           - dhyc*getCellUpwindCoeff(cY(i,j,k,n),cY(i,j+1,k,n))
                                           - dhzc*getCellUpwindCoeff(cZ(i,j,k,n),cZ(i,j,k+1,n));
        }
        else if (flag(i,j,k).isSingleValued())
        {
            Real kappa = vfrc(i,j,k);
            Real apxm = apx(i,j,k);
            Real apxp = apx(i+1,j,k);
            Real apym = apy(i,j,k);
            Real apyp = apy(i,j+1,k);
            Real apzm = apz(i,j,k);
            Real apzp = apz(i,j,k+1);

            Real sxm =  bX(i,j,k,n);
            Real sxmc =  getCellUpwindCoeffLeft(cX(i,j,k,n), cX(i+1,j,k,n));
            if (apxm != 0.0 && apxm != 1.0 && !beta_on_centroid) {
                int jj = j + static_cast<int>(amrex::Math::copysign(1.0, fcx(i,j,k,0)));
                int kk = k + static_cast<int>(amrex::Math::copysign(1.0, fcx(i,j,k,1)));
                Real fracy = (ccm(i-1,jj,k) || ccm(i,jj,k))
                    ? amrex::Math::abs(fcx(i,j,k,0)) : 0.0;
                Real fracz = (ccm(i-1,j,kk) || ccm(i,j,kk))
                    ? amrex::Math::abs(fcx(i,j,k,1)) : 0.0;
                sxm = (1.0-fracy)*(1.0-fracz)*sxm;
                // TODO: IS THIS CORRECT? 
                sxmc = (1.0-fracy)*(1.0-fracz)*sxmc;
            }

            Real sxp = -bX(i+1,j,k,n);
            Real sxpc = -getCellUpwindCoeffRight(cX(i,j,k,n), cX(i+1,j,k,n));
            if (apxp != 0.0 && apxp != 1.0 && !beta_on_centroid) {
                int jj = j + static_cast<int>(amrex::Math::copysign(1.0,fcx(i+1,j,k,0)));
                int kk = k + static_cast<int>(amrex::Math::copysign(1.0,fcx(i+1,j,k,1)));
                Real fracy = (ccm(i,jj,k) || ccm(i+1,jj,k))
                    ? amrex::Math::abs(fcx(i+1,j,k,0)) : 0.0;
                Real fracz = (ccm(i,j,kk) || ccm(i+1,j,kk))
                    ? amrex::Math::abs(fcx(i+1,j,k,1)) : 0.0;
                sxp = (1.0-fracy)*(1.0-fracz)*sxp;
                sxpc = (1.0-fracy)*(1.0-fracz)*sxpc;
            }

            Real sym =  bY(i,j,k,n);
            Real symc =  getCellUpwindCoeffLeft(cY(i,j,k,n), cY(i,j+1,k,n));
            if (apym != 0.0 && apym != 1.0 && !beta_on_centroid) {
                int ii = i + static_cast<int>(amrex::Math::copysign(1.0,fcy(i,j,k,0)));
                int kk = k + static_cast<int>(amrex::Math::copysign(1.0,fcy(i,j,k,1)));
                Real fracx = (ccm(ii,j-1,k) || ccm(ii,j,k))
                    ? amrex::Math::abs(fcy(i,j,k,0)) : 0.0;
                Real fracz = (ccm(i,j-1,kk) || ccm(i,j,kk))
                    ? amrex::Math::abs(fcy(i,j,k,1)) : 0.0;
                sym = (1.0-fracx)*(1.0-fracz)*sym;
                symc = (1.0-fracx)*(1.0-fracz)*symc;
            }

            Real syp = -bY(i,j+1,k,n);
            Real sypc = -getCellUpwindCoeffRight(cY(i,j,k,n), cY(i,j+1,k,n));
            if (apyp != 0.0 && apyp != 1.0 && !beta_on_centroid) {
                int ii = i + static_cast<int>(amrex::Math::copysign(1.0,fcy(i,j+1,k,0)));
                int kk = k + static_cast<int>(amrex::Math::copysign(1.0,fcy(i,j+1,k,1)));
                Real fracx = (ccm(ii,j,k) || ccm(ii,j+1,k))
                    ? amrex::Math::abs(fcy(i,j+1,k,0)) : 0.0;
                Real fracz = (ccm(i,j,kk) || ccm(i,j+1,kk))
                    ? amrex::Math::abs(fcy(i,j+1,k,1)) : 0.0;
                syp = (1.0-fracx)*(1.0-fracz)*syp;
                sypc = (1.0-fracx)*(1.0-fracz)*sypc;
            }

            Real szm =  bZ(i,j,k,n);
            Real szmc =  getCellUpwindCoeffLeft(cZ(i,j,k,n), cZ(i,j,k+1,n));
            if (apzm != 0.0 && apzm != 1.0 && !beta_on_centroid) {
                int ii = i + static_cast<int>(amrex::Math::copysign(1.0,fcz(i,j,k,0)));
                int jj = j + static_cast<int>(amrex::Math::copysign(1.0,fcz(i,j,k,1)));
                Real fracx = (ccm(ii,j,k-1) || ccm(ii,j,k))
                    ? amrex::Math::abs(fcz(i,j,k,0)) : 0.0;
                Real fracy = (ccm(i,jj,k-1) || ccm(i,jj,k))
                    ? amrex::Math::abs(fcz(i,j,k,1)) : 0.0;
                szm = (1.0-fracx)*(1.0-fracy)*szm;
                szmc = (1.0-fracx)*(1.0-fracy)*szmc;
            }

            Real szp = -bZ(i,j,k+1,n);
            Real szpc = -getCellUpwindCoeffRight(cZ(i,j,k,n), cZ(i,j,k+1,n));
            if (apzp != 0.0 && apzp != 1.0 && !beta_on_centroid) {
                int ii = i + static_cast<int>(amrex::Math::copysign(1.0,fcz(i,j,k+1,0)));
                int jj = j + static_cast<int>(amrex::Math::copysign(1.0,fcz(i,j,k+1,1)));
                Real fracx = (ccm(ii,j,k) || ccm(ii,j,k+1))
                    ? amrex::Math::abs(fcz(i,j,k+1,0)) : 0.0;
                Real fracy = (ccm(i,jj,k) || ccm(i,jj,k+1))
                    ? amrex::Math::abs(fcz(i,j,k+1,1)) : 0.0;
                szp = (1.0-fracx)*(1.0-fracy)*szp;
                szpc = (1.0-fracx)*(1.0-fracy)*szpc;
            }

            Real vfrcinv = 1.0/kappa;
            Real gamma = alpha*a(i,j,k) + vfrcinv *
                (dhx*(apxm*sxm-apxp*sxp) +
                 dhy*(apym*sym-apyp*syp) +
                 dhz*(apzm*szm-apzp*szp) - 
                 dhxc*(apxm*sxmc-apxp*sxpc) -  
                 dhyc*(apym*symc-apyp*sypc) -  
                 dhzc*(apzm*szmc-apzp*szpc));

            if (is_dirichlet) {
                Real dapx = apxm-apxp;
                Real dapy = apym-apyp;
                Real dapz = apzm-apzp;
                Real anorm = std::sqrt(dapx*dapx+dapy*dapy+dapz*dapz);
                Real anorminv = 1.0/anorm;
                Real anrmx = dapx * anorminv;
                Real anrmy = dapy * anorminv;
                Real anrmz = dapz * anorminv;
                Real bctx = bc(i,j,k,0);
                Real bcty = bc(i,j,k,1);
                Real bctz = bc(i,j,k,2);
                Real dx_eb = get_dx_eb(vfrc(i,j,k));

                Real dg = dx_eb / amrex::max(amrex::Math::abs(anrmx),amrex::Math::abs(anrmy),
                                             amrex::Math::abs(anrmz));

                Real gx = bctx - dg*anrmx;
                Real gy = bcty - dg*anrmy;
                Real gz = bctz - dg*anrmz;
                Real sx = amrex::Math::copysign(1.0,anrmx);
                Real sy = amrex::Math::copysign(1.0,anrmy);
                Real sz = amrex::Math::copysign(1.0,anrmz);

                gx *= sx;
                gy *= sy;
                gz *= sz;
                Real gxy = gx*gy;
                Real gxz = gx*gz;
                Real gyz = gy*gz;
                Real gxyz = gx*gy*gz;
                Real phig_gamma = (1.0+gx+gy+gz+gxy+gxz+gyz+gxyz);
                Real feb_gamma = -phig_gamma/dg * ba(i,j,k) * beb(i,j,k,n);
                gamma += vfrcinv*(-dhx)*feb_gamma;
            }

            phi(i,j,k,n) /= gamma;
        }
    });
}

}

#endif
