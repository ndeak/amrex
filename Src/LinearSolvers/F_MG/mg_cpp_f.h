#ifndef _MG_CPP_F_H_
#define _MG_CPP_F_H_

#if defined(BL_FORT_USE_UPPERCASE)
#define mgt_alloc                 MGT_ALLOC
#define mgt_set_level             MGT_SET_LEVEL
#define mgt_finalize              MGT_FINALIZE
#define mgt_init_coeffs_lev       MGT_INIT_COEFFS_LEV
#define mgt_finalize_stencil      MGT_FINALIZE_STENCIL
#define mgt_finalize_stencil_lev  MGT_FINALIZE_STENCIL_LEV
#define mgt_set_rh_1d             MGT_SET_RH_1D
#define mgt_get_rh_1d             MGT_GET_RH_1D
#define mgt_set_uu_1d             MGT_SET_UU_1D
#define mgt_get_uu_1d             MGT_GET_UU_1D
#define mgt_set_cfa_1d            MGT_SET_CFA_1D
#define mgt_set_cfbx_1d           MGT_SET_CFBX_1D
#define mgt_set_rh_2d             MGT_SET_RH_2D
#define mgt_get_rh_2d             MGT_GET_RH_2D
#define mgt_set_uu_2d             MGT_SET_UU_2D
#define mgt_get_uu_2d             MGT_GET_UU_2D
#define mgt_set_cfa_2d            MGT_SET_CFA_2D
#define mgt_set_cfbx_2d           MGT_SET_CFBX_2D
#define mgt_set_cfby_2d           MGT_SET_CFBY_2D
#define mgt_set_rh_3d             MGT_SET_RH_3D
#define mgt_get_rh_3d             MGT_GET_RH_3D
#define mgt_set_uu_3d             MGT_SET_UU_3D
#define mgt_get_uu_3d             MGT_GET_UU_3D
#define mgt_set_cfa_3d            MGT_SET_CFA_3D
#define mgt_set_cfbx_3d           MGT_SET_CFBX_3D
#define mgt_set_cfby_3d           MGT_SET_CFBY_3D
#define mgt_set_cfbz_3d           MGT_SET_CFBZ_3D
#define mgt_dealloc               MGT_DEALLOC
#define mgt_solve                 MGT_SOLVE
#define mgt_set_nu1               MGT_SET_NU1
#define mgt_set_nu2               MGT_SET_NU2
#define mgt_set_eps               MGT_SET_EPS
#define mgt_set_bottom_solver     MGT_SET_BOTTOM_SOLVER
#define mgt_set_bottom_max_iter   MGT_SET_BOTTOM_MAX_ITER
#define mgt_set_bottom_solver_eps MGT_SET_BOTTOM_SOLVER_EPS
#define mgt_set_max_nlevel        MGT_SET_MAX_NLEVEL
#define mgt_set_verbose           MGT_SET_VERBOSE
#define mgt_get_ipar_defaults     MGT_GET_IPAR_DEFAULTS
#define mgt_get_rpar_defaults     MGT_GET_RPAR_DEFAULTS
#elif defined(BL_FORT_USE_UNDERSCORE)
#define mgt_alloc                 mgt_alloc_
#define mgt_set_level             mgt_set_level_
#define mgt_finalize              mgt_finalize_
#define mgt_init_coeffs_lev       mgt_init_coeffs_lev_
#define mgt_finalize_stencil      mgt_finalize_stencil_
#define mgt_finalize_stencil_lev  mgt_finalize_stencil_lev_
#define mgt_set_rh_1d             mgt_set_rh_1d_
#define mgt_set_cfa_1d            mgt_set_cfa_1d_
#define mgt_set_cfbx_1d           mgt_set_cfbx_1d_
#define mgt_get_rh_1d             mgt_get_rh_1d_
#define mgt_set_uu_1d             mgt_set_uu_1d_
#define mgt_get_uu_1d             mgt_get_uu_1d_
#define mgt_set_rh_2d             mgt_set_rh_2d_
#define mgt_get_rh_2d             mgt_get_rh_2d_
#define mgt_set_uu_2d             mgt_set_uu_2d_
#define mgt_get_uu_2d             mgt_get_uu_2d_
#define mgt_set_cfa_2d            mgt_set_cfa_2d_
#define mgt_set_cfbx_2d           mgt_set_cfbx_2d_
#define mgt_set_cfby_2d           mgt_set_cfby_2d_
#define mgt_get_cfa_2d            mgt_get_cfa_2d_
#define mgt_get_cfbx_2d           mgt_get_cfbx_2d_
#define mgt_get_cfby_2d           mgt_get_cfby_2d_
#define mgt_set_rh_3d             mgt_set_rh_3d_
#define mgt_get_rh_3d             mgt_get_rh_3d_
#define mgt_set_uu_3d             mgt_set_uu_3d_
#define mgt_get_uu_3d             mgt_get_uu_3d_
#define mgt_set_cfa_3d            mgt_set_cfa_3d_
#define mgt_set_cfbx_3d           mgt_set_cfbx_3d_
#define mgt_set_cfby_3d           mgt_set_cfby_3d_
#define mgt_set_cfbz_3d           mgt_set_cfbz_3d_
#define mgt_get_cfa_3d            mgt_get_cfa_3d_
#define mgt_get_cfbx_3d           mgt_get_cfbx_3d_
#define mgt_get_cfby_3d           mgt_get_cfby_3d_
#define mgt_get_cfbz_3d           mgt_get_cfbz_3d_
#define mgt_dealloc               mgt_dealloc_
#define mgt_solve                 mgt_solve_
#define mgt_set_nu1               mgt_set_nu1_
#define mgt_set_nu2               mgt_set_nu2_
#define mgt_set_eps               mgt_set_eps_
#define mgt_set_bottom_solver     mgt_set_bottom_solver_
#define mgt_set_bottom_max_iter   mgt_set_bottom_max_iter_
#define mgt_set_bottom_solver_eps mgt_set_bottom_solver_eps_
#define mgt_set_max_nlevel        mgt_set_max_nlevel_
#define mgt_set_min_width         mgt_set_min_width_
#define mgt_set_verbose           mgt_set_verbose_
#define mgt_get_ipar_defaults     mgt_get_ipar_defaults_
#define mgt_get_rpar_defaults     mgt_get_rpar_defaults_
#elif defined(BL_FORT_USE_DBL_UNDERSCORE)
#define mgt_alloc                 mgt_alloc__
#define mgt_set_level             mgt_set_level__
#define mgt_finalize              mgt_finalize__
#define mgt_init_coeffs_lev       mgt_init_coeffs_lev__
#define mgt_finalize_stencil      mgt_finalize_stencil__
#define mgt_finalize_stencil_lev  mgt_finalize_stencil_lev__
#define mgt_set_rh_1d             mgt_set_rh_1d__
#define mgt_get_rh_1d             mgt_get_rh_1d__
#define mgt_set_uu_1d             mgt_set_uu_1d__
#define mgt_get_uu_1d             mgt_get_uu_1d__
#define mgt_set_cf_1d             mgt_set_cf_1d__
#define mgt_set_cfa_1d            mgt_set_cfa_1d__
#define mgt_set_cfbx_1d           mgt_set_cfbx_1d__
#define mgt_set_rh_2d             mgt_set_rh_2d__
#define mgt_get_rh_2d             mgt_get_rh_2d__
#define mgt_set_uu_2d             mgt_set_uu_2d__
#define mgt_get_uu_2d             mgt_get_uu_2d__
#define mgt_set_cfa_2d            mgt_set_cfa_2d__
#define mgt_set_cfbx_2d           mgt_set_cfbx_2d__
#define mgt_set_cfby_2d           mgt_set_cfby_2d__
#define mgt_set_rh_3d             mgt_set_rh_3d__
#define mgt_get_rh_3d             mgt_get_rh_3d__
#define mgt_set_uu_3d             mgt_set_uu_3d__
#define mgt_get_uu_3d             mgt_get_uu_3d__
#define mgt_set_cfa_3d            mgt_set_cfa_3d__
#define mgt_set_cfbx_3d           mgt_set_cfbx_3d__
#define mgt_set_cfby_3d           mgt_set_cfby_3d__
#define mgt_set_cfbz_3d           mgt_set_cfbz_3d__
#define mgt_dealloc               mgt_dealloc__
#define mgt_solve                 mgt_solve__
#define mgt_set_nu1               mgt_set_nu1__
#define mgt_set_nu2               mgt_set_nu2__
#define mgt_set_eps               mgt_set_eps__
#define mgt_set_bottom_solver     mgt_set_bottom_solver__
#define mgt_set_bottom_max_iter   mgt_set_bottom_max_iter__
#define mgt_set_bottom_solver_eps mgt_set_bottom_solver_eps__
#define mgt_set_max_nlevel        mgt_set_max_nlevel__
#define mgt_set_min_width         mgt_set_min_width__
#define mgt_set_verbose           mgt_set_verbose__
#define mgt_get_ipar_defaults     mgt_get_ipar_defaults__
#define mgt_get_rpar_defaults     mgt_get_rpar_defaults__
#endif

#ifdef __cplusplus
extern "C" 
{
#endif

  /* The contants match the ones in 'bc.f90'; care is needed to ensure 
     that the continue to match. */

  const int MGT_BC_PER = -1;	/* Periodic  */
  const int MGT_BC_INT =  0;	/* Interior  */
  const int MGT_BC_DIR =  1;	/* Dirichlet */
  const int MGT_BC_NEU =  2;	/* Neumann   */

  void mgt_alloc(int* mgt, const int* dm, const int* nlevel, const int* nodal);

  void mgt_set_level(const int* mgt, const int* lev, const int* nb, const int* dm, 
		     const int* lo, const int* hi, 
		     const int* pd_lo, const int* pd_hi, 
		     const int* bc, const int* pm,
		     const int* pmap);

  void mgt_finalize(const int* mgt, const int* nipar, const int* ipar, 
                    const int* nrpar, const double* rpar, const double* dx);

  void mgt_init_coeffs_lev(const int* mgt, const int* lev);
  
  void mgt_finalize_stencil(const int* mgt);
  
  void mgt_finalize_stencil_lev(const int* mgt, const int* lev,
			    const double* xa, const double* xb,
			    const double* pxa, const double* pxbb);

  void mgt_set_rh_1d(const int* mgt, const int* lev, const int* n, const double* rh, 
		     const int* plo, const int* phi, 
		     const int* lo, const int* hi);
  
  void mgt_get_rh_1d(const int* mgt, const int* lev, const int* n, double* rh, 
		     const int* plo, const int* phi, 
		     const int* lo, const int* hi);
  
  void mgt_set_cfa_1d(const int* mgt, const int* lev, const int* n, const double* cf,
		      const int* plo, const int* phi, 
		      const int* lo, const int* hi);
  
  void mgt_set_cfbx_1d(const int* mgt, const int* lev, const int* n, const double* cf,
		       const int* plo, const int* phi, 
		       const int* lo, const int* hi);
  
  void mgt_set_uu_1d(const int* mgt, const int* lev, const int* n, const double* uu, 
		     const int* plo, const int* phi, 
		     const int* lo, const int* hi);
  
  void mgt_get_uu_1d(const int* mgt, const int* lev, const int* n, double* uu, 
		     const int* plo, const int* phi, 
		     const int* lo, const int* hi);
  
  void mgt_set_rh_2d(const int* mgt, const int* lev, const int* n, const double* rh, 
		     const int* plo, const int* phi, 
		     const int* lo, const int* hi);
  
  void mgt_get_rh_2d(const int* mgt, const int* lev, const int* n, double* rh, 
		     const int* plo, const int* phi, 
		     const int* lo, const int* hi);
  
  void mgt_set_uu_2d(const int* mgt, const int* lev, const int* n, const double* uu, 
		     const int* plo, const int* phi, 
		     const int* lo, const int* hi);
  
  void mgt_set_cfa_2d(const int* mgt, const int* lev, const int* n, const double* cf,
		      const int* plo, const int* phi, 
		      const int* lo, const int* hi);
  
  void mgt_set_cfbx_2d(const int* mgt, const int* lev, const int* n, const double* cf,
		       const int* plo, const int* phi, 
		       const int* lo, const int* hi);
  
  void mgt_set_cfby_2d(const int* mgt, const int* lev, const int* n, const double* cf,
		       const int* plo, const int* phi, 
		       const int* lo, const int* hi);
  
  void mgt_get_uu_2d(const int* mgt, const int* lev, const int* n, double* uu, 
		     const int* plo, const int* phi, 
		     const int* lo, const int* hi);
  
  void mgt_set_rh_3d(const int* mgt, const int* lev, const int* n, const double* rh, 
		     const int* plo, const int* phi, 
		     const int* lo, const int* hi);
  
  void mgt_get_rh_3d(const int* mgt, const int* lev, const int* n, double* rh, 
		     const int* plo, const int* phi, 
		     const int* lo, const int* hi);
  
  void mgt_set_cfa_3d(const int* mgt, const int* lev, const int* n, const double* cf,
		      const int* plo, const int* phi, 
		      const int* lo, const int* hi);
  
  void mgt_set_cfbx_3d(const int* mgt, const int* lev, const int* n, const double* cf,
		       const int* plo, const int* phi, 
		       const int* lo, const int* hi);
  
  void mgt_set_cfby_3d(const int* mgt, const int* lev, const int* n, const double* cf,
		       const int* plo, const int* phi, 
		       const int* lo, const int* hi);
  
  void mgt_set_cfbz_3d(const int* mgt, const int* lev, const int* n, const double* cf,
		       const int* plo, const int* phi, 
		       const int* lo, const int* hi);
  
  void mgt_set_uu_3d(const int* mgt, const int* lev, const int* n, const double* uu, 
		     const int* plo, const int* phi, 
		     const int* lo, const int* hi);
  
  void mgt_get_uu_3d(const int* mgt, const int* lev, const int* n, double* uu, 
		     const int* plo, const int* phi, 
		     const int* lo, const int* hi);

  void mgt_dealloc(const int* mgt);
  
  void mgt_solve(const int* mgt, const double& tol, const double& abs_tol);
  
  void mgt_set_nu1(const int* mgt, const int* nu1);
  
  void mgt_set_nu2(const int* mgt, const int* nu2);
  
  void mgt_set_eps(const int* mgt, const double* eps);
  
  void mgt_set_bottom_solver(const int* mgt, const int* bottom_solver);
  
  void mgt_set_bottom_max_iter(const int* mgt, const int* bottom_max_iter);
  
  void mgt_set_bottom_solver_eps(const int* mgt, const double* bottom_solver_eps);
  
  void mgt_set_max_nlevel(const int* mgt, const int* max_nlevel);
  
  void mgt_set_min_width(const int* mgt, const int* min_width);

  void mgt_set_verbose(const int* mgt, const int* verbose);
  
  void mgt_get_ipar_defaults(int* ipar, const int* nipar);

  void mgt_get_rpar_defaults(double* rpar, const int* nipar);
  
  
#ifdef __cplusplus
}
#endif

#endif
