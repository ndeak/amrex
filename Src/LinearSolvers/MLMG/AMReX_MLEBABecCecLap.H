#ifndef AMREX_MLEBABECCECLAP_H_
#define AMREX_MLEBABECCECLAP_H_
#include <AMReX_Config.H>

#include <AMReX_EBFabFactory.H>
#include <AMReX_MLCellABecCecLap.H>
#include <AMReX_Array.H>
#include <limits>

namespace amrex {

// (alpha * a - beta * (del dot b grad) - eta * (del dot c)) phi

class MLEBABecCecLap
    : public MLCellABecCecLap
{
public:

    MLEBABecCecLap () {}
    MLEBABecCecLap (const Vector<Geometry>& a_geom,
                 const Vector<BoxArray>& a_grids,
                 const Vector<DistributionMapping>& a_dmap,
                 const LPInfo& a_info,
                 const Vector<EBFArrayBoxFactory const*>& a_factory,
                 const int a_ncomp = 1);

    virtual ~MLEBABecCecLap ();

    MLEBABecCecLap (const MLEBABecCecLap&) = delete;
    MLEBABecCecLap (MLEBABecCecLap&&) = delete;
    MLEBABecCecLap& operator= (const MLEBABecCecLap&) = delete;
    MLEBABecCecLap& operator= (MLEBABecCecLap&&) = delete;

    // TODO: Need to include extra define function (see how define is fleshed out in MLCellABecLap.cpp but not Cec counterpart)
    void define (const Vector<Geometry>& a_geom,
                 const Vector<BoxArray>& a_grids,
                 const Vector<DistributionMapping>& a_dmap,
                 const LPInfo& a_info,
                 const Vector<EBFArrayBoxFactory const*>& a_factory);

    void setPhiOnCentroid ();

    void setScalars (Real a, Real b, Real c);
    void setRelaxation (Real omega) noexcept;
    void setACoeffs (int amrlev, const MultiFab& alpha);
    void setACoeffs (int amrlev, Real alpha);

    void setBCoeffs (int amrlev, const Array<MultiFab const*,AMREX_SPACEDIM>& beta,
                     Location a_beta_loc);
    void setBCoeffs (int amrlev, const Array<MultiFab const*,AMREX_SPACEDIM>& beta)
        {setBCoeffs (amrlev, beta, Location::FaceCenter);}
    void setBCoeffs (int amrlev, Real beta);
    void setBCoeffs (int amrlev, Vector<Real> const& beta);

    void setCCoeffs (int amrlev, const Array<MultiFab const*,AMREX_SPACEDIM>& eta,
                     Location a_eta_loc);
    void setCCoeffs (int amrlev, const Array<MultiFab const*,AMREX_SPACEDIM>& eta)
        {setCCoeffs (amrlev, eta, Location::FaceCenter);}
    void setCCoeffs (int amrlev, Real eta);
    void setCCoeffs (int amrlev, Vector<Real> const& eta);

    // TODO: what is this for, do we need it?
    void checkDiagonalDominance(int amrlev, int mglev);
    void getDiagonal(MultiFab& diag);
    void getNetFaceCoeff(MultiFab& coeff_d, int dir);

    // Tells the solver that EB boundaries have Dirichlet bc's specified by "phi"
    void setEBDirichlet      (int amrlev, const MultiFab& phi, const MultiFab& beta);
    void setEBDirichlet      (int amrlev, const MultiFab& phi, Real beta);
    void setEBDirichlet      (int amrlev, const MultiFab& phi, Vector<Real> const& beta);

    // Tells the solver that EB boundaries have homogeneous Dirichlet bc's
    void setEBHomogDirichlet (int amrlev,                      const MultiFab& beta);
    void setEBHomogDirichlet (int amrlev,                      Real beta);
    void setEBHomogDirichlet (int amrlev,                      Vector<Real> const& beta);

    virtual int getNComp () const override { return m_ncomp; }

    virtual bool needsUpdate () const override {
        return (m_needs_update || MLCellABecCecLap::needsUpdate());
    }
    virtual void update () override;

    virtual std::unique_ptr<FabFactory<FArrayBox> > makeFactory (int amrlev, int mglev) const final override;

    virtual bool isCrossStencil () const override { return false; }

    virtual void applyBC (int amrlev, int mglev, MultiFab& in, BCMode bc_mode, StateMode s_mode,
                          const MLMGBndry* bndry=nullptr, bool skip_fillboundary=false) const final override;
    virtual void apply (int amrlev, int mglev, MultiFab& out, MultiFab& in, BCMode bc_mode,
                        StateMode s_mode, const MLMGBndry* bndry=nullptr) const override;
    virtual void compGrad (int amrlev, const Array<MultiFab*,AMREX_SPACEDIM>& grad,
                           MultiFab& sol, Location loc) const final override;

    virtual void prepareForSolve () override;
    virtual bool isSingular (int amrlev) const override { return m_is_singular[amrlev]; }
    virtual bool isBottomSingular () const override { return m_is_singular[0]; }
    virtual void Fapply (int amrlev, int mglev, MultiFab& out, const MultiFab& in) const final override;
    virtual void Fsmooth (int amrlev, int mglev, MultiFab& sol, const MultiFab& rhs, int redblack) const final override;
    virtual void FFlux (int amrlev, const MFIter& mfi,
                        const Array<FArrayBox*,AMREX_SPACEDIM>& flux,
                        const FArrayBox& sol, Location loc,
                        const int face_only=0) const final override;

    virtual void normalize (int amrlev, int mglev, MultiFab& mf) const final override;

    virtual Real getAScalar () const final override { return m_a_scalar; }
    virtual Real getBScalar () const final override { return m_b_scalar; }
    virtual Real getCScalar () const final override { return m_c_scalar; }
    virtual MultiFab const* getACoeffs (int amrlev, int mglev) const final override
        { return &(m_a_coeffs[amrlev][mglev]); }
    virtual Array<MultiFab const*,AMREX_SPACEDIM> getBCoeffs (int amrlev, int mglev) const final override
        { return amrex::GetArrOfConstPtrs(m_b_coeffs[amrlev][mglev]); }
    virtual Array<MultiFab const*,AMREX_SPACEDIM> getCCoeffs (int amrlev, int mglev) const final override
        { return amrex::GetArrOfConstPtrs(m_c_coeffs[amrlev][mglev]); }

    virtual std::unique_ptr<MLLinOp> makeNLinOp (int /*grid_size*/) const final override {
        amrex::Abort("MLABecCecLap::makeNLinOp: Not implmented");
        return std::unique_ptr<MLLinOp>{};
    }
  
    // TODO: needed?
    // virtual void applyOverset (int amlev, MultiFab& rhs) const override;

    virtual void restriction (int, int, MultiFab& crse, MultiFab& fine) const final override;

    virtual void interpolation (int amrlev, int fmglev, MultiFab& fine, const MultiFab& crse) const final override;

    virtual void averageDownSolutionRHS (int camrlev, MultiFab& crse_sol, MultiFab& crse_rhs,
                                         const MultiFab& fine_sol, const MultiFab& fine_rhs) final override;

    virtual void getEBFluxes (const Vector<MultiFab*>& a_flux,
                              const Vector<MultiFab*>& a_sol) const override;

#if defined(AMREX_USE_HYPRE) && (AMREX_SPACEDIM > 1)
    virtual std::unique_ptr<Hypre> makeHypre (Hypre::Interface hypre_interface) const override;
#endif

#ifdef AMREX_USE_PETSC
    virtual std::unique_ptr<PETScABecCecLap> makePETSc () const override;
#endif

protected:

    int m_ncomp = 1;

    bool m_needs_update = true;

    // FIXME: for now - assume B and C coefficients both at same location?
    Location m_beta_loc; // Location of coefficients: face centers or face centroids
    Location m_eta_loc; // Location of coefficients: face centers or face centroids
    Location m_phi_loc;  // Location of solution variable: cell centers or cell centroids

    Real m_a_scalar = std::numeric_limits<Real>::quiet_NaN();
    Real m_b_scalar = std::numeric_limits<Real>::quiet_NaN();
    Real m_c_scalar = std::numeric_limits<Real>::quiet_NaN();
    Real m_omega    = std::numeric_limits<Real>::quiet_NaN();

    Vector<Vector<MultiFab> > m_a_coeffs;
    Vector<Vector<Array<MultiFab,AMREX_SPACEDIM> > > m_b_coeffs;
    Vector<Vector<Array<MultiFab,AMREX_SPACEDIM> > > m_c_coeffs;
    Vector<Vector<iMultiFab> > m_cc_mask;

    Vector<std::unique_ptr<MultiFab> > m_eb_phi;
    Vector<Vector<std::unique_ptr<MultiFab> > > m_eb_b_coeffs;

    Vector<int> m_is_singular;

    mutable int m_is_eb_inhomog;

    //
    // functions
    //
    bool isEBDirichlet   () const noexcept { return m_eb_phi[0] != nullptr; }

    // void averageDownCoeffsSameAmrLevel (int amrlev, Vector<MultiFab>& a,
    //                                     Vector<Array<MultiFab,AMREX_SPACEDIM> >& b,
    //                                     const Vector<MultiFab*>& b_eb);

    // TODO what is b_eb? may need that plus c_eb?
    void averageDownCoeffsSameAmrLevel (int amrlev, Vector<MultiFab>& a,
                                        Vector<Array<MultiFab,AMREX_SPACEDIM> >& b,
                                        Vector<Array<MultiFab,AMREX_SPACEDIM> >& c,
                                        const Vector<MultiFab*>& b_eb);


    void averageDownCoeffs ();
    void averageDownCoeffsToCoarseAmrLevel (int flev);
};

}

#endif
