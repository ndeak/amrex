
#include <AMReX_MLEBABecCecLap.H>
#include <AMReX_MLABecCecLaplacian.H>
#include <AMReX_EBFArrayBox.H>

#include <AMReX_MLABecCecLap_K.H>
#include <AMReX_MLEBABecCecLap_K.H>

namespace amrex {

void
MLEBABecCecLap::Fapply (int amrlev, int mglev, MultiFab& out, const MultiFab& in) const
{
    BL_PROFILE("MLEBABecCecLap::Fapply()");

    const MultiFab& acoef = m_a_coeffs[amrlev][mglev];
    AMREX_D_TERM(const MultiFab& bxcoef = m_b_coeffs[amrlev][mglev][0];,
                 const MultiFab& bycoef = m_b_coeffs[amrlev][mglev][1];,
                 const MultiFab& bzcoef = m_b_coeffs[amrlev][mglev][2];);
    AMREX_D_TERM(const MultiFab& cxcoef = m_c_coeffs[amrlev][mglev][0];,
                 const MultiFab& cycoef = m_c_coeffs[amrlev][mglev][1];,
                 const MultiFab& czcoef = m_c_coeffs[amrlev][mglev][2];);
    const iMultiFab& ccmask = m_cc_mask[amrlev][mglev];

    const auto dxinvarr = m_geom[amrlev][mglev].InvCellSizeArray();

    auto factory = dynamic_cast<EBFArrayBoxFactory const*>(m_factory[amrlev][mglev].get());
    const FabArray<EBCellFlagFab>* flags = (factory) ? &(factory->getMultiEBCellFlagFab()) : nullptr;
    const MultiFab* vfrac = (factory) ? &(factory->getVolFrac()) : nullptr;
    auto area = (factory) ? factory->getAreaFrac()
        : Array<const MultiCutFab*,AMREX_SPACEDIM>{AMREX_D_DECL(nullptr,nullptr,nullptr)};
    auto fcent = (factory) ? factory->getFaceCent()
        : Array<const MultiCutFab*,AMREX_SPACEDIM>{AMREX_D_DECL(nullptr,nullptr,nullptr)};
    const MultiCutFab* barea = (factory) ? &(factory->getBndryArea()) : nullptr;
    const MultiCutFab* bcent = (factory) ? &(factory->getBndryCent()) : nullptr;
    const auto         ccent = (factory) ? &(factory->getCentroid()) : nullptr;

    const bool is_eb_dirichlet =  isEBDirichlet();
    const bool is_eb_inhomog = m_is_eb_inhomog;

    const int ncomp = getNComp();

    Array4<Real const> foo;

    const Real ascalar = m_a_scalar;
    const Real bscalar = m_b_scalar;
    const Real cscalar = m_c_scalar;

    const Box& domain_box = m_geom[amrlev][mglev].Domain();

    AMREX_D_TERM(
        const int domlo_x = domain_box.smallEnd(0);
        const int domhi_x = domain_box.bigEnd(0);,
        const int domlo_y = domain_box.smallEnd(1);
        const int domhi_y = domain_box.bigEnd(1);,
        const int domlo_z = domain_box.smallEnd(2);
        const int domhi_z = domain_box.bigEnd(2););

    AMREX_D_TERM(
        const bool extdir_x = !(m_geom[amrlev][mglev].isPeriodic(0));,
        const bool extdir_y = !(m_geom[amrlev][mglev].isPeriodic(1));,
        const bool extdir_z = !(m_geom[amrlev][mglev].isPeriodic(2)););

    MFItInfo mfi_info;
    if (Gpu::notInLaunchRegion()) mfi_info.EnableTiling().SetDynamic(true);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(out, mfi_info); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.tilebox();
        Array4<Real const> const& xfab = in.const_array(mfi);
        Array4<Real> const& yfab = out.array(mfi);
        Array4<Real const> const& afab = acoef.const_array(mfi);
        AMREX_D_TERM(Array4<Real const> const& bxfab = bxcoef.const_array(mfi);,
                     Array4<Real const> const& byfab = bycoef.const_array(mfi);,
                     Array4<Real const> const& bzfab = bzcoef.const_array(mfi););
        AMREX_D_TERM(Array4<Real const> const& cxfab = cxcoef.const_array(mfi);,
                     Array4<Real const> const& cyfab = cycoef.const_array(mfi);,
                     Array4<Real const> const& czfab = czcoef.const_array(mfi););

        auto fabtyp = (flags) ? (*flags)[mfi].getType(bx) : FabType::regular;

        if (fabtyp == FabType::covered) {
            AMREX_HOST_DEVICE_PARALLEL_FOR_4D( bx, ncomp, i, j, k, n,
            {
                yfab(i,j,k,n) = 0.0;
            });
        } else if (fabtyp == FabType::regular) {
            AMREX_LAUNCH_HOST_DEVICE_LAMBDA ( bx, tbx,
            {
                mlabecceclap_adotx(tbx, yfab, xfab, afab,
                                AMREX_D_DECL(bxfab,byfab,bzfab),
                                AMREX_D_DECL(cxfab,cyfab,czfab),
                                dxinvarr, ascalar, bscalar, cscalar, ncomp, m_uw_order);
            });
        } else {
            Array4<int const> const& ccmfab = ccmask.const_array(mfi);
            Array4<EBCellFlag const> const& flagfab = flags->const_array(mfi);
            Array4<Real const> const& vfracfab = vfrac->const_array(mfi);
            AMREX_D_TERM(Array4<Real const> const& apxfab = area[0]->const_array(mfi);,
                         Array4<Real const> const& apyfab = area[1]->const_array(mfi);,
                         Array4<Real const> const& apzfab = area[2]->const_array(mfi););
            AMREX_D_TERM(Array4<Real const> const& fcxfab = fcent[0]->const_array(mfi);,
                         Array4<Real const> const& fcyfab = fcent[1]->const_array(mfi);,
                         Array4<Real const> const& fczfab = fcent[2]->const_array(mfi););
            Array4<Real const> const& bafab = barea->const_array(mfi);
            Array4<Real const> const& bcfab = bcent->const_array(mfi);
            Array4<Real const> const& ccfab = ccent->const_array(mfi);
            Array4<Real const> const& bebfab = (is_eb_dirichlet)
                ? m_eb_b_coeffs[amrlev][mglev]->const_array(mfi) : foo;
            Array4<Real const> const& phiebfab = (is_eb_dirichlet && is_eb_inhomog)
                ? m_eb_phi[amrlev]->const_array(mfi) : foo;

            bool beta_on_centroid = (m_beta_loc == Location::FaceCentroid);
            bool  phi_on_centroid = (m_phi_loc  == Location::CellCentroid);

            bool treat_phi_as_on_centroid = ( phi_on_centroid && (mglev == 0) );

            if (treat_phi_as_on_centroid) {
               AMREX_LAUNCH_HOST_DEVICE_LAMBDA ( bx, tbx,
               {
                   mlebabecceclap_adotx_centroid(tbx, yfab, xfab, afab, AMREX_D_DECL(bxfab,byfab,bzfab),
                                     AMREX_D_DECL(cxfab,cyfab,czfab), flagfab, vfracfab,
                                     AMREX_D_DECL(apxfab,apyfab,apzfab),
                                     AMREX_D_DECL(fcxfab,fcyfab,fczfab),
                                     ccfab, bafab, bcfab, bebfab, phiebfab,
                                     AMREX_D_DECL(domlo_x, domlo_y, domlo_z),
                                     AMREX_D_DECL(domhi_x, domhi_y, domhi_z),
                                     AMREX_D_DECL(extdir_x, extdir_y, extdir_z),
                                     is_eb_dirichlet, is_eb_inhomog, dxinvarr,
                                     ascalar, bscalar, cscalar, ncomp);
               });
            } else {
               AMREX_LAUNCH_HOST_DEVICE_LAMBDA ( bx, tbx,
               {
                   mlebabecceclap_adotx(tbx, yfab, xfab, afab, AMREX_D_DECL(bxfab,byfab,bzfab),
                                     AMREX_D_DECL(cxfab,cyfab,czfab), ccmfab, flagfab, vfracfab,
                                     AMREX_D_DECL(apxfab,apyfab,apzfab),
                                     AMREX_D_DECL(fcxfab,fcyfab,fczfab),
                                     bafab, bcfab, bebfab,
                                     is_eb_dirichlet,
                                     phiebfab,
                                     is_eb_inhomog, dxinvarr,
                                     ascalar, bscalar, cscalar, ncomp, beta_on_centroid, phi_on_centroid);
               });
            }
        }
    }
}

void
MLEBABecCecLap::Fsmooth (int amrlev, int mglev, MultiFab& sol, const MultiFab& rhs, int redblack) const
{
    BL_PROFILE("MLEBABecCecLap::Fsmooth()");

    const MultiFab& acoef = m_a_coeffs[amrlev][mglev];
    AMREX_D_TERM(const MultiFab& bxcoef = m_b_coeffs[amrlev][mglev][0];,
                 const MultiFab& bycoef = m_b_coeffs[amrlev][mglev][1];,
                 const MultiFab& bzcoef = m_b_coeffs[amrlev][mglev][2];);
    AMREX_D_TERM(const MultiFab& cxcoef = m_c_coeffs[amrlev][mglev][0];,
                 const MultiFab& cycoef = m_c_coeffs[amrlev][mglev][1];,
                 const MultiFab& czcoef = m_c_coeffs[amrlev][mglev][2];);
    const iMultiFab& ccmask = m_cc_mask[amrlev][mglev];
    const auto& undrrelxr = m_undrrelxr[amrlev][mglev];
    const auto& maskvals  = m_maskvals [amrlev][mglev];

    OrientationIter oitr;

    const FabSet& f0 = undrrelxr[oitr()]; ++oitr;
    const FabSet& f1 = undrrelxr[oitr()]; ++oitr;
#if (AMREX_SPACEDIM > 1)
    const FabSet& f2 = undrrelxr[oitr()]; ++oitr;
    const FabSet& f3 = undrrelxr[oitr()]; ++oitr;
#if (AMREX_SPACEDIM > 2)
    const FabSet& f4 = undrrelxr[oitr()]; ++oitr;
    const FabSet& f5 = undrrelxr[oitr()]; ++oitr;
#endif
#endif

    const MultiMask& mm0 = maskvals[0];
    const MultiMask& mm1 = maskvals[1];
#if (AMREX_SPACEDIM > 1)
    const MultiMask& mm2 = maskvals[2];
    const MultiMask& mm3 = maskvals[3];
#if (AMREX_SPACEDIM > 2)
    const MultiMask& mm4 = maskvals[4];
    const MultiMask& mm5 = maskvals[5];
#endif
#endif

    const int nc = getNComp();
    const Real* h = m_geom[amrlev][mglev].CellSize();
    AMREX_D_TERM(const Real dhx = m_b_scalar/(h[0]*h[0]);,
                 const Real dhy = m_b_scalar/(h[1]*h[1]);,
                 const Real dhz = m_b_scalar/(h[2]*h[2]));
    AMREX_D_TERM(const Real dhxc = m_c_scalar/h[0];,
                 const Real dhyc = m_c_scalar/h[1];,
                 const Real dhzc = m_c_scalar/h[2]);
    const Real alpha = m_a_scalar;

    auto factory = dynamic_cast<EBFArrayBoxFactory const*>(m_factory[amrlev][mglev].get());
    const FabArray<EBCellFlagFab>* flags = (factory) ? &(factory->getMultiEBCellFlagFab()) : nullptr;
    const MultiFab* vfrac = (factory) ? &(factory->getVolFrac()) : nullptr;
    auto area = (factory) ? factory->getAreaFrac()
        : Array<const MultiCutFab*,AMREX_SPACEDIM>{AMREX_D_DECL(nullptr,nullptr,nullptr)};
    auto fcent = (factory) ? factory->getFaceCent()
        : Array<const MultiCutFab*,AMREX_SPACEDIM>{AMREX_D_DECL(nullptr,nullptr,nullptr)};
    const MultiCutFab* barea = (factory) ? &(factory->getBndryArea()) : nullptr;
    const MultiCutFab* bcent = (factory) ? &(factory->getBndryCent()) : nullptr;

    bool is_eb_dirichlet =  isEBDirichlet();

    Array4<Real const> foo;

    MFItInfo mfi_info;
    if (Gpu::notInLaunchRegion()) mfi_info.SetDynamic(true);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(sol, mfi_info);  mfi.isValid(); ++mfi)
    {
        const auto& m0 = mm0.array(mfi);
        const auto& m1 = mm1.array(mfi);
#if (AMREX_SPACEDIM > 1)
        const auto& m2 = mm2.array(mfi);
        const auto& m3 = mm3.array(mfi);
#if (AMREX_SPACEDIM > 2)
        const auto& m4 = mm4.array(mfi);
        const auto& m5 = mm5.array(mfi);
#endif
#endif

        const Box& vbx = mfi.validbox();
        const auto& solnfab = sol.array(mfi);
        const auto& rhsfab  = rhs.const_array(mfi);
        const auto& afab    = acoef.const_array(mfi);

        AMREX_D_TERM(const auto& bxfab = bxcoef.const_array(mfi);,
                     const auto& byfab = bycoef.const_array(mfi);,
                     const auto& bzfab = bzcoef.const_array(mfi););

        AMREX_D_TERM(const auto& cxfab = cxcoef.const_array(mfi);,
                     const auto& cyfab = cycoef.const_array(mfi);,
                     const auto& czfab = czcoef.const_array(mfi););

        const auto& f0fab = f0.const_array(mfi);
        const auto& f1fab = f1.const_array(mfi);
#if (AMREX_SPACEDIM > 1)
        const auto& f2fab = f2.const_array(mfi);
        const auto& f3fab = f3.const_array(mfi);
#if (AMREX_SPACEDIM > 2)
        const auto& f4fab = f4.const_array(mfi);
        const auto& f5fab = f5.const_array(mfi);
#endif
#endif

        auto fabtyp = (flags) ? (*flags)[mfi].getType(vbx) : FabType::regular;

        if (fabtyp == FabType::covered)
        {
            AMREX_HOST_DEVICE_PARALLEL_FOR_4D ( vbx, nc, i, j, k, n,
            {
                solnfab(i,j,k,n) = 0.0;
            });
        }
        else if (fabtyp == FabType::regular)
        {
            AMREX_LAUNCH_HOST_DEVICE_LAMBDA ( vbx, thread_box,
            {
                // TODO: add omega to class, HC for now
                Real omega = 1.0;
                abeccec_gsrb(thread_box, solnfab, rhsfab, alpha, omega, afab,
                          AMREX_D_DECL(dhx, dhy, dhz),
                          AMREX_D_DECL(dhxc, dhyc, dhzc),
                          AMREX_D_DECL(bxfab, byfab, bzfab),
                          AMREX_D_DECL(cxfab, cyfab, czfab),
                          AMREX_D_DECL(m0,m2,m4),
                          AMREX_D_DECL(m1,m3,m5),
                          AMREX_D_DECL(f0fab,f2fab,f4fab),
                          AMREX_D_DECL(f1fab,f3fab,f5fab),
                          vbx, redblack, nc, m_uw_order);
            });
        }
        else
        {
            Array4<int const> const& ccmfab = ccmask.const_array(mfi);
            Array4<EBCellFlag const> const& flagfab = flags->const_array(mfi);
            Array4<Real const> const& vfracfab = vfrac->const_array(mfi);
            AMREX_D_TERM(Array4<Real const> const& apxfab = area[0]->const_array(mfi);,
                         Array4<Real const> const& apyfab = area[1]->const_array(mfi);,
                         Array4<Real const> const& apzfab = area[2]->const_array(mfi););
            AMREX_D_TERM(Array4<Real const> const& fcxfab = fcent[0]->const_array(mfi);,
                         Array4<Real const> const& fcyfab = fcent[1]->const_array(mfi);,
                         Array4<Real const> const& fczfab = fcent[2]->const_array(mfi););
            Array4<Real const> const& bafab = barea->const_array(mfi);
            Array4<Real const> const& bcfab = bcent->const_array(mfi);
            Array4<Real const> const& bebfab = (is_eb_dirichlet)
                ? m_eb_b_coeffs[amrlev][mglev]->const_array(mfi) : foo;

            bool beta_on_centroid = (m_beta_loc == Location::FaceCentroid);
            bool  phi_on_centroid = (m_phi_loc  == Location::CellCentroid);

            if (phi_on_centroid) amrex::Abort("phi_on_centroid is still a WIP");

            AMREX_LAUNCH_HOST_DEVICE_LAMBDA ( vbx, thread_box,
            {
                mlebabecceclap_gsrb(thread_box, solnfab, rhsfab, alpha, afab,
                                 AMREX_D_DECL(dhx, dhy, dhz),
                                 AMREX_D_DECL(dhxc, dhyc, dhzc),
                                 AMREX_D_DECL(bxfab,byfab,bzfab),
                                 AMREX_D_DECL(cxfab,cyfab,czfab),
                                 AMREX_D_DECL(m0,m2,m4),
                                 AMREX_D_DECL(m1,m3,m5),
                                 AMREX_D_DECL(f0fab,f2fab,f4fab),
                                 AMREX_D_DECL(f1fab,f3fab,f5fab),
                                 ccmfab, flagfab, vfracfab,
                                 AMREX_D_DECL(apxfab,apyfab,apzfab),
                                 AMREX_D_DECL(fcxfab,fcyfab,fczfab),
                                 bafab, bcfab, bebfab,
                                 is_eb_dirichlet, beta_on_centroid, phi_on_centroid,
                                 vbx, redblack, nc);
            });
        }
    }
}

void
MLEBABecCecLap::FFlux (int amrlev, const MFIter& mfi, const Array<FArrayBox*,AMREX_SPACEDIM>& flux,
                    const FArrayBox& sol, Location flux_loc, const int face_only) const
{
    BL_PROFILE("MLEBABecCecLap::FFlux()");
    const int compute_flux_at_centroid = (Location::FaceCentroid == flux_loc) ? 1 : 0;
    const int mglev = 0;
    const Box& box = mfi.tilebox();
    const int ncomp = getNComp();

    auto factory = dynamic_cast<EBFArrayBoxFactory const*>(m_factory[amrlev][mglev].get());
    const FabArray<EBCellFlagFab>* flags = (factory) ? &(factory->getMultiEBCellFlagFab()) : nullptr;

    const Real* dxinv = m_geom[amrlev][mglev].InvCellSize();
    AMREX_D_TERM(const auto& bx = m_b_coeffs[amrlev][mglev][0][mfi];,
                 const auto& by = m_b_coeffs[amrlev][mglev][1][mfi];,
                 const auto& bz = m_b_coeffs[amrlev][mglev][2][mfi];);
    AMREX_D_TERM(const auto& cx = m_c_coeffs[amrlev][mglev][0][mfi];,
                 const auto& cy = m_c_coeffs[amrlev][mglev][1][mfi];,
                 const auto& cz = m_c_coeffs[amrlev][mglev][2][mfi];);
    const iMultiFab& ccmask = m_cc_mask[amrlev][mglev];

    AMREX_D_TERM(Box const& xbx = amrex::surroundingNodes(box,0);,
                 Box const& ybx = amrex::surroundingNodes(box,1);,
                 Box const& zbx = amrex::surroundingNodes(box,2););
    AMREX_D_TERM(Array4<Real> const& fx = flux[0]->array();,
                 Array4<Real> const& fy = flux[1]->array();,
                 Array4<Real> const& fz = flux[2]->array(););

    const auto fabtyp = (flags) ? (*flags)[mfi].getType(box) : FabType::regular;
    if (fabtyp == FabType::covered) {
        AMREX_HOST_DEVICE_PARALLEL_FOR_4D(xbx, ncomp, i, j, k, n,
        {
            fx(i,j,k,n) = 0.0;
        });
        AMREX_HOST_DEVICE_PARALLEL_FOR_4D(ybx, ncomp, i, j, k, n,
        {
            fy(i,j,k,n) = 0.0;
        });
#if (AMREX_SPACEDIM == 3)
        AMREX_HOST_DEVICE_PARALLEL_FOR_4D(zbx, ncomp, i, j, k, n,
        {
            fz(i,j,k,n) = 0.0;
        });
#endif
    } else if (fabtyp == FabType::regular) {
        MLABecCecLaplacian::FFlux(box, dxinv, m_b_scalar, m_c_scalar,
                               Array<FArrayBox const*,AMREX_SPACEDIM>{AMREX_D_DECL(&bx,&by,&bz)},
                               Array<FArrayBox const*,AMREX_SPACEDIM>{AMREX_D_DECL(&cx,&cy,&cz)},
                               flux, sol, face_only, ncomp, m_uw_order);
    } else if (compute_flux_at_centroid) {
        const auto& area = factory->getAreaFrac();
        const auto& fcent = factory->getFaceCent();
        AMREX_D_TERM(Array4<Real const> const& apx = area[0]->const_array(mfi);,
                     Array4<Real const> const& apy = area[1]->const_array(mfi);,
                     Array4<Real const> const& apz = area[2]->const_array(mfi););
        AMREX_D_TERM(Array4<Real const> const& fcx = fcent[0]->const_array(mfi);,
                     Array4<Real const> const& fcy = fcent[1]->const_array(mfi);,
                     Array4<Real const> const& fcz = fcent[2]->const_array(mfi););
        Array4<Real const> const& phi = sol.const_array();
        AMREX_D_TERM(Array4<Real const> const& bxcoef = bx.const_array();,
                     Array4<Real const> const& bycoef = by.const_array();,
                     Array4<Real const> const& bzcoef = bz.const_array(););
        AMREX_D_TERM(Array4<Real const> const& cxcoef = cx.const_array();,
                     Array4<Real const> const& cycoef = cy.const_array();,
                     Array4<Real const> const& czcoef = cz.const_array(););
        Array4<int const> const& msk = ccmask.const_array(mfi);
        AMREX_D_TERM(Real dhx = m_b_scalar*dxinv[0];,
                     Real dhy = m_b_scalar*dxinv[1];,
                     Real dhz = m_b_scalar*dxinv[2];);
        AMREX_D_TERM(Real dhxc = m_c_scalar*dxinv[0];,
                     Real dhyc = m_c_scalar*dxinv[1];,
                     Real dhzc = m_c_scalar*dxinv[2];);

        bool beta_on_centroid = (m_beta_loc == Location::FaceCentroid);
        bool  phi_on_centroid = (m_phi_loc  == Location::CellCentroid);

        if (phi_on_centroid) amrex::Abort("phi_on_centroid is still a WIP");

        AMREX_LAUNCH_HOST_DEVICE_LAMBDA_DIM (
            xbx, txbx,
            {
                mlebabecceclap_flux_x(txbx, fx, apx, fcx, phi, bxcoef, cxcoef, msk, dhx, dhxc, face_only, ncomp, xbx,
                                   beta_on_centroid, phi_on_centroid);
            }
            , ybx, tybx,
            {
                mlebabecceclap_flux_y(tybx, fy, apy, fcy, phi, bycoef, cycoef, msk, dhy, dhyc, face_only, ncomp, ybx,
                                   beta_on_centroid, phi_on_centroid);
            }
            , zbx, tzbx,
            {
                mlebabecceclap_flux_z(tzbx, fz, apz, fcz, phi, bzcoef, czcoef, msk, dhz, dhzc, face_only, ncomp, zbx,
                                   beta_on_centroid, phi_on_centroid);
            }
        );
    } else {
        const auto& area = factory->getAreaFrac();
        AMREX_D_TERM(Array4<Real const> const& apx = area[0]->const_array(mfi);,
                     Array4<Real const> const& apy = area[1]->const_array(mfi);,
                     Array4<Real const> const& apz = area[2]->const_array(mfi););
        Array4<Real const> const& phi = sol.const_array();
        AMREX_D_TERM(Array4<Real const> const& bxcoef = bx.const_array();,
                     Array4<Real const> const& bycoef = by.const_array();,
                     Array4<Real const> const& bzcoef = bz.const_array(););
        AMREX_D_TERM(Array4<Real const> const& cxcoef = cx.const_array();,
                     Array4<Real const> const& cycoef = cy.const_array();,
                     Array4<Real const> const& czcoef = cz.const_array(););
        AMREX_D_TERM(Real dhx = m_b_scalar*dxinv[0];,
                     Real dhy = m_b_scalar*dxinv[1];,
                     Real dhz = m_b_scalar*dxinv[2];);
        AMREX_D_TERM(Real dhxc = m_c_scalar*dxinv[0];,
                     Real dhyc = m_c_scalar*dxinv[1];,
                     Real dhzc = m_c_scalar*dxinv[2];);

        AMREX_LAUNCH_HOST_DEVICE_LAMBDA_DIM (
            xbx, txbx,
            {
                mlebabecceclap_flux_x_0(txbx, fx, apx, phi, bxcoef, cxcoef, dhx, dhxc, face_only, ncomp, xbx);
            }
            , ybx, tybx,
            {
                mlebabecceclap_flux_y_0(tybx, fy, apy, phi, bycoef, cycoef, dhy, dhyc, face_only, ncomp, ybx);
            }
            , zbx, tzbx,
            {
                mlebabecceclap_flux_z_0(tzbx, fz, apz, phi, bzcoef, czcoef, dhz, dhzc, face_only, ncomp, zbx);
            }
        );
    }
}

}
