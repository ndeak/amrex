
#include <AMReX_MLABecCecLaplacian.H>
#include <AMReX_MultiFabUtil.H>

#include <AMReX_MLABecCecLap_K.H>
#include <AMReX_Slopes_K.H>

namespace amrex {

MLABecCecLaplacian::MLABecCecLaplacian (const Vector<Geometry>& a_geom,
                                        const Vector<BoxArray>& a_grids,
                                        const Vector<DistributionMapping>& a_dmap,
                                        const LPInfo& a_info,
                                        const Vector<FabFactory<FArrayBox> const*>& a_factory)
{
    define(a_geom, a_grids, a_dmap, a_info, a_factory);
}

MLABecCecLaplacian::MLABecCecLaplacian (const Vector<Geometry>& a_geom,
                                        const Vector<BoxArray>& a_grids,
                                        const Vector<DistributionMapping>& a_dmap,
                                        const Vector<iMultiFab const*>& a_overset_mask,
                                        const LPInfo& a_info,
                                        const Vector<FabFactory<FArrayBox> const*>& a_factory)
{
    define(a_geom, a_grids, a_dmap, a_overset_mask, a_info, a_factory);
}

void
MLABecCecLaplacian::define (const Vector<Geometry>& a_geom,
                            const Vector<BoxArray>& a_grids,
                            const Vector<DistributionMapping>& a_dmap,
                            const LPInfo& a_info,
                            const Vector<FabFactory<FArrayBox> const*>& a_factory)
{
    BL_PROFILE("MLABecCecLaplacian::define()");

    MLCellABecCecLap::define(a_geom, a_grids, a_dmap, a_info, a_factory);

    const int ncomp = getNComp();

    m_a_coeffs.resize(m_num_amr_levels);
    m_b_coeffs.resize(m_num_amr_levels);
    m_c_coeffs.resize(m_num_amr_levels);
    m_overset_mask.resize(m_num_amr_levels);
    for (int amrlev = 0; amrlev < m_num_amr_levels; ++amrlev)
    {
        m_a_coeffs[amrlev].resize(m_num_mg_levels[amrlev]);
        m_b_coeffs[amrlev].resize(m_num_mg_levels[amrlev]);
        m_c_coeffs[amrlev].resize(m_num_mg_levels[amrlev]);
        m_overset_mask[amrlev].resize(m_num_mg_levels[amrlev]);
        for (int mglev = 0; mglev < m_num_mg_levels[amrlev]; ++mglev)
        {
            m_a_coeffs[amrlev][mglev].define(m_grids[amrlev][mglev],
                                             m_dmap[amrlev][mglev],
                                             1, 0, MFInfo(), *m_factory[amrlev][mglev]);
            for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
            {
                const BoxArray& ba = amrex::convert(m_grids[amrlev][mglev],
                                                    IntVect::TheDimensionVector(idim));
                m_b_coeffs[amrlev][mglev][idim].define(ba,
                                                       m_dmap[amrlev][mglev],
                                                       ncomp, 0, MFInfo(), *m_factory[amrlev][mglev]);
                m_c_coeffs[amrlev][mglev][idim].define(ba,
                                                       m_dmap[amrlev][mglev],
                                                       ncomp, 0, MFInfo(), *m_factory[amrlev][mglev]);
            }
        }
    }
}

void
MLABecCecLaplacian::define (const Vector<Geometry>& a_geom,
                            const Vector<BoxArray>& a_grids,
                            const Vector<DistributionMapping>& a_dmap,
                            const Vector<iMultiFab const*>& a_overset_mask,
                            const LPInfo& a_info,
                            const Vector<FabFactory<FArrayBox> const*>& a_factory)
{
    BL_PROFILE("MLABecCecLaplacian::define(overset)");

    int namrlevs = a_geom.size();
    m_overset_mask.resize(namrlevs);
    for (int amrlev = 0; amrlev < namrlevs; ++amrlev)
    {
        m_overset_mask[amrlev].emplace_back(new iMultiFab(a_grids[amrlev], a_dmap[amrlev], 1, 1));
        iMultiFab::Copy(*m_overset_mask[amrlev][0], *a_overset_mask[amrlev], 0, 0, 1, 0);
        if (amrlev > 1) {
            AMREX_ALWAYS_ASSERT(amrex::refine(a_geom[amrlev-1].Domain(),2)
                                == a_geom[amrlev].Domain());
        }
    }

    int amrlev = 0;
    Box dom = a_geom[0].Domain();
    for (int mglev = 1; mglev <= a_info.max_coarsening_level; ++mglev)
    {
        AMREX_ALWAYS_ASSERT(mg_coarsen_ratio == 2);
        iMultiFab const& fine = *m_overset_mask[amrlev][mglev-1];
        if (dom.coarsenable(2) and fine.boxArray().coarsenable(2)) {
            dom.coarsen(2);
            std::unique_ptr<iMultiFab> crse(new iMultiFab(amrex::coarsen(fine.boxArray(),2),
                                                          fine.DistributionMap(), 1, 1));
            ReduceOps<ReduceOpSum> reduce_op;
            ReduceData<int> reduce_data(reduce_op);
            using ReduceTuple = typename decltype(reduce_data)::Type;
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
            for (MFIter mfi(*crse, TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                const Box& bx = mfi.tilebox();
                Array4<int const> const& fmsk = fine.const_array(mfi);
                Array4<int> const& cmsk = crse->array(mfi);
                reduce_op.eval(bx, reduce_data,
                [=] AMREX_GPU_HOST_DEVICE (Box const& b) -> ReduceTuple
                {
                    return { coarsen_overset_mask(b, cmsk, fmsk) };
                });
            }
            ReduceTuple hv = reduce_data.value();
            if (amrex::get<0>(hv) == 0) {
                m_overset_mask[amrlev].push_back(std::move(crse));
            } else {
                break;
            }
        } else {
            break;
        }
    }
    int max_overset_mask_coarsening_level = m_overset_mask[amrlev].size()-1;
    ParallelAllReduce::Min(max_overset_mask_coarsening_level, ParallelContext::CommunicatorSub());
    m_overset_mask[amrlev].resize(max_overset_mask_coarsening_level+1);

    LPInfo linfo = a_info;
    linfo.max_coarsening_level = std::min(a_info.max_coarsening_level,
                                          max_overset_mask_coarsening_level);
    define(a_geom, a_grids, a_dmap, linfo, a_factory);

    amrlev = 0;
    for (int mglev = 1; mglev < m_num_mg_levels[amrlev]; ++mglev) {
        if (! amrex::isMFIterSafe(*m_overset_mask[amrlev][mglev], m_a_coeffs[amrlev][mglev])) {
            std::unique_ptr<iMultiFab> osm(new iMultiFab(m_grids[amrlev][mglev],
                                                         m_dmap[amrlev][mglev], 1, 1));
            osm->ParallelCopy(*m_overset_mask[amrlev][mglev]);
            std::swap(osm, m_overset_mask[amrlev][mglev]);
        }
    }

    for (amrlev = 0; amrlev < m_num_amr_levels; ++amrlev) {
        for (int mglev = 0; mglev < m_num_mg_levels[amrlev]; ++mglev) {
            m_overset_mask[amrlev][mglev]->setBndry(1);
            m_overset_mask[amrlev][mglev]->FillBoundary(m_geom[amrlev][mglev].periodicity());
        }
    }
}

MLABecCecLaplacian::~MLABecCecLaplacian ()
{}

void
MLABecCecLaplacian::setScalars (Real a, Real b, Real c) noexcept
{
    m_a_scalar = a;
    m_b_scalar = b;
    m_c_scalar = c;
    if (a == 0.0)
    {
        for (int amrlev = 0; amrlev < m_num_amr_levels; ++amrlev)
        {
            m_a_coeffs[amrlev][0].setVal(0.0);
        }
    }
}

void
MLABecCecLaplacian::setRelaxation (Real omega) noexcept
{
   m_omega = omega;
}

void
MLABecCecLaplacian::setUWOrder (int uw_order) noexcept
{
   m_uw_order = uw_order;
}

void
MLABecCecLaplacian::setACoeffs (int amrlev, const MultiFab& alpha)
{
    MultiFab::Copy(m_a_coeffs[amrlev][0], alpha, 0, 0, 1, 0);
    m_needs_update = true;
}

void
MLABecCecLaplacian::setACoeffs (int amrlev, Real alpha)
{
    m_a_coeffs[amrlev][0].setVal(alpha);
    m_needs_update = true;
}

void
MLABecCecLaplacian::setBCoeffs (int amrlev,
                                const Array<MultiFab const*,AMREX_SPACEDIM>& beta)
{
    const int ncomp = getNComp();
    AMREX_ALWAYS_ASSERT(beta[0]->nComp() == 1 or beta[0]->nComp() == ncomp);
    if (beta[0]->nComp() == ncomp)
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            for (int icomp = 0; icomp < ncomp; ++icomp) {
                MultiFab::Copy(m_b_coeffs[amrlev][0][idim], *beta[idim], icomp, icomp, 1, 0);
            }
        }
    else 
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            for (int icomp = 0; icomp < ncomp; ++icomp) {
                MultiFab::Copy(m_b_coeffs[amrlev][0][idim], *beta[idim], 0, icomp, 1, 0);
            }
        }
    m_needs_update = true;
}

void
MLABecCecLaplacian::setBCoeffs (int amrlev, Real beta)
{
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        m_b_coeffs[amrlev][0][idim].setVal(beta);
    }
    m_needs_update = true;
}

void
MLABecCecLaplacian::setBCoeffs (int amrlev, Vector<Real> const& beta)
{
    const int ncomp = getNComp();
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        for (int icomp = 0; icomp < ncomp; ++icomp) {
            m_b_coeffs[amrlev][0][idim].setVal(beta[icomp]);
        }
    }
    m_needs_update = true;
}

void
MLABecCecLaplacian::setCCoeffs (int amrlev,
                                const Array<MultiFab const*,AMREX_SPACEDIM>& eta)
{
    const int ncomp = getNComp();
    AMREX_ALWAYS_ASSERT(eta[0]->nComp() == 1 or eta[0]->nComp() == ncomp);
    if (eta[0]->nComp() == ncomp)
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            for (int icomp = 0; icomp < ncomp; ++icomp) {
                MultiFab::Copy(m_c_coeffs[amrlev][0][idim], *eta[idim], icomp, icomp, 1, 0);
            }
        }
    else 
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            for (int icomp = 0; icomp < ncomp; ++icomp) {
                MultiFab::Copy(m_c_coeffs[amrlev][0][idim], *eta[idim], 0, icomp, 1, 0);
            }
        }
    m_needs_update = true;
}

void
MLABecCecLaplacian::averageDownCoeffs ()
{
    BL_PROFILE("MLABecCecLaplacian::averageDownCoeffs()");

    for (int amrlev = m_num_amr_levels-1; amrlev > 0; --amrlev)
    {
        auto& fine_a_coeffs = m_a_coeffs[amrlev];
        auto& fine_b_coeffs = m_b_coeffs[amrlev];
        auto& fine_c_coeffs = m_c_coeffs[amrlev];

        averageDownCoeffsSameAmrLevel(amrlev, fine_a_coeffs, fine_b_coeffs, fine_c_coeffs);
        averageDownCoeffsToCoarseAmrLevel(amrlev);
    }

    averageDownCoeffsSameAmrLevel(0, m_a_coeffs[0], m_b_coeffs[0], m_c_coeffs[0]);
}

void
MLABecCecLaplacian::averageDownCoeffsSameAmrLevel (int amrlev, Vector<MultiFab>& a,
                                                   Vector<Array<MultiFab,AMREX_SPACEDIM> >& b,
                                                   Vector<Array<MultiFab,AMREX_SPACEDIM> >& c)
{
    int nmglevs = a.size();
    for (int mglev = 1; mglev < nmglevs; ++mglev)
    {
        IntVect ratio = (amrlev > 0) ? IntVect(mg_coarsen_ratio) : mg_coarsen_ratio_vec[mglev-1];

        if (m_a_scalar == 0.0)
        {
            a[mglev].setVal(0.0);
        }
        else
        {
            amrex::average_down(a[mglev-1], a[mglev], 0, 1, ratio);
        }
        
        Vector<const MultiFab*> fine {AMREX_D_DECL(&(b[mglev-1][0]),
                                                   &(b[mglev-1][1]),
                                                   &(b[mglev-1][2]))};
        Vector<MultiFab*> crse {AMREX_D_DECL(&(b[mglev][0]),
                                             &(b[mglev][1]),
                                             &(b[mglev][2]))};

        Vector<const MultiFab*> finec {AMREX_D_DECL(&(c[mglev-1][0]),
                                                    &(c[mglev-1][1]),
                                                    &(c[mglev-1][2]))};
        Vector<MultiFab*> crsec {AMREX_D_DECL(&(c[mglev][0]),
                                              &(c[mglev][1]),
                                              &(c[mglev][2]))};

        amrex::average_down_faces(fine, crse, ratio, 0);
        amrex::average_down_faces(finec, crsec, ratio, 0);
    }

    for (int mglev = 1; mglev < nmglevs; ++mglev)
    {
        if (m_overset_mask[amrlev][mglev]) {
            const Real fac = static_cast<Real>(1 << mglev); // 2**mglev
            const Real osfac = 2.0*fac/(fac+1.0);
            const int ncomp = getNComp();
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
            for (MFIter mfi(a[mglev],TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                AMREX_D_TERM(Box const& xbx = mfi.nodaltilebox(0);,
                             Box const& ybx = mfi.nodaltilebox(1);,
                             Box const& zbx = mfi.nodaltilebox(2));
                AMREX_D_TERM(Array4<Real> const& bx = b[mglev][0].array(mfi);,
                             Array4<Real> const& by = b[mglev][1].array(mfi);,
                             Array4<Real> const& bz = b[mglev][2].array(mfi));
                AMREX_D_TERM(Array4<Real> const& cx = c[mglev][0].array(mfi);,
                             Array4<Real> const& cy = c[mglev][1].array(mfi);,
                             Array4<Real> const& cz = c[mglev][2].array(mfi));
                Array4<int const> const& osm = m_overset_mask[amrlev][mglev]->const_array(mfi);
                AMREX_LAUNCH_HOST_DEVICE_LAMBDA_DIM
                    (xbx, t_xbx,
                     {
                         overset_rescale_bcoef_x(t_xbx, bx, osm, ncomp, osfac);
                         overset_rescale_bcoef_x(t_xbx, cx, osm, ncomp, osfac);
                     },
                     ybx, t_ybx,
                     {
                         overset_rescale_bcoef_y(t_ybx, by, osm, ncomp, osfac);
                         overset_rescale_bcoef_y(t_ybx, cy, osm, ncomp, osfac);
                     },
                     zbx, t_zbx,
                     {
                         overset_rescale_bcoef_z(t_zbx, bz, osm, ncomp, osfac);
                         overset_rescale_bcoef_z(t_zbx, cz, osm, ncomp, osfac);
                     });
            }
        }
    }
}

void
MLABecCecLaplacian::averageDownCoeffsToCoarseAmrLevel (int flev)
{
    auto& fine_a_coeffs = m_a_coeffs[flev  ].back();
    auto& fine_b_coeffs = m_b_coeffs[flev  ].back();
    auto& fine_c_coeffs = m_c_coeffs[flev  ].back();
    auto& crse_a_coeffs = m_a_coeffs[flev-1].front();
    auto& crse_b_coeffs = m_b_coeffs[flev-1].front();
    auto& crse_c_coeffs = m_c_coeffs[flev-1].front();

    if (m_a_scalar != 0.0) {
        // We coarsen from the back of flev to the front of flev-1.
        // So we use mg_coarsen_ratio.
        amrex::average_down(fine_a_coeffs, crse_a_coeffs, 0, 1, mg_coarsen_ratio);
    }

    amrex::average_down_faces(amrex::GetArrOfConstPtrs(fine_b_coeffs),
                              amrex::GetArrOfPtrs(crse_b_coeffs),
                              IntVect(mg_coarsen_ratio), m_geom[flev-1][0]);

    amrex::average_down_faces(amrex::GetArrOfConstPtrs(fine_c_coeffs),
                              amrex::GetArrOfPtrs(crse_c_coeffs),
                              IntVect(mg_coarsen_ratio), m_geom[flev-1][0]);
}

void
MLABecCecLaplacian::applyMetricTermsCoeffs ()
{
#if (AMREX_SPACEDIM != 3)
    for (int alev = 0; alev < m_num_amr_levels; ++alev)
    {
        const int mglev = 0;
        applyMetricTerm(alev, mglev, m_a_coeffs[alev][mglev]);
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
        {
            applyMetricTerm(alev, mglev, m_b_coeffs[alev][mglev][idim]);
            applyMetricTerm(alev, mglev, m_c_coeffs[alev][mglev][idim]);
        }
    }
#endif
}

void
MLABecCecLaplacian::prepareForSolve ()
{
    BL_PROFILE("MLABecCecLaplacian::prepareForSolve()");

    MLCellABecCecLap::prepareForSolve();

#if (AMREX_SPACEDIM != 3)
    applyMetricTermsCoeffs();
#endif

    averageDownCoeffs();

    m_is_singular.clear();
    m_is_singular.resize(m_num_amr_levels, false);
    auto itlo = std::find(m_lobc[0].begin(), m_lobc[0].end(), BCType::Dirichlet);
    auto ithi = std::find(m_hibc[0].begin(), m_hibc[0].end(), BCType::Dirichlet);
    if (itlo == m_lobc[0].end() && ithi == m_hibc[0].end())
    {  // No Dirichlet
        for (int alev = 0; alev < m_num_amr_levels; ++alev)
        {
            // For now this assumes that overset regions are treated as Dirichlet bc's
            if (m_domain_covered[alev] && !m_overset_mask[alev][0]) 
            {
                if (m_a_scalar == 0.0)
                {
                    m_is_singular[alev] = true;
                }
                else
                {
                    Real asum = m_a_coeffs[alev].back().sum();
                    Real amax = m_a_coeffs[alev].back().norm0();
                    m_is_singular[alev] = (asum <= amax * 1.e-12);
                }
            }
        }
    }

    m_needs_update = false;
}

void
MLABecCecLaplacian::Fapply (int amrlev, int mglev, MultiFab& out, const MultiFab& in) const
{
  
    // FIXME: make second order
    BL_PROFILE("MLABecCecLaplacian::Fapply()");

    const MultiFab& acoef = m_a_coeffs[amrlev][mglev];
    AMREX_D_TERM(const MultiFab& bxcoef = m_b_coeffs[amrlev][mglev][0];,
                 const MultiFab& bycoef = m_b_coeffs[amrlev][mglev][1];,
                 const MultiFab& bzcoef = m_b_coeffs[amrlev][mglev][2];);
    AMREX_D_TERM(const MultiFab& cxcoef = m_c_coeffs[amrlev][mglev][0];,
                 const MultiFab& cycoef = m_c_coeffs[amrlev][mglev][1];,
                 const MultiFab& czcoef = m_c_coeffs[amrlev][mglev][2];);

    const auto dxinv = m_geom[amrlev][mglev].InvCellSizeArray();
    const Box& domain = m_geom[amrlev][mglev].Domain();

    const Real ascalar = m_a_scalar;
    const Real bscalar = m_b_scalar;
    const Real cscalar = m_c_scalar;

    const int ncomp = getNComp();

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(out, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.tilebox();
        const auto& xfab = in.array(mfi);
        const auto& yfab = out.array(mfi);
        const auto& afab = acoef.array(mfi);
        AMREX_D_TERM(const auto& bxfab = bxcoef.array(mfi);,
                     const auto& byfab = bycoef.array(mfi);,
                     const auto& bzfab = bzcoef.array(mfi););
        AMREX_D_TERM(const auto& cxfab = cxcoef.array(mfi);,
                     const auto& cyfab = cycoef.array(mfi);,
                     const auto& czfab = czcoef.array(mfi););
        if (m_overset_mask[amrlev][mglev]) {
            const auto& osm = m_overset_mask[amrlev][mglev]->array(mfi);
            AMREX_LAUNCH_HOST_DEVICE_LAMBDA ( bx, tbx,
            {
                mlabecceclap_adotx_os(tbx, yfab, xfab, afab, AMREX_D_DECL(bxfab,byfab,bzfab),
                                      AMREX_D_DECL(cxfab,cyfab,czfab), osm, dxinv, 
                                      ascalar, bscalar, cscalar, ncomp, m_uw_order);
            });
        } else {
            AMREX_LAUNCH_HOST_DEVICE_LAMBDA ( bx, tbx,
            {
                mlabecceclap_adotx(tbx, yfab, xfab, afab, AMREX_D_DECL(bxfab,byfab,bzfab),
                                   AMREX_D_DECL(cxfab,cyfab,czfab), dxinv, 
                                   ascalar, bscalar, cscalar, ncomp, m_uw_order);
            });
        }
    }
}

void
MLABecCecLaplacian::normalize (int amrlev, int mglev, MultiFab& mf) const
{
    // FIXME: make second order
    BL_PROFILE("MLABecCecLaplacian::normalize()");

    const MultiFab& acoef = m_a_coeffs[amrlev][mglev];
    AMREX_D_TERM(const MultiFab& bxcoef = m_b_coeffs[amrlev][mglev][0];,
                 const MultiFab& bycoef = m_b_coeffs[amrlev][mglev][1];,
                 const MultiFab& bzcoef = m_b_coeffs[amrlev][mglev][2];);
    AMREX_D_TERM(const MultiFab& cxcoef = m_c_coeffs[amrlev][mglev][0];,
                 const MultiFab& cycoef = m_c_coeffs[amrlev][mglev][1];,
                 const MultiFab& czcoef = m_c_coeffs[amrlev][mglev][2];);

    const auto dxinv = m_geom[amrlev][mglev].InvCellSizeArray();

    const Real ascalar = m_a_scalar;
    const Real bscalar = m_b_scalar;
    const Real cscalar = m_c_scalar;

    const int ncomp = getNComp();

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(mf, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.tilebox();
        const auto& fab = mf.array(mfi);
        const auto& afab = acoef.array(mfi);
        AMREX_D_TERM(const auto& bxfab = bxcoef.array(mfi);,
                     const auto& byfab = bycoef.array(mfi);,
                     const auto& bzfab = bzcoef.array(mfi););
        AMREX_D_TERM(const auto& cxfab = cxcoef.array(mfi);,
                     const auto& cyfab = cycoef.array(mfi);,
                     const auto& czfab = czcoef.array(mfi););

        AMREX_LAUNCH_HOST_DEVICE_LAMBDA ( bx, tbx,
        {
            mlabecceclap_normalize(tbx, fab, afab, AMREX_D_DECL(bxfab,byfab,bzfab),
                                   AMREX_D_DECL(cxfab,cyfab,czfab), dxinv, 
                                   ascalar, bscalar, cscalar, ncomp, m_uw_order);
        });
    }
}

void
MLABecCecLaplacian::Fsmooth (int amrlev, int mglev, MultiFab& sol, const MultiFab& rhs, int redblack) const
{
    // FIXME: make second order
    BL_PROFILE("MLABecCecLaplacian::Fsmooth()");

    bool regular_coarsening = true;
    if (amrlev == 0 and mglev > 0) {
        regular_coarsening = mg_coarsen_ratio_vec[mglev-1] == mg_coarsen_ratio;
    }

    const MultiFab& acoef = m_a_coeffs[amrlev][mglev];
    AMREX_D_TERM(const MultiFab& bxcoef = m_b_coeffs[amrlev][mglev][0];,
                 const MultiFab& bycoef = m_b_coeffs[amrlev][mglev][1];,
                 const MultiFab& bzcoef = m_b_coeffs[amrlev][mglev][2];);
    AMREX_D_TERM(const MultiFab& cxcoef = m_c_coeffs[amrlev][mglev][0];,
                 const MultiFab& cycoef = m_c_coeffs[amrlev][mglev][1];,
                 const MultiFab& czcoef = m_c_coeffs[amrlev][mglev][2];);
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
    AMREX_D_TERM(const Real dhxsq = m_b_scalar/(h[0]*h[0]);,
                 const Real dhysq = m_b_scalar/(h[1]*h[1]);,
                 const Real dhzsq = m_b_scalar/(h[2]*h[2]));
    AMREX_D_TERM(const Real dhx = m_c_scalar/(h[0]);,
                 const Real dhy = m_c_scalar/(h[1]);,
                 const Real dhz = m_c_scalar/(h[2]));
    const Real alpha = m_a_scalar;
    const Real omega = m_omega;

    MFItInfo mfi_info;
    if (Gpu::notInLaunchRegion()) mfi_info.EnableTiling().SetDynamic(true);

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(sol,mfi_info); mfi.isValid(); ++mfi)
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

	const Box& tbx = mfi.tilebox();
        const Box& vbx = mfi.validbox();
        const auto& solnfab = sol.array(mfi);
        const auto& rhsfab  = rhs.array(mfi);
        const auto& afab    = acoef.array(mfi);

        AMREX_D_TERM(const auto& bxfab = bxcoef.array(mfi);,
                     const auto& byfab = bycoef.array(mfi);,
                     const auto& bzfab = bzcoef.array(mfi););
        AMREX_D_TERM(const auto& cxfab = cxcoef.array(mfi);,
                     const auto& cyfab = cycoef.array(mfi);,
                     const auto& czfab = czcoef.array(mfi););

        const auto& f0fab = f0.array(mfi);
        const auto& f1fab = f1.array(mfi);
#if (AMREX_SPACEDIM > 1)
        const auto& f2fab = f2.array(mfi);
        const auto& f3fab = f3.array(mfi);
#if (AMREX_SPACEDIM > 2)
        const auto& f4fab = f4.array(mfi);
        const auto& f5fab = f5.array(mfi);
#endif
#endif

#ifdef AMREX_USE_DPCPP
        // xxxxx DPCPP todo: kernel size
        Vector<Array4<Real const> > ha(2*AMREX_SPACEDIM);
        ha[0] = f0fab;
        ha[1] = f1fab;
#if (AMREX_SPACEDIM > 1)
        ha[2] = f2fab;
        ha[3] = f3fab;
#if (AMREX_SPACEDIM == 3)
        ha[4] = f4fab;
        ha[5] = f5fab;
#endif
#endif
        Gpu::AsyncArray<Array4<Real const> > aa(ha.data(), 2*AMREX_SPACEDIM);
        auto dp = aa.data();

        if (m_overset_mask[amrlev][mglev]) {
            const auto& osm = m_overset_mask[amrlev][mglev]->array(mfi);
            AMREX_LAUNCH_HOST_DEVICE_LAMBDA ( tbx, thread_box,
            {
                abeccec_gsrb_os(thread_box, solnfab, rhsfab, alpha, omega, afab,
                                AMREX_D_DECL(dhxsq, dhysq, dhzsq),
                                AMREX_D_DECL(dhx, dhy, dhz),
                                AMREX_D_DECL(bxfab, byfab, bzfab),
                                AMREX_D_DECL(cxfab, cyfab, czfab),
                                AMREX_D_DECL(m0,m2,m4),
                                AMREX_D_DECL(m1,m3,m5),
                                AMREX_D_DECL(dp[0],dp[2],dp[4]),
                                AMREX_D_DECL(dp[1],dp[3],dp[5]),
                                osm, vbx, redblack, nc, m_uw_order);
            });
        } else if (regular_coarsening) {
            AMREX_LAUNCH_HOST_DEVICE_LAMBDA ( tbx, thread_box,
            {
                abeccec_gsrb(thread_box, solnfab, rhsfab, alpha, omega, afab,
                             AMREX_D_DECL(dhxsq, dhysq, dhzsq),
                             AMREX_D_DECL(dhx, dhy, dhz),
                             AMREX_D_DECL(bxfab, byfab, bzfab),
                             AMREX_D_DECL(cxfab, cyfab, czfab),
                             AMREX_D_DECL(m0,m2,m4),
                             AMREX_D_DECL(m1,m3,m5),
                             AMREX_D_DECL(dp[0],dp[2],dp[4]),
                             AMREX_D_DECL(dp[1],dp[3],dp[5]),
                             vbx, redblack, nc, m_uw_order);
            });
        } else {
            Abort("Line GS not available for ABecCecLap");
        }
#else
        if (m_overset_mask[amrlev][mglev]) {
            const auto& osm = m_overset_mask[amrlev][mglev]->array(mfi);
            AMREX_LAUNCH_HOST_DEVICE_LAMBDA ( tbx, thread_box,
            {
                abeccec_gsrb_os(thread_box, solnfab, rhsfab, alpha, omega, afab,
                                AMREX_D_DECL(dhxsq, dhysq, dhzsq),
                                AMREX_D_DECL(dhx, dhy, dhz),
                                AMREX_D_DECL(bxfab, byfab, bzfab),
                                AMREX_D_DECL(cxfab, cyfab, czfab),
                                AMREX_D_DECL(m0,m2,m4),
                                AMREX_D_DECL(m1,m3,m5),
                                AMREX_D_DECL(f0fab,f2fab,f4fab),
                                AMREX_D_DECL(f1fab,f3fab,f5fab),
                                osm, vbx, redblack, nc, m_uw_order);
            });
        } else if (regular_coarsening) {
            AMREX_LAUNCH_HOST_DEVICE_LAMBDA ( tbx, thread_box,
            {
                abeccec_gsrb(thread_box, solnfab, rhsfab, alpha, omega, afab,
                             AMREX_D_DECL(dhxsq, dhysq, dhzsq),
                             AMREX_D_DECL(dhx, dhy, dhz),
                             AMREX_D_DECL(bxfab, byfab, bzfab),
                             AMREX_D_DECL(cxfab, cyfab, czfab),
                             AMREX_D_DECL(m0,m2,m4),
                             AMREX_D_DECL(m1,m3,m5),
                             AMREX_D_DECL(f0fab,f2fab,f4fab),
                             AMREX_D_DECL(f1fab,f3fab,f5fab),
                             vbx, redblack, nc, m_uw_order);
            });
        } else {
            Abort("Line GS not available for ABecCecLap");
        }
#endif
    }
}

void
MLABecCecLaplacian::checkDiagonalDominance (int amrlev, int mglev)
{
    BL_PROFILE("MLABecCecLaplacian::checkDiagonalDominance()");

    const MultiFab& acoef = m_a_coeffs[amrlev][mglev];
    AMREX_D_TERM(const MultiFab& bxcoef = m_b_coeffs[amrlev][mglev][0];,
                 const MultiFab& bycoef = m_b_coeffs[amrlev][mglev][1];,
                 const MultiFab& bzcoef = m_b_coeffs[amrlev][mglev][2];);
    AMREX_D_TERM(const MultiFab& cxcoef = m_c_coeffs[amrlev][mglev][0];,
                 const MultiFab& cycoef = m_c_coeffs[amrlev][mglev][1];,
                 const MultiFab& czcoef = m_c_coeffs[amrlev][mglev][2];);

    const int nc = getNComp();
    const Real* h = m_geom[amrlev][mglev].CellSize();
    AMREX_D_TERM(const Real dhxsq = m_b_scalar/(h[0]*h[0]);,
                 const Real dhysq = m_b_scalar/(h[1]*h[1]);,
                 const Real dhzsq = m_b_scalar/(h[2]*h[2]));
    AMREX_D_TERM(const Real dhx = m_c_scalar/(h[0]);,
                 const Real dhy = m_c_scalar/(h[1]);,
                 const Real dhz = m_c_scalar/(h[2]));
    const Real alpha = m_a_scalar;

    MFItInfo mfi_info;
    if (Gpu::notInLaunchRegion()) mfi_info.EnableTiling().SetDynamic(true);

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(acoef,mfi_info); mfi.isValid(); ++mfi)
    {
        const Box& tbx = mfi.tilebox();

        const auto& afab    = acoef.array(mfi);
        AMREX_D_TERM(const auto& bxfab = bxcoef.array(mfi);,
                     const auto& byfab = bycoef.array(mfi);,
                     const auto& bzfab = bzcoef.array(mfi););
        AMREX_D_TERM(const auto& cxfab = cxcoef.array(mfi);,
                     const auto& cyfab = cycoef.array(mfi);,
                     const auto& czfab = czcoef.array(mfi););

        AMREX_LAUNCH_HOST_DEVICE_LAMBDA ( tbx, thread_box,
        {
            abeccec_chkdiag(thread_box, alpha, afab,
                            AMREX_D_DECL(dhxsq, dhysq, dhzsq),
                            AMREX_D_DECL(dhx, dhy, dhz),
                            AMREX_D_DECL(bxfab, byfab, bzfab),
                            AMREX_D_DECL(cxfab, cyfab, czfab),
                            nc, m_uw_order);
        });
    }
}

void
MLABecCecLaplacian::getDiagonal(MultiFab& diag)
{
    BL_PROFILE("MLABecCecLaplacian::getDiagonal()");

    int amrlev = 0.0;
    int mglev = 0.0;

    const MultiFab& acoef = m_a_coeffs[amrlev][mglev];
    AMREX_D_TERM(const MultiFab& bxcoef = m_b_coeffs[amrlev][mglev][0];,
                 const MultiFab& bycoef = m_b_coeffs[amrlev][mglev][1];,
                 const MultiFab& bzcoef = m_b_coeffs[amrlev][mglev][2];);
    AMREX_D_TERM(const MultiFab& cxcoef = m_c_coeffs[amrlev][mglev][0];,
                 const MultiFab& cycoef = m_c_coeffs[amrlev][mglev][1];,
                 const MultiFab& czcoef = m_c_coeffs[amrlev][mglev][2];);

    const int nc = getNComp();
    const Real* h = m_geom[amrlev][mglev].CellSize();
    AMREX_D_TERM(const Real dhxsq = m_b_scalar/(h[0]*h[0]);,
                 const Real dhysq = m_b_scalar/(h[1]*h[1]);,
                 const Real dhzsq = m_b_scalar/(h[2]*h[2]));
    AMREX_D_TERM(const Real dhx = m_c_scalar/(h[0]);,
                 const Real dhy = m_c_scalar/(h[1]);,
                 const Real dhz = m_c_scalar/(h[2]));
    const Real alpha = m_a_scalar;

    MFItInfo mfi_info;
    if (Gpu::notInLaunchRegion()) mfi_info.EnableTiling().SetDynamic(true);

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(acoef,mfi_info); mfi.isValid(); ++mfi)
    {
        const Box& tbx = mfi.tilebox();

        const auto& afab    = acoef.array(mfi);
        AMREX_D_TERM(const auto& bxfab = bxcoef.array(mfi);,
                     const auto& byfab = bycoef.array(mfi);,
                     const auto& bzfab = bzcoef.array(mfi););
        AMREX_D_TERM(const auto& cxfab = cxcoef.array(mfi);,
                     const auto& cyfab = cycoef.array(mfi);,
                     const auto& czfab = czcoef.array(mfi););
        const auto& adiag    = diag.array(mfi);

        AMREX_LAUNCH_HOST_DEVICE_LAMBDA ( tbx, thread_box,
        {
            abeccec_getdiag(thread_box, alpha, afab,
                            AMREX_D_DECL(dhxsq, dhysq, dhzsq),
                            AMREX_D_DECL(dhx, dhy, dhz),
                            AMREX_D_DECL(bxfab, byfab, bzfab),
                            AMREX_D_DECL(cxfab, cyfab, czfab),
                            nc, adiag);
        });
    }
}

void
MLABecCecLaplacian::getNetFaceCoeff(MultiFab& coeffs_dir,
                                  int dir)
{
    BL_PROFILE("MLABecCecLaplacian::getNetFaceCoeff()");

    int amrlev = 0.0;
    int mglev = 0.0;

    const MultiFab& acoef = m_a_coeffs[amrlev][mglev];
    AMREX_D_TERM(const MultiFab& bxcoef = m_b_coeffs[amrlev][mglev][0];,
                 const MultiFab& bycoef = m_b_coeffs[amrlev][mglev][1];,
                 const MultiFab& bzcoef = m_b_coeffs[amrlev][mglev][2];);
    AMREX_D_TERM(const MultiFab& cxcoef = m_c_coeffs[amrlev][mglev][0];,
                 const MultiFab& cycoef = m_c_coeffs[amrlev][mglev][1];,
                 const MultiFab& czcoef = m_c_coeffs[amrlev][mglev][2];);

    const int nc = getNComp();
    const Real* h = m_geom[amrlev][mglev].CellSize();
    AMREX_D_TERM(const Real dhxsq = m_b_scalar/(h[0]*h[0]);,
                 const Real dhysq = m_b_scalar/(h[1]*h[1]);,
                 const Real dhzsq = m_b_scalar/(h[2]*h[2]));
    AMREX_D_TERM(const Real dhx = m_c_scalar/(h[0]);,
                 const Real dhy = m_c_scalar/(h[1]);,
                 const Real dhz = m_c_scalar/(h[2]));
    const Real alpha = m_a_scalar;

    MFItInfo mfi_info;
    if (Gpu::notInLaunchRegion()) mfi_info.EnableTiling().SetDynamic(true);

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(acoef,mfi_info); mfi.isValid(); ++mfi)
    {
        const Box& tbx = mfi.tilebox();

        const auto& afab    = acoef.array(mfi);
        AMREX_D_TERM(const auto& bxfab = bxcoef.array(mfi);,
                     const auto& byfab = bycoef.array(mfi);,
                     const auto& bzfab = bzcoef.array(mfi););
        AMREX_D_TERM(const auto& cxfab = cxcoef.array(mfi);,
                     const auto& cyfab = cycoef.array(mfi);,
                     const auto& czfab = czcoef.array(mfi););
        const auto& coeffd    = coeffs_dir.array(mfi);

        AMREX_LAUNCH_HOST_DEVICE_LAMBDA ( tbx, thread_box,
        {
            abeccec_getnetfcoeff(thread_box, alpha, afab,
                                 AMREX_D_DECL(dhxsq, dhysq, dhzsq),
                                 AMREX_D_DECL(dhx, dhy, dhz),
                                 AMREX_D_DECL(bxfab, byfab, bzfab),
                                 AMREX_D_DECL(cxfab, cyfab, czfab),
                                 nc, dir, coeffd);
        });
    }
}

void
MLABecCecLaplacian::FFlux (int amrlev, const MFIter& mfi,
                           const Array<FArrayBox*,AMREX_SPACEDIM>& flux,
                           const FArrayBox& sol, Location, const int face_only) const
{
    // FIXME: make second order
    BL_PROFILE("MLABecCecLaplacian::FFlux()");

    const int mglev = 0;
    const Box& box = mfi.tilebox();
    const Real* dxinv = m_geom[amrlev][mglev].InvCellSize();
    const int ncomp = getNComp();
    FFlux(box, dxinv, m_b_scalar, m_c_scalar,
          Array<FArrayBox const*,AMREX_SPACEDIM>{{AMREX_D_DECL(&(m_b_coeffs[amrlev][mglev][0][mfi]),
                                                               &(m_b_coeffs[amrlev][mglev][1][mfi]),
                                                               &(m_b_coeffs[amrlev][mglev][2][mfi]))}},
          Array<FArrayBox const*,AMREX_SPACEDIM>{{AMREX_D_DECL(&(m_c_coeffs[amrlev][mglev][0][mfi]),
                                                               &(m_c_coeffs[amrlev][mglev][1][mfi]),
                                                               &(m_c_coeffs[amrlev][mglev][2][mfi]))}},
          flux, sol, face_only, ncomp, m_uw_order);
}

void
MLABecCecLaplacian::FFlux (Box const& box, Real const* dxinv, Real bscalar, Real cscalar,
                           Array<FArrayBox const*, AMREX_SPACEDIM> const& bcoef,
                           Array<FArrayBox const*, AMREX_SPACEDIM> const& ccoef,
                           Array<FArrayBox*,AMREX_SPACEDIM> const& flux,
                           FArrayBox const& sol, int face_only, int ncomp, int const uw_order)
{
    // FIXME: make second order

    AMREX_D_TERM(const auto bx = bcoef[0]->array();,
                 const auto by = bcoef[1]->array();,
                 const auto bz = bcoef[2]->array(););
    AMREX_D_TERM(const auto cx = ccoef[0]->array();,
                 const auto cy = ccoef[1]->array();,
                 const auto cz = ccoef[2]->array(););
    AMREX_D_TERM(const auto& fxarr = flux[0]->array();,
                 const auto& fyarr = flux[1]->array();,
                 const auto& fzarr = flux[2]->array(););
    const auto& solarr = sol.array();

    if (face_only)
    {
        Real fac = bscalar*dxinv[0];
        Real facc = cscalar;
        Box blo = amrex::bdryLo(box, 0);
        int blen = box.length(0);
        AMREX_LAUNCH_HOST_DEVICE_LAMBDA ( blo, tbox,
        {
            mlabecceclap_flux_xface(tbox, fxarr, solarr, bx, cx, fac, facc, blen, ncomp, uw_order);
        });
#if (AMREX_SPACEDIM >= 2)
        fac = bscalar*dxinv[1];
        blo = amrex::bdryLo(box, 1);
        blen = box.length(1);
        AMREX_LAUNCH_HOST_DEVICE_LAMBDA ( blo, tbox,
        {
            mlabecceclap_flux_yface(tbox, fyarr, solarr, by, cy, fac, facc, blen, ncomp, uw_order);
        });
#endif
#if (AMREX_SPACEDIM == 3)
        fac = bscalar*dxinv[2];
        blo = amrex::bdryLo(box, 2);
        blen = box.length(2);
        AMREX_LAUNCH_HOST_DEVICE_LAMBDA ( blo, tbox,
        {
            mlabecceclap_flux_zface(tbox, fzarr, solarr, bz, cz, fac, facc, blen, ncomp, uw_order);
        });
#endif
    }
    else
    {
        Real fac = bscalar*dxinv[0];
        Real facc = cscalar;
        Box bflux = amrex::surroundingNodes(box, 0);
        AMREX_LAUNCH_HOST_DEVICE_LAMBDA ( bflux, tbox,
        {
            mlabecceclap_flux_x(tbox, fxarr, solarr, bx, cx, fac, facc, ncomp, uw_order);
        });
#if (AMREX_SPACEDIM >= 2)
        fac = bscalar*dxinv[1];
        bflux = amrex::surroundingNodes(box, 1);
        AMREX_LAUNCH_HOST_DEVICE_LAMBDA ( bflux, tbox,
        {
            mlabecceclap_flux_y(tbox, fyarr, solarr, by, cy, fac, facc, ncomp, uw_order);
        });
#endif
#if (AMREX_SPACEDIM == 3)
        fac = bscalar*dxinv[2];
        bflux = amrex::surroundingNodes(box, 2);
        AMREX_LAUNCH_HOST_DEVICE_LAMBDA ( bflux, tbox,
        {
            mlabecceclap_flux_z(tbox, fzarr, solarr, bz, cz, fac, facc, ncomp, uw_order);
        });
#endif
    }
}

void
MLABecCecLaplacian::update ()
{
    if (MLCellABecCecLap::needsUpdate()) MLCellABecCecLap::update();

#if (AMREX_SPACEDIM != 3)
    applyMetricTermsCoeffs();
#endif

    averageDownCoeffs();

    m_is_singular.clear();
    m_is_singular.resize(m_num_amr_levels, false);
    auto itlo = std::find(m_lobc[0].begin(), m_lobc[0].end(), BCType::Dirichlet);
    auto ithi = std::find(m_hibc[0].begin(), m_hibc[0].end(), BCType::Dirichlet);
    if (itlo == m_lobc[0].end() && ithi == m_hibc[0].end())
    {  // No Dirichlet
        for (int alev = 0; alev < m_num_amr_levels; ++alev)
        {
            // For now this assumes that overset regions are treated as Dirichlet bc's
            if (m_domain_covered[alev] && !m_overset_mask[alev][0]) 
            {
                if (m_a_scalar == 0.0)
                {
                    m_is_singular[alev] = true;
                }
                else
                {
                    Real asum = m_a_coeffs[alev].back().sum();
                    Real amax = m_a_coeffs[alev].back().norm0();
                    m_is_singular[alev] = (asum <= amax * 1.e-12);
                }
            }
        }
    }

    m_needs_update = false;
}

void
MLABecCecLaplacian::applyOverset (int amrlev, Any& rhs) const
{

    AMREX_ASSERT(rhs.is<MultiFab>());
    MultiFab& rhsm = rhs.get<MultiFab>();

    if (m_overset_mask[amrlev][0]) {
        const int ncomp = getNComp();
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(*m_overset_mask[amrlev][0],TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box& bx = mfi.tilebox();
            Array4<Real> const& rfab = rhsm.array(mfi);
            Array4<int const> const& osm = m_overset_mask[amrlev][0]->const_array(mfi);
            AMREX_HOST_DEVICE_PARALLEL_FOR_4D(bx, ncomp, i, j, k, n,
            {
                if (osm(i,j,k) == 0) rfab(i,j,k,n) = 0.0;
            });
        }
    }
}

}
