#ifndef AMREX_EB2_IF_TRUNCATEDCONE_H_
#define AMREX_EB2_IF_TRUNCATEDCONE_H_

#include <AMReX_Array.H>
#include <AMReX_EB2_IF_Base.H>

// For all implicit functions, >0: body; =0: boundary; <0: fluid

namespace amrex { namespace EB2 {

class TruncatedConeIF
    : public GPUable
{
public:

    // inside: is the fluid inside the truncated cone?
    TruncatedConeIF (Real a_radius1, Real a_radius2, Real a_height, const RealArray& a_base, int a_dir, bool a_inside)
        : m_radius1(a_radius1),
          m_radius2(a_radius2),
          m_height(a_height),
          m_base(makeXDim3(a_base)),
          m_dir(a_dir),
          m_sign( a_inside ? 1.0 : -1.0 )
        {}

    TruncatedConeIF (const TruncatedConeIF& rhs) noexcept = default;
    TruncatedConeIF (TruncatedConeIF&& rhs) noexcept = default;
    TruncatedConeIF& operator= (const TruncatedConeIF& rhs) = delete;
    TruncatedConeIF& operator= (TruncatedConeIF&& rhs) = delete;

    AMREX_GPU_HOST_DEVICE inline
    Real operator() (AMREX_D_DECL(Real x, Real y, Real z)) const noexcept {

        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_radius2 < m_radius1, "truncated cone radius invalid - radius2 must be smaller than radius1");
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_height != 0, "truncated cone height must not be zero");

        Real r = 0.0, dist = 0.0;
        switch (m_dir) {
        case 0 :
        {
          Real top_coord = m_base.x + m_height;
          if(x >= amrex::min(m_base.x, top_coord) && x <= amrex::max(m_base.x, top_coord)){
            r = m_radius1 - ( (x - m_base.x) / (m_height)) * (m_radius1 - m_radius2);
#if (AMREX_SPACEDIM == 3)
            dist = r - std::pow((y-m_base.y)*(y-m_base.y) + (z-m_base.z)*(z-m_base.z), 0.5);
#elif (AMREX_SPACEDIM == 2)
            dist = r - abs(y-m_base.y);
#endif
          }
          else{
            dist = -1.0;
          }
            break;
        }
        case 1 :
        {
          Real top_coord = m_base.y + m_height;
          if(y >= amrex::min(m_base.y, top_coord) && y <= amrex::max(m_base.y, top_coord)){
            r = m_radius1 - ( (y - m_base.y) / (m_height)) * (m_radius1 - m_radius2);
#if (AMREX_SPACEDIM == 3)
            dist = r - std::pow((x-m_base.x)*(x-m_base.x) + (z-m_base.z)*(z-m_base.z), 0.5);
#elif (AMREX_SPACEDIM == 2)
            dist = r - abs(x-m_base.x);
#endif
          }
          else{
            dist = -1.0;
          }
            break;
        }
        default:
        {
          Real top_coord = m_base.x + m_height;
          if(z >= amrex::min(m_base.z, top_coord) && z <= amrex::max(m_base.z, top_coord)){
            r = m_radius1 - ( (z - m_base.z) / (m_height)) * (m_radius1 - m_radius2);
#if (AMREX_SPACEDIM == 3)
            dist = r - std::pow((y-m_base.y)*(y-m_base.y) + (x-m_base.x)*(x-m_base.x), 0.5);
#endif
          }
          else{
            dist = -1.0;
          }
            break;
        }
        }
        // printf("Coords are (%.2f , %.2f, %.2f) : dist = %.6f\n", x, y, z, dist);
        return -1.0 * m_sign * dist;
    }

    inline Real operator() (const RealArray& p) const noexcept {
        return this->operator()(AMREX_D_DECL(p[0],p[1],p[2]));
    }

protected:
  
    Real  m_radius1;
    Real  m_radius2;
    Real  m_height;
    XDim3 m_base;
    int m_dir;
    Real  m_sign;
};

}}

#endif
