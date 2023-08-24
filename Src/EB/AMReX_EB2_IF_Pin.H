#ifndef AMREX_EB2_IF_PIN_H_
#define AMREX_EB2_IF_PIN_H_

#include <AMReX_Array.H>
#include <AMReX_EB2_IF_Base.H>

// For all implicit functions, >0: body; =0: boundary; <0: fluid

namespace amrex { namespace EB2 {

class PinIF
    : public GPUable
{
public:

    // inside: is the fluid inside the pin?
    PinIF (Real a_c, const RealArray& a_tip, int a_dir,Real rmax, bool a_inside)
        : m_c(a_c),
          m_tip(makeXDim3(a_tip)),
          m_dir(a_dir),
          m_rmax(rmax),
          m_sign( a_inside ? 1.0 : -1.0 )
        {}

    PinIF (const PinIF& rhs) noexcept = default;
    PinIF (PinIF&& rhs) noexcept = default;
    PinIF& operator= (const PinIF& rhs) = delete;
    PinIF& operator= (PinIF&& rhs) = delete;

    AMREX_GPU_HOST_DEVICE inline
    Real operator() (AMREX_D_DECL(Real x, Real y, Real z)) const noexcept {
        Real r = 0.0, dist = 0.0;
        int c_sign = (m_c > 0) ? 1.0 : -1.0;
        switch (m_dir) {
        case 0 :
        {
#if (AMREX_SPACEDIM == 3)
            r = std::pow( (y-m_tip.y)*(y-m_tip.y)+(z-m_tip.z)*(z-m_tip.z), 0.5 );
            dist = x - (m_tip.x + m_c*r*r);
#elif (AMREX_SPACEDIM == 2)
            r = (y - m_tip.y);
            dist = x - (m_tip.x + m_c*r*r);
#endif
            break;
        }
        case 1 :
        {
#if (AMREX_SPACEDIM == 3)
            r = std::pow( (x-m_tip.x)*(x-m_tip.x)+(z-m_tip.z)*(z-m_tip.z), 0.5 );
            dist = y - (m_tip.y + m_c*r*r);
            // Added by Alfredo to limit the maximum radius of th pin
            Real zmax = m_tip.y+m_c*m_rmax*m_rmax;
            if ((c_sign<0)&&(y<zmax)) dist = (r*r-m_rmax*m_rmax);
            if ((c_sign>0)&&(y>zmax)) dist = -(r*r-m_rmax*m_rmax);
#elif (AMREX_SPACEDIM == 2)
            r = (x - m_tip.x);
            dist = y - (m_tip.y + m_c*r*r);
#endif
            break;
        }
        default:
        {
#if (AMREX_SPACEDIM == 3)
            r = std::pow( (x-m_tip.x)*(x-m_tip.x)+(y-m_tip.y)*(y-m_tip.y), 0.5 );
            dist = z - (m_tip.z + m_c*r*r);
#elif (AMREX_SPACEDIM == 2)
            r = 0.0;
            dist = 0.0
#endif
            break;
        }
        }
        //int c_sign = (m_c > 0) ? 1.0 : -1.0;
        // printf("Coords are (%.2f , %.2f, %.2f) : dist = %.6f\n", x, y, z, dist);
        return -1.0 * m_sign * dist * c_sign;
    }

    inline Real operator() (const RealArray& p) const noexcept {
        return this->operator()(AMREX_D_DECL(p[0],p[1],p[2]));
    }

protected:
  
    Real  m_c;
    Real  m_rmax;
    XDim3 m_tip;
    int m_dir;
    //
    Real  m_sign;
};

}}

#endif
