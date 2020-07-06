#include <AMReX.H>
#include <AMReX_EB2.H>
#include <AMReX_EB2_IF.H>
#include <AMReX_ParmParse.H>

using namespace amrex;

int main (int argc, char* argv[])
{
  amrex::Initialize(argc, argv);
  {
    const Real sphere_radius = 0.1;
     
    const RealBox rb({D_DECL(-1.0,-1.0,-1.0)}, {D_DECL(1.0,1.0,1.0)});
    const Array<int,AMREX_SPACEDIM> is_periodic{D_DECL(false, false, false)};
    Geometry::Setup(&rb, 0, is_periodic.data());
    Geometry geom;

    int i;
    ParmParse pp;
    pp.get("i",i);
    AMREX_ALWAYS_ASSERT(i>0 && i<6);
    {
      const Real timer_init = amrex::second();
      const int n_levels = 6+i;
      const int n_cells = std::pow(2,n_levels + 1);
      const Box domain(IntVect(D_DECL(0,0,0)),
                       IntVect(D_DECL(n_cells-1,n_cells-1,n_cells-1)));
      geom.define(domain);
 
      EB2::SphereIF sphere(0.1, {D_DECL(0.0,0.0,0.0)}, false);
      auto gshop = EB2::makeShop(sphere); 
      EB2::Build(gshop, geom, amrex::max(0,amrex::min(i+3,6)), 4, 1);

      const Real timer_end = amrex::second();        
      const Real timer_tot = timer_end - timer_init;

      Print() << std::endl;
      Print() << "Number of cells = " << n_cells << std::endl; 
      Print() << "Time = "<< timer_tot << std::endl;
    }

    Print() << "Max memory usage: " << Real(TotalBytesAllocatedInFabsHWM())/(1024*1024) << " MB/rank" << std::endl;
    amrex::Finalize();
  }
}
