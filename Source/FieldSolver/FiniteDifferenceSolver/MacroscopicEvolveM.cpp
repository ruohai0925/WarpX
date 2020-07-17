/* copyright
blank
*/

#include "Utils/WarpXAlgorithmSelection.H"
#include "FiniteDifferenceSolver.H"
#include "FiniteDifferenceAlgorithms/CartesianYeeAlgorithm.H"
#include "FiniteDifferenceAlgorithms/CartesianCKCAlgorithm.H"
#include "FiniteDifferenceAlgorithms/CartesianNodalAlgorithm.H"

#include "Utils/WarpXConst.H"
#include <AMReX_Gpu.H>

using namespace amrex;

#ifdef WARPX_MAG_LLG
// update M field over one timestep

void FiniteDifferenceSolver::MacroscopicEvolveM (
    std::array< std::unique_ptr<amrex::MultiFab>, 3 >& Mfield, // Mfield contains three components MultiFab
    std::array< std::unique_ptr<amrex::MultiFab>, 3 >& H_biasfield, // H bias 
    std::array< std::unique_ptr<amrex::MultiFab>, 3 > const& Bfield,
    amrex::Real const dt,
    std::unique_ptr<MacroscopicProperties> const& macroscopic_properties) {

    /* if (m_do_nodal) {

        EvolveMCartesian <CartesianNodalAlgorithm> ( Mfield, Bfield, dt );

    } else if (m_fdtd_algo == MaxwellSolverAlgo::Yee) {

        EvolveMCartesian <CartesianYeeAlgorithm> ( Mfield, Bfield, dt );

    } else if (m_fdtd_algo == MaxwellSolverAlgo::CKC) {

        EvolveMCartesian <CartesianCKCAlgorithm> ( Mfield, Bfield, dt );
    }
    else
    {
        amrex::Abort("Unknown algorithm");
    } */

    if (m_fdtd_algo == MaxwellSolverAlgo::Yee)
    {
        MacroscopicEvolveMCartesian <CartesianYeeAlgorithm> (Mfield, H_biasfield, Bfield, dt, macroscopic_properties);
    }
    else {
       amrex::Abort("Only yee algorithm is compatible for M updates.");
    }
    } // closes function EvolveM
#endif
#ifdef WARPX_MAG_LLG
    template<typename T_Algo>
    void FiniteDifferenceSolver::MacroscopicEvolveMCartesian (
        std::array< std::unique_ptr<amrex::MultiFab>, 3 > & Mfield,
        std::array< std::unique_ptr<amrex::MultiFab>, 3 >& H_biasfield, // H bias
        std::array< std::unique_ptr<amrex::MultiFab>, 3 > const& Bfield,
        amrex::Real const dt,
        std::unique_ptr<MacroscopicProperties> const& macroscopic_properties )
    {
        // static constexpr amrex::Real alpha = 1e-4;
        // static constexpr amrex::Real Ms = 1e4;
        // Real constexpr cons1 = -mag_gamma_interp; // should be mu0*gamma, mu0 is absorbed by B used in this case
        // Real constexpr cons2 = -cons1*alpha/Ms; // factor of the second term in scalar LLG

        for (MFIter mfi(*Mfield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi) /* remember to FIX */
        {
          auto& mag_Ms_mf = macroscopic_properties->getmag_Ms_mf();
          auto& mag_alpha_mf = macroscopic_properties->getmag_alpha_mf();
          auto& mag_gamma_mf = macroscopic_properties->getmag_gamma_mf();
          // exctract material properties
          Array4<Real> const& mag_Ms_arr = mag_Ms_mf.array(mfi);
          Array4<Real> const& mag_alpha_arr = mag_alpha_mf.array(mfi);
          Array4<Real> const& mag_gamma_arr = mag_gamma_mf.array(mfi);

            // extract field data
            Array4<Real> const& M_xface = Mfield[0]->array(mfi); // note M_xface include x,y,z components at |_x faces
            Array4<Real> const& M_yface = Mfield[1]->array(mfi); // note M_yface include x,y,z components at |_y faces
            Array4<Real> const& M_zface = Mfield[2]->array(mfi); // note M_zface include x,y,z components at |_z faces
            Array4<Real> const& Hx_bias = H_biasfield[0]->array(mfi); // Hx_bias is the x component at |_x faces
            Array4<Real> const& Hy_bias = H_biasfield[1]->array(mfi); // Hy_bias is the y component at |_y faces
            Array4<Real> const& Hz_bias = H_biasfield[2]->array(mfi); // Hz_bias is the z component at |_z faces
            Array4<Real> const& Bx = Bfield[0]->array(mfi); // Bx is the x component at |_x faces
            Array4<Real> const& By = Bfield[1]->array(mfi); // By is the y component at |_y faces
            Array4<Real> const& Bz = Bfield[2]->array(mfi); // Bz is the z component at |_z faces

            // extract stencil coefficients
            Real const * const AMREX_RESTRICT coefs_x = m_stencil_coefs_x.dataPtr();
            int const n_coefs_x = m_stencil_coefs_x.size();
            Real const * const AMREX_RESTRICT coefs_y = m_stencil_coefs_y.dataPtr();
            int const n_coefs_y = m_stencil_coefs_y.size();
            Real const * const AMREX_RESTRICT coefs_z = m_stencil_coefs_z.dataPtr();
            int const n_coefs_z = m_stencil_coefs_z.size();

            // extract tileboxes for which to loop
            Box const& tbx = mfi.tilebox(Bfield[0]->ixType().toIntVect()); /* just define which grid type */
            Box const& tby = mfi.tilebox(Bfield[1]->ixType().toIntVect());
            Box const& tbz = mfi.tilebox(Bfield[2]->ixType().toIntVect());

            // loop over cells and update fields
            amrex::ParallelFor(tbx, tby, tbz,
            [=] AMREX_GPU_DEVICE (int i, int j, int k){

              // when working on M_xface(i,j,k, 0:2) we have direct access to M_xface(i,j,k,0:2) and Hx(i,j,k)
              // Hy and Hz can be acquired by interpolation
              // H_maxwell
              Real Hx_xface = MacroscopicProperties::getH_Maxwell(i, j, k, 0, amrex::IntVect(1,0,0), amrex::IntVect(1,0,0), Bx, M_xface);
              Real Hy_xface = MacroscopicProperties::getH_Maxwell(i, j, k, 1, amrex::IntVect(0,1,0), amrex::IntVect(1,0,0), By, M_xface);
              Real Hz_xface = MacroscopicProperties::getH_Maxwell(i, j, k, 2, amrex::IntVect(0,0,1), amrex::IntVect(1,0,0), Bz, M_xface);
              // H_bias
              Real Hx_bias_xface = MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(1,0,0), amrex::IntVect(1,0,0), Hx_bias);
              Real Hy_bias_xface = MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(0,1,0), amrex::IntVect(1,0,0), Hy_bias);
              Real Hz_bias_xface = MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(0,0,1), amrex::IntVect(1,0,0), Hz_bias);
              // H_eff = H_maxwell + H_bias + H_exchange + H_anisotropy ... (only the first two terms are considered here)
Real Hx_eff = Hx_bias_xface;
Real Hy_eff = Hy_bias_xface;
Real Hz_eff = Hz_bias_xface;

              // Real Hx_eff = Hx_xface + Hx_bias_xface;
              // Real Hy_eff = Hy_xface + Hy_bias_xface;
              // Real Hz_eff = Hz_xface + Hz_bias_xface;

              // magnetic material properties mag_alpha and mag_Ms are defined at cell nodes
              // keep the interpolation
              Real mag_gamma_interp = MacroscopicProperties::macro_avg_to_face(i,j,k,amrex::IntVect(1,0,0),mag_gamma_arr); 
              Real Gil_damp = PhysConst::mu0 * mag_gamma_interp
                              * MacroscopicProperties::macro_avg_to_face(i,j,k,amrex::IntVect(1,0,0),mag_alpha_arr)
                              / MacroscopicProperties::macro_avg_to_face(i,j,k,amrex::IntVect(1,0,0),mag_Ms_arr);

              // now you have access to use M_xface(i,j,k,0) M_xface(i,j,k,1), M_xface(i,j,k,2), Hx(i,j,k), Hy, Hz on the RHS of these update lines below
              // x component on x-faces of grid
              M_xface(i, j, k, 0) += dt * (-PhysConst::mu0 * mag_gamma_interp) * ( M_xface(i, j, k, 1) * Hz_eff - M_xface(i, j, k, 2) * Hy_eff)
                - dt * Gil_damp * ( M_xface(i, j, k, 1) * (M_xface(i, j, k, 0) * Hy_eff - M_xface(i, j, k, 1) * Hx_eff)
                - M_xface(i, j, k, 2) * ( M_xface(i, j, k, 2) * Hx_eff - M_xface(i, j, k, 0) * Hz_eff));

              // y component on x-faces of grid
              M_xface(i, j, k, 1) += dt * (-PhysConst::mu0 * mag_gamma_interp) * ( M_xface(i, j, k, 2) * Hx_eff - M_xface(i, j, k, 0) * Hz_eff)
                - dt * Gil_damp * ( M_xface(i, j, k, 2) * (M_xface(i, j, k, 1) * Hz_eff - M_xface(i, j, k, 2) * Hy_eff)
                - M_xface(i, j, k, 0) * ( M_xface(i, j, k, 0) * Hy_eff - M_xface(i, j, k, 1) * Hx_eff));

              // z component on x-faces of grid
              M_xface(i, j, k, 2) += dt * (-PhysConst::mu0 * mag_gamma_interp) * ( M_xface(i, j, k, 0) * Hy_eff - M_xface(i, j, k, 1) * Hx_eff)
                - dt * Gil_damp * ( M_xface(i, j, k, 0) * ( M_xface(i, j, k, 2) * Hx_eff - M_xface(i, j, k, 0) * Hz_eff)
                - M_xface(i, j, k, 1) * ( M_xface(i, j, k, 1) * Hz_eff - M_xface(i, j, k, 2) * Hy_eff));

if(i==2 && j==2 && k==2) {
	      amrex::Print() << "    x " <<std::endl;
	      amrex::Print() << " i " << i << " j " << j << " k " << k <<  std::endl;
	      amrex::Print() << " Hx_xface " << Hx_xface << std::endl;
	      amrex::Print() << " Hy_xface " << Hy_xface << std::endl;
	      amrex::Print() << " Hz_xface " << Hz_xface << std::endl;
	      amrex::Print() << " Hx_bias_xface " << Hx_bias_xface << std::endl;
	      amrex::Print() << " Hy_bias_xface " << Hy_bias_xface << std::endl;
	      amrex::Print() << " Hz_bias_xface " << Hz_bias_xface << std::endl;
         amrex::Print() << " M_xface(i,j,k,0) " << M_xface(i,j,k,0) << std::endl;
	      amrex::Print() << " M_xface(i,j,k,1) " << M_xface(i,j,k,1) << std::endl;
	      amrex::Print() << " M_xface(i,j,k,2) " << M_xface(i,j,k,2) << std::endl;
	      amrex::Print() << " mag_gamma_interp " << mag_gamma_interp << std::endl;
}

              },

            [=] AMREX_GPU_DEVICE (int i, int j, int k){

              // when working on M_yface(i,j,k,0:2) we have direct access to M_yface(i,j,k,0:2) and Hy(i,j,k)
              // Hy and Hz can be acquired by interpolation
              // H_maxwell
              Real Hx_yface = MacroscopicProperties::getH_Maxwell(i, j, k, 0, amrex::IntVect(1,0,0), amrex::IntVect(0,1,0), Bx, M_yface);
              Real Hy_yface = MacroscopicProperties::getH_Maxwell(i, j, k, 1, amrex::IntVect(0,1,0), amrex::IntVect(0,1,0), By, M_yface);
              Real Hz_yface = MacroscopicProperties::getH_Maxwell(i, j, k, 2, amrex::IntVect(0,0,1), amrex::IntVect(0,1,0), Bz, M_yface);
              // H_bias
              Real Hx_bias_yface = MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(1,0,0), amrex::IntVect(0,1,0), Hx_bias);
              Real Hy_bias_yface = MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(0,1,0), amrex::IntVect(0,1,0), Hy_bias);
              Real Hz_bias_yface = MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(0,0,1), amrex::IntVect(0,1,0), Hz_bias);
              // H_eff = H_maxwell + H_bias + H_exchange + H_anisotropy ... (only the first two terms are considered here)
              // Real Hx_eff = Hx_yface + Hx_bias_yface;
              // Real Hy_eff = Hy_yface + Hy_bias_yface;
              // Real Hz_eff = Hz_yface + Hz_bias_yface;
Real Hx_eff = Hx_bias_yface;
Real Hy_eff = Hy_bias_yface;
Real Hz_eff = Hz_bias_yface;

              // magnetic material properties mag_alpha and mag_Ms are defined at cell nodes
              // keep the interpolation
              Real mag_gamma_interp = MacroscopicProperties::macro_avg_to_face(i,j,k,amrex::IntVect(0,1,0),mag_gamma_arr); 
              Real Gil_damp = PhysConst::mu0 * mag_gamma_interp
                              * MacroscopicProperties::macro_avg_to_face(i,j,k,amrex::IntVect(0,1,0),mag_alpha_arr)
                              / MacroscopicProperties::macro_avg_to_face(i,j,k,amrex::IntVect(0,1,0),mag_Ms_arr);
 
              // x component on y-faces of grid
              M_yface(i, j, k, 0) += dt * (-PhysConst::mu0 * mag_gamma_interp) * ( M_yface(i, j, k, 1) * Hz_eff - M_yface(i, j, k, 2) * Hy_eff)
                - dt * Gil_damp * ( M_yface(i, j, k, 1) * (M_yface(i, j, k, 0) * Hy_eff - M_yface(i, j, k, 1) * Hx_eff)
                - M_yface(i, j, k, 2) * ( M_yface(i, j, k, 2) * Hx_eff - M_yface(i, j, k, 0) * Hz_eff));
	      
              // y component on y-faces of grid
              M_yface(i, j, k, 1) += dt * (-PhysConst::mu0 * mag_gamma_interp) * ( M_yface(i, j, k, 2) * Hx_eff - M_yface(i, j, k, 0) * Hz_eff)
                - dt * Gil_damp * ( M_yface(i, j, k, 2) * (M_yface(i, j, k, 1) * Hz_eff - M_yface(i, j, k, 2) * Hy_eff)
                - M_yface(i, j, k, 0) * ( M_yface(i, j, k, 0) * Hy_eff - M_yface(i, j, k, 1) * Hx_eff));

              // z component on y-faces of grid
              M_yface(i, j, k, 2) += dt * (-PhysConst::mu0 * mag_gamma_interp) * ( M_yface(i, j, k, 0) * Hy_eff - M_yface(i, j, k, 1) * Hx_eff)
                - dt * Gil_damp * ( M_yface(i, j, k, 0) * ( M_yface(i, j, k, 2) * Hx_eff - M_yface(i, j, k, 0) * Hz_eff)
                - M_yface(i, j, k, 1) * ( M_yface(i, j, k, 1) * Hz_eff - M_yface(i, j, k, 2) * Hy_eff));
 
              },

            [=] AMREX_GPU_DEVICE (int i, int j, int k){

              // when working on M_zface(i,j,k,0:2) we have direct access to M_zface(i,j,k,0:2) and Hz(i,j,k)
              // Hy and Hz can be acquired by interpolation
              // H_maxwell
              Real Hx_zface = MacroscopicProperties::getH_Maxwell(i, j, k, 0, amrex::IntVect(1,0,0), amrex::IntVect(0,0,1), Bx, M_zface);
              Real Hy_zface = MacroscopicProperties::getH_Maxwell(i, j, k, 1, amrex::IntVect(0,1,0), amrex::IntVect(0,0,1), By, M_zface);
              Real Hz_zface = MacroscopicProperties::getH_Maxwell(i, j, k, 2, amrex::IntVect(0,0,1), amrex::IntVect(0,0,1), Bz, M_zface);
              // H_bias
              Real Hx_bias_zface = MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(1,0,0), amrex::IntVect(0,0,1), Hx_bias);
              Real Hy_bias_zface = MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(0,1,0), amrex::IntVect(0,0,1), Hy_bias);
              Real Hz_bias_zface = MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(0,0,1), amrex::IntVect(0,0,1), Hz_bias);
              // H_eff = H_maxwell + H_bias + H_exchange + H_anisotropy ... (only the first two terms are considered here)
              // Real Hx_eff = Hx_zface + Hx_bias_zface;
              // Real Hy_eff = Hy_zface + Hy_bias_zface;
              // Real Hz_eff = Hz_zface + Hz_bias_zface;
Real Hx_eff = Hx_bias_zface;
Real Hy_eff = Hy_bias_zface;
Real Hz_eff = Hz_bias_zface;

              // magnetic material properties mag_alpha and mag_Ms are defined at cell nodes
              // keep the interpolation
              Real mag_gamma_interp = MacroscopicProperties::macro_avg_to_face(i,j,k,amrex::IntVect(0,0,1),mag_gamma_arr); 
              Real Gil_damp = PhysConst::mu0 * mag_gamma_interp
                              * MacroscopicProperties::macro_avg_to_face(i,j,k,amrex::IntVect(0,0,1),mag_alpha_arr)
                              / MacroscopicProperties::macro_avg_to_face(i,j,k,amrex::IntVect(0,0,1),mag_Ms_arr);
             
              // x component on z-faces of grid
              M_zface(i, j, k, 0) += dt * (-PhysConst::mu0 * mag_gamma_interp) * ( M_zface(i, j, k, 1) * Hz_eff - M_zface(i, j, k, 2) * Hy_eff)
                - dt * Gil_damp * ( M_zface(i, j, k, 1) * (M_zface(i, j, k, 0) * Hy_eff - M_zface(i, j, k, 1) * Hx_eff)
                - M_zface(i, j, k, 2) * ( M_zface(i, j, k, 2) * Hx_eff - M_zface(i, j, k, 0) * Hz_eff));

              // y component on z-faces of grid
              M_zface(i, j, k, 1) += dt * (-PhysConst::mu0 * mag_gamma_interp) * ( M_zface(i, j, k, 2) * Hx_eff - M_zface(i, j, k, 0) * Hz_eff)
                - dt * Gil_damp * ( M_zface(i, j, k, 2) * (M_zface(i, j, k, 1) * Hz_eff - M_zface(i, j, k, 2) * Hy_eff)
                - M_zface(i, j, k, 0) * ( M_zface(i, j, k, 0) * Hy_eff - M_zface(i, j, k, 1) * Hx_eff));

              // z component on z-faces of grid
              M_zface(i, j, k, 2) += dt * (-PhysConst::mu0 * mag_gamma_interp) * ( M_zface(i, j, k, 0) * Hy_eff - M_zface(i, j, k, 1) * Hx_eff)
                - dt * Gil_damp * ( M_zface(i, j, k, 0) * ( M_zface(i, j, k, 2) * Hx_eff - M_zface(i, j, k, 0) * Hz_eff)
                - M_zface(i, j, k, 1) * ( M_zface(i, j, k, 1) * Hz_eff - M_yface(i, j, k, 2) * Hy_eff)); 
              });
        }
    }
#endif
