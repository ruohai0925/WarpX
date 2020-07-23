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

void FiniteDifferenceSolver::MacroscopicEvolveM_2nd (
    // The MField here is a vector of three multifabs, with M on each face, and each multifab is a three-component multifab.
    // Each M-multifab has three components, one for each component in x, y, z. (All multifabs are four dimensional, (i,j,k,n)), where, n=1 for E, B, but, n=3 for M_xface, M_yface, M_zface
    std::array< std::unique_ptr<amrex::MultiFab>, 3 >& Mfield, // Mfield contains three components MultiFab
    std::array< std::unique_ptr<amrex::MultiFab>, 3 >& H_biasfield, // H bias
    std::array< std::unique_ptr<amrex::MultiFab>, 3 > const& Bfield,
    std::array< std::unique_ptr<amrex::MultiFab>, 3 > const& Bfield_old,
    amrex::Real const dt,
    std::unique_ptr<MacroscopicProperties> const& macroscopic_properties) {

    if (m_fdtd_algo == MaxwellSolverAlgo::Yee)
    {
        MacroscopicEvolveMCartesian_2nd <CartesianYeeAlgorithm> (Mfield, H_biasfield, Bfield, Bfield_old, dt, macroscopic_properties);
    }
    else {
       amrex::Abort("Only yee algorithm is compatible for M updates.");
    }
    } // closes function MacroscopicEvolveM_2nd
#endif
#ifdef WARPX_MAG_LLG
    template<typename T_Algo>
    void FiniteDifferenceSolver::MacroscopicEvolveMCartesian_2nd (
        std::array< std::unique_ptr<amrex::MultiFab>, 3 > & Mfield,
        std::array< std::unique_ptr<amrex::MultiFab>, 3 >& H_biasfield, // H bias
        std::array< std::unique_ptr<amrex::MultiFab>, 3 > const& Bfield,
        std::array< std::unique_ptr<amrex::MultiFab>, 3 > const& Bfield_old,
        amrex::Real const dt,
        std::unique_ptr<MacroscopicProperties> const& macroscopic_properties )
    {

        // build temporary vector<multifab,3> Mfield_prev, Mfield_error, a_temp, a_temp_static, b_temp
        std::array< std::unique_ptr<amrex::MultiFab>, 3 > Mfield_prev;
        std::array< std::unique_ptr<amrex::MultiFab>, 3 > Mfield_error;
        std::array< std::unique_ptr<amrex::MultiFab>, 3 > a_temp;
        std::array< std::unique_ptr<amrex::MultiFab>, 3 > a_temp_static;
        std::array< std::unique_ptr<amrex::MultiFab>, 3 > b_temp;

        // initialize Mfield_previous
        for (int i = 0; i < 3; i++){
        Mfield_prev[i].reset( new MultiFab(Mfield[i]->boxArray(),Mfield[i]->DistributionMap(),3,Mfield[i]->nGrow()));
        Mfield_error[i].reset( new MultiFab(Mfield[i]->boxArray(),Mfield[i]->DistributionMap(),3,Mfield[i]->nGrow()));
        MultiFab::Copy(*Mfield_prev[i],*Mfield[i],0,0,3,Mfield[i]->nGrow());
        Mfield_error[i]->setVal(0.0);
        }
        // initialize a_temp, b_temp
        for (int i = 0; i < 3; i++){
        a_temp[i].reset( new MultiFab(Mfield[i]->boxArray(),Mfield[i]->DistributionMap(),3,Mfield[i]->nGrow()));
        a_temp_static[i].reset( new MultiFab(Mfield[i]->boxArray(),Mfield[i]->DistributionMap(),3,Mfield[i]->nGrow()));
        b_temp[i].reset( new MultiFab(Mfield[i]->boxArray(),Mfield[i]->DistributionMap(),3,Mfield[i]->nGrow()));
        a_temp[i]->setVal(0.0);
        a_temp_static[i]->setVal(0.0);
        b_temp[i]->setVal(0.0);
        }

        // calculate the b_temp, a_temp_static
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
            Array4<Real> const& Bx_old = Bfield_old[0]->array(mfi); // Bx is the x component at |_x faces
            Array4<Real> const& By_old = Bfield_old[1]->array(mfi); // By is the y component at |_y faces
            Array4<Real> const& Bz_old = Bfield_old[2]->array(mfi); // Bz is the z component at |_z faces

            // extract field data of a_temp_static and b_temp
            Array4<Real> const& a_temp_static_xface = a_temp_static[0]->array(mfi);
            Array4<Real> const& a_temp_static_yface = a_temp_static[1]->array(mfi);
            Array4<Real> const& a_temp_static_zface = a_temp_static[2]->array(mfi);
            Array4<Real> const& b_temp_xface= b_temp[0]->array(mfi);
            Array4<Real> const& b_temp_yface= b_temp[1]->array(mfi);
            Array4<Real> const& b_temp_zface= b_temp[2]->array(mfi);

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

            // How to loop? Build a new MacroscopicEvolveM_second_order_scheme()?;
            // loop over cells and update fields
            amrex::ParallelFor(tbx, tby, tbz,
            [=] AMREX_GPU_DEVICE (int i, int j, int k){

              // when working on M_xface(i,j,k, 0:2) we have direct access to M_xface(i,j,k,0:2) and Hx(i,j,k)
              // Hy and Hz can be acquired by interpolation
              // H_maxwell
              Real Hx_xface = MacroscopicProperties::getH_Maxwell(i, j, k, 0, amrex::IntVect(1,0,0), amrex::IntVect(1,0,0), Bx_old, M_xface);
              Real Hy_xface = MacroscopicProperties::getH_Maxwell(i, j, k, 1, amrex::IntVect(0,1,0), amrex::IntVect(1,0,0), By_old, M_xface);
              Real Hz_xface = MacroscopicProperties::getH_Maxwell(i, j, k, 2, amrex::IntVect(0,0,1), amrex::IntVect(1,0,0), Bz_old, M_xface);
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
              Real a_temp_static_coeff = MacroscopicProperties::macro_avg_to_face(i,j,k,amrex::IntVect(1,0,0),mag_alpha_arr)
                              / MacroscopicProperties::macro_avg_to_face(i,j,k,amrex::IntVect(1,0,0),mag_Ms_arr);

              Real b_temp_coeff = PhysConst::mu0 * mag_gamma_interp *
                        (1.0 + std::pow(MacroscopicProperties::macro_avg_to_face(i,j,k,amrex::IntVect(1,0,0),mag_alpha_arr), 2.0))/ 2.0;

              // calculate a_temp_static_xface
              // x component on x-faces of grid
              a_temp_static_xface(i, j, k, 0) = a_temp_static_coeff * M_xface(i, j, k, 0);

              // y component on x-faces of grid
              a_temp_static_xface(i, j, k, 1) = a_temp_static_coeff * M_xface(i, j, k, 1);

              // z component on x-faces of grid
              a_temp_static_xface(i, j, k, 2) = a_temp_static_coeff * M_xface(i, j, k, 2);

              // calculate b_temp_xface
              // x component on x-faces of grid
              b_temp_xface(i, j, k, 0) = M_xface(i, j, k, 0) + dt * b_temp_coeff * ( M_xface(i, j, k, 1) * Hz_eff - M_xface(i, j, k, 2) * Hy_eff);

              // y component on x-faces of grid
              b_temp_xface(i, j, k, 1) = M_xface(i, j, k, 1) + dt * b_temp_coeff * ( M_xface(i, j, k, 2) * Hx_eff - M_xface(i, j, k, 0) * Hz_eff);

              // z component on x-faces of grid
              b_temp_xface(i, j, k, 2) = M_xface(i, j, k, 2) + dt * b_temp_coeff * ( M_xface(i, j, k, 0) * Hy_eff - M_xface(i, j, k, 1) * Hx_eff);
              },

            [=] AMREX_GPU_DEVICE (int i, int j, int k){

              // when working on M_yface(i,j,k,0:2) we have direct access to M_yface(i,j,k,0:2) and Hy(i,j,k)
              // Hy and Hz can be acquired by interpolation
              // H_maxwell
              Real Hx_yface = MacroscopicProperties::getH_Maxwell(i, j, k, 0, amrex::IntVect(1,0,0), amrex::IntVect(0,1,0), Bx_old, M_yface);
              Real Hy_yface = MacroscopicProperties::getH_Maxwell(i, j, k, 1, amrex::IntVect(0,1,0), amrex::IntVect(0,1,0), By_old, M_yface);
              Real Hz_yface = MacroscopicProperties::getH_Maxwell(i, j, k, 2, amrex::IntVect(0,0,1), amrex::IntVect(0,1,0), Bz_old, M_yface);
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
              Real a_temp_static_coeff = MacroscopicProperties::macro_avg_to_face(i,j,k,amrex::IntVect(0,1,0),mag_alpha_arr)
                              / MacroscopicProperties::macro_avg_to_face(i,j,k,amrex::IntVect(0,1,0),mag_Ms_arr);

              Real b_temp_coeff = PhysConst::mu0 * mag_gamma_interp *
                        (1.0 + std::pow(MacroscopicProperties::macro_avg_to_face(i,j,k,amrex::IntVect(0,1,0),mag_alpha_arr), 2.0))/ 2.0;

              // calculate a_temp_static_yface
              // x component on y-faces of grid
              a_temp_static_yface(i, j, k, 0) = a_temp_static_coeff * M_yface(i, j, k, 0);

              // y component on y-faces of grid
              a_temp_static_yface(i, j, k, 1) = a_temp_static_coeff * M_yface(i, j, k, 1);

              // z component on y-faces of grid
              a_temp_static_yface(i, j, k, 2) = a_temp_static_coeff * M_yface(i, j, k, 2);

              // calculate b_temp_yface
              // x component on y-faces of grid
              b_temp_yface(i, j, k, 0) = M_yface(i, j, k, 0) + dt * b_temp_coeff * ( M_yface(i, j, k, 1) * Hz_eff - M_yface(i, j, k, 2) * Hy_eff);

              // y component on y-faces of grid
              b_temp_yface(i, j, k, 1) = M_yface(i, j, k, 1) + dt * b_temp_coeff * ( M_yface(i, j, k, 2) * Hx_eff - M_yface(i, j, k, 0) * Hz_eff);

              // z component on y-faces of grid
              b_temp_yface(i, j, k, 2) = M_yface(i, j, k, 2) + dt * b_temp_coeff * ( M_yface(i, j, k, 0) * Hy_eff - M_yface(i, j, k, 1) * Hx_eff);
              },

            [=] AMREX_GPU_DEVICE (int i, int j, int k){

              // when working on M_zface(i,j,k,0:2) we have direct access to M_zface(i,j,k,0:2) and Hz(i,j,k)
              // Hy and Hz can be acquired by interpolation
              // H_maxwell
              Real Hx_zface = MacroscopicProperties::getH_Maxwell(i, j, k, 0, amrex::IntVect(1,0,0), amrex::IntVect(0,0,1), Bx_old, M_zface);
              Real Hy_zface = MacroscopicProperties::getH_Maxwell(i, j, k, 1, amrex::IntVect(0,1,0), amrex::IntVect(0,0,1), By_old, M_zface);
              Real Hz_zface = MacroscopicProperties::getH_Maxwell(i, j, k, 2, amrex::IntVect(0,0,1), amrex::IntVect(0,0,1), Bz_old, M_zface);
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
              Real a_temp_static_coeff = MacroscopicProperties::macro_avg_to_face(i,j,k,amrex::IntVect(0,0,1),mag_alpha_arr)
                              / MacroscopicProperties::macro_avg_to_face(i,j,k,amrex::IntVect(0,0,1),mag_Ms_arr);

              Real b_temp_coeff = PhysConst::mu0 * mag_gamma_interp *
                        (1.0 + std::pow(MacroscopicProperties::macro_avg_to_face(i,j,k,amrex::IntVect(0,0,1),mag_alpha_arr), 2.0))/ 2.0;

              // calculate a_temp_static_zface
              // x component on z-faces of grid
              a_temp_static_zface(i, j, k, 0) = a_temp_static_coeff * M_zface(i, j, k, 0);

              // y component on z-faces of grid
              a_temp_static_zface(i, j, k, 1) = a_temp_static_coeff * M_zface(i, j, k, 1);

              // z component on z-faces of grid
              a_temp_static_zface(i, j, k, 2) = a_temp_static_coeff * M_zface(i, j, k, 2);

              // calculate b_temp_zface
              // x component on z-faces of grid
              b_temp_zface(i, j, k, 0) = M_zface(i, j, k, 0) + dt * b_temp_coeff * ( M_zface(i, j, k, 1) * Hz_eff - M_zface(i, j, k, 2) * Hy_eff);

              // y component on z-faces of grid
              b_temp_zface(i, j, k, 1) = M_zface(i, j, k, 1) + dt * b_temp_coeff * ( M_zface(i, j, k, 2) * Hx_eff - M_zface(i, j, k, 0) * Hz_eff);

              // z component on z-faces of grid
              b_temp_zface(i, j, k, 2) = M_zface(i, j, k, 2) + dt * b_temp_coeff * ( M_zface(i, j, k, 0) * Hy_eff - M_zface(i, j, k, 1) * Hx_eff);
              });
        }

        // initialize M_max_iter, M_iter, M_tol, M_iter_error
        // change these to user-defined input in the next PR
        int M_max_iter = 100;
        int M_iter = 0;
        amrex::Real M_tol = 0.0001;
        amrex::Real M_iter_maxerror = -1.0;
        int stop_iter = 0;

        // begin the iteration
        while (!stop_iter) {
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

            // extract field data of Mfield_prev, Mfield_error, a_temp, a_temp_static, and b_temp
            Array4<Real> const& M_prev_xface = Mfield_prev[0]->array(mfi);
            Array4<Real> const& M_prev_yface = Mfield_prev[1]->array(mfi);
            Array4<Real> const& M_prev_zface = Mfield_prev[2]->array(mfi);
            Array4<Real> const& M_error_xface = Mfield_error[0]->array(mfi);
            Array4<Real> const& M_error_yface = Mfield_error[1]->array(mfi);
            Array4<Real> const& M_error_zface = Mfield_error[2]->array(mfi);
            Array4<Real> const& a_temp_xface = a_temp[0]->array(mfi);
            Array4<Real> const& a_temp_yface = a_temp[1]->array(mfi);
            Array4<Real> const& a_temp_zface = a_temp[2]->array(mfi);
            Array4<Real> const& a_temp_static_xface = a_temp_static[0]->array(mfi);
            Array4<Real> const& a_temp_static_yface = a_temp_static[1]->array(mfi);
            Array4<Real> const& a_temp_static_zface = a_temp_static[2]->array(mfi);
            Array4<Real> const& b_temp_xface= b_temp[0]->array(mfi);
            Array4<Real> const& b_temp_yface= b_temp[1]->array(mfi);
            Array4<Real> const& b_temp_zface= b_temp[2]->array(mfi);

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

            // reset the value of the M_iter_maxerror
            M_iter_maxerror = -1.0;

            // loop over cells and update fields
            amrex::ParallelFor(tbx, tby, tbz,
            [=] AMREX_GPU_DEVICE (int i, int j, int k){

              // when working on M_xface(i,j,k, 0:2) we have direct access to M_xface(i,j,k,0:2) and Hx(i,j,k)
              // Hy and Hz can be acquired by interpolation
              // H_maxwell
              Real Hx_xface = MacroscopicProperties::getH_Maxwell(i, j, k, 0, amrex::IntVect(1,0,0), amrex::IntVect(1,0,0), Bx, M_prev_xface);
              Real Hy_xface = MacroscopicProperties::getH_Maxwell(i, j, k, 1, amrex::IntVect(0,1,0), amrex::IntVect(1,0,0), By, M_prev_xface);
              Real Hz_xface = MacroscopicProperties::getH_Maxwell(i, j, k, 2, amrex::IntVect(0,0,1), amrex::IntVect(1,0,0), Bz, M_prev_xface);
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

              Real a_temp_dynamic_coeff = PhysConst::mu0 * std::abs(mag_gamma_interp) *
                        (1.0 + std::pow(MacroscopicProperties::macro_avg_to_face(i,j,k,amrex::IntVect(1,0,0),mag_alpha_arr), 2.0))/ 2.0;

          // calculate a_temp_xface
              // x component on x-faces of grid
              a_temp_xface(i, j, k, 0) = -( dt * a_temp_dynamic_coeff * Hx_eff + a_temp_static_xface(i, j, k, 0) );

              // y component on x-faces of grid
              a_temp_xface(i, j, k, 1) = -( dt * a_temp_dynamic_coeff * Hy_eff + a_temp_static_xface(i, j, k, 1) );

              // z component on x-faces of grid
              a_temp_xface(i, j, k, 2) = -( dt * a_temp_dynamic_coeff * Hz_eff + a_temp_static_xface(i, j, k, 2) );

              // calculate M_xface
              // x component on x-faces of grid
              M_xface(i, j, k, 0) = MacroscopicProperties::getM_half (i, j, k, 0, a_temp_xface, b_temp_xface);

              // y component on x-faces of grid
              M_xface(i, j, k, 1) = MacroscopicProperties::getM_half (i, j, k, 1, a_temp_xface, b_temp_xface);

              // z component on x-faces of grid
              M_xface(i, j, k, 2) = MacroscopicProperties::getM_half (i, j, k, 2, a_temp_xface, b_temp_xface);

              // x component on x-faces of grid
              M_error_xface(i, j, k, 0) = std::abs((M_xface(i, j, k, 0) - M_prev_xface(i, j, k, 0))) / (std::abs(M_prev_xface(i, j, k, 0)) + 1.0e-12);

              // y component on x-faces of grid
              M_error_xface(i, j, k, 1) = std::abs((M_xface(i, j, k, 1) - M_prev_xface(i, j, k, 1))) / (std::abs(M_prev_xface(i, j, k, 1)) + 1.0e-12);

              // z component on x-faces of grid
              M_error_xface(i, j, k, 2) = std::abs((M_xface(i, j, k, 2) - M_prev_xface(i, j, k, 2))) / (std::abs(M_prev_xface(i, j, k, 2)) + 1.0e-12);

              },

            [=] AMREX_GPU_DEVICE (int i, int j, int k){

              // when working on M_yface(i,j,k,0:2) we have direct access to M_yface(i,j,k,0:2) and Hy(i,j,k)
              // Hy and Hz can be acquired by interpolation
              // H_maxwell
              Real Hx_yface = MacroscopicProperties::getH_Maxwell(i, j, k, 0, amrex::IntVect(1,0,0), amrex::IntVect(0,1,0), Bx, M_prev_yface);
              Real Hy_yface = MacroscopicProperties::getH_Maxwell(i, j, k, 1, amrex::IntVect(0,1,0), amrex::IntVect(0,1,0), By, M_prev_yface);
              Real Hz_yface = MacroscopicProperties::getH_Maxwell(i, j, k, 2, amrex::IntVect(0,0,1), amrex::IntVect(0,1,0), Bz, M_prev_yface);
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

              Real a_temp_dynamic_coeff = PhysConst::mu0 * std::abs(mag_gamma_interp) *
                        (1.0 + std::pow(MacroscopicProperties::macro_avg_to_face(i,j,k,amrex::IntVect(0,1,0),mag_alpha_arr), 2.0))/ 2.0;

              // calculate a_temp_yface
              // x component on y-faces of grid
              a_temp_yface(i, j, k, 0) = -( dt * a_temp_dynamic_coeff * Hx_eff + a_temp_static_yface(i, j, k, 0) );

              // y component on y-faces of grid
              a_temp_yface(i, j, k, 1) = -( dt * a_temp_dynamic_coeff * Hy_eff + a_temp_static_yface(i, j, k, 1) );

              // z component on y-faces of grid
              a_temp_yface(i, j, k, 2) = -( dt * a_temp_dynamic_coeff * Hz_eff + a_temp_static_yface(i, j, k, 2) );

              // calculate M_yface
              // x component on y-faces of grid
              M_yface(i, j, k, 0) = MacroscopicProperties::getM_half (i, j, k, 0, a_temp_yface, b_temp_yface);

              // y component on y-faces of grid
              M_yface(i, j, k, 1) = MacroscopicProperties::getM_half (i, j, k, 1, a_temp_yface, b_temp_yface);

              // z component on y-faces of grid
              M_yface(i, j, k, 2) = MacroscopicProperties::getM_half (i, j, k, 2, a_temp_yface, b_temp_yface);

              // x component on y-faces of grid
              M_error_yface(i, j, k, 0) = std::abs((M_yface(i, j, k, 0) - M_prev_yface(i, j, k, 0))) / (std::abs(M_prev_yface(i, j, k, 0)) + 1.0e-12);

              // y component on y-faces of grid
              M_error_yface(i, j, k, 1) = std::abs((M_yface(i, j, k, 1) - M_prev_yface(i, j, k, 1))) / (std::abs(M_prev_yface(i, j, k, 1)) + 1.0e-12);

              // z component on y-faces of grid
              M_error_yface(i, j, k, 2) = std::abs((M_yface(i, j, k, 2) - M_prev_yface(i, j, k, 2))) / (std::abs(M_prev_yface(i, j, k, 2)) + 1.0e-12);
              },

            [=] AMREX_GPU_DEVICE (int i, int j, int k){

              // when working on M_zface(i,j,k,0:2) we have direct access to M_zface(i,j,k,0:2) and Hz(i,j,k)
              // Hy and Hz can be acquired by interpolation
              // H_maxwell
              Real Hx_zface = MacroscopicProperties::getH_Maxwell(i, j, k, 0, amrex::IntVect(1,0,0), amrex::IntVect(0,0,1), Bx, M_prev_zface);
              Real Hy_zface = MacroscopicProperties::getH_Maxwell(i, j, k, 1, amrex::IntVect(0,1,0), amrex::IntVect(0,0,1), By, M_prev_zface);
              Real Hz_zface = MacroscopicProperties::getH_Maxwell(i, j, k, 2, amrex::IntVect(0,0,1), amrex::IntVect(0,0,1), Bz, M_prev_zface);
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

              Real a_temp_dynamic_coeff = PhysConst::mu0 * std::abs(mag_gamma_interp) *
                        (1.0 + std::pow(MacroscopicProperties::macro_avg_to_face(i,j,k,amrex::IntVect(0,0,1),mag_alpha_arr), 2.0))/ 2.0;

              // calculate a_temp_zface
              // x component on z-faces of grid
              a_temp_zface(i, j, k, 0) = -( dt * a_temp_dynamic_coeff * Hx_eff + a_temp_static_zface(i, j, k, 0) );

              // y component on z-faces of grid
              a_temp_zface(i, j, k, 1) = -( dt * a_temp_dynamic_coeff * Hy_eff + a_temp_static_zface(i, j, k, 1) );

              // z component on z-faces of grid
              a_temp_zface(i, j, k, 2) = -( dt * a_temp_dynamic_coeff * Hz_eff + a_temp_static_zface(i, j, k, 2) );

              // calculate M_zface
              // x component on z-faces of grid
              M_zface(i, j, k, 0) = MacroscopicProperties::getM_half (i, j, k, 0, a_temp_zface, b_temp_zface);

              // y component on z-faces of grid
              M_zface(i, j, k, 1) = MacroscopicProperties::getM_half (i, j, k, 1, a_temp_zface, b_temp_zface);

              // z component on z-faces of grid
              M_zface(i, j, k, 2) = MacroscopicProperties::getM_half (i, j, k, 2, a_temp_zface, b_temp_zface);

              // x component on z-faces of grid
              M_error_zface(i, j, k, 0) = std::abs((M_zface(i, j, k, 0) - M_prev_zface(i, j, k, 0))) / (std::abs(M_prev_zface(i, j, k, 0)) + 1.0e-12);

              // y component on z-faces of grid
              M_error_zface(i, j, k, 1) = std::abs((M_zface(i, j, k, 1) - M_prev_zface(i, j, k, 1))) / (std::abs(M_prev_zface(i, j, k, 1)) + 1.0e-12);

              // z component on z-faces of grid
              M_error_zface(i, j, k, 2) = std::abs((M_zface(i, j, k, 2) - M_prev_zface(i, j, k, 2))) / (std::abs(M_prev_zface(i, j, k, 2)) + 1.0e-12);
              });
        }

        // Check the error between Mfield and Mfield_prev and decide whether another iteration is needed
        for (int i = 0; i < 1; i++){
            for (int j = 0; j < 3; j++){
                Real M_iter_error = Mfield_error[i]->norm0(j);
                if(M_iter_error >= M_iter_maxerror) {
                     M_iter_maxerror = M_iter_error;
                }
            }
        }

        if (M_iter_maxerror <= M_tol) {
           stop_iter = 1;
        }
        else {
             // Copy Mfield to Mfield_previous
             for (int i = 0; i < 3; i++){
             MultiFab::Copy(*Mfield_prev[i],*Mfield[i],0,0,3,Mfield[i]->nGrow());
             }
        }

        if(M_iter >= M_max_iter) {
           amrex::Abort("The M_iter exceeds the M_max_iter");
           amrex::Print() << "The M_iter = " << M_iter << " exceeds the M_max_iter = " << M_max_iter << std::endl;
 	}
        else {
           M_iter++;
           amrex::Print() << "Finish " << M_iter << " times iteration with M_iter_maxerror = " << M_iter_maxerror << " and M_tol = " << M_tol << std::endl;
        }

        }
        // end the iteration
    }
#endif
