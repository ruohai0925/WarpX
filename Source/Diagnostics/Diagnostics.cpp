#include "Diagnostics.H"
#include "WarpX.H"
#include "Average.H"
#include "Utils/WarpXUtil.H"

using namespace amrex;

Diagnostics::Diagnostics (int i, std::string name)
    : diag_index(i), diag_name(name)
{
    ReadParameters();
}

Diagnostics::~Diagnostics ()
{
    delete m_flush_format;
}

void
Diagnostics::ReadParameters ()
{
    auto & warpx = WarpX::GetInstance();
    // Read list of fields requested by the user.
    ParmParse pp(diag_name);
    file_prefix = "diags/" + diag_name;
    pp.query("file_prefix", file_prefix);
    pp.query("period", m_period);
    pp.query("plot_raw_fields", m_plot_raw_fields);
    pp.query("plot_raw_fields_guards", m_plot_raw_fields_guards);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_plot_F==false, "cannot plot_F yet");
    if (!pp.queryarr("fields_to_plot", varnames)){
        varnames = {"Ex", "Ey", "Ez", "Bx", "By", "Bz", "jx", "jy", "jz"};
    }
    // set plot_rho to true of the users requests it, so that
    // rho is computed at each iteration.
    if (WarpXUtilStr::is_in(varnames, "rho")) warpx.setplot_rho(true);
    // Sanity check if user requests to plot F
    if (WarpXUtilStr::is_in(varnames, "F")){
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            warpx.do_dive_cleaning,
            "plot F only works if warpx.do_dive_cleaning = 1");
    }
    // If user requests to plot proc_number for a serial run,
    // delete proc_number from fields_to_plot
    if (ParallelDescriptor::NProcs() == 1){
        varnames.erase(
            std::remove(varnames.begin(), varnames.end(), "proc_number"),
            varnames.end());
    }
}

void
Diagnostics::InitData ()
{
    Print()<<"Diagnostics::InitData\n";
    auto & warpx = WarpX::GetInstance();
    nlev = warpx.finestLevel() + 1;
    // Initialize vector of pointers to the fields requested by the user.
    all_field_functors.resize( nlev );
    mf_avg.resize( nlev );

    for ( int lev=0; lev<nlev; lev++ ){
        all_field_functors[lev].resize( varnames.size() );
        // Fill vector of functors for all components except individual
        // cyclindrical modes
        for (int comp=0, n=all_field_functors[lev].size(); comp<n; comp++){
            if        ( varnames[comp] == "Ex" ){
                all_field_functors[lev][comp] = new CellCenterFunctor(warpx.get_pointer_Efield_aux(lev, 0), lev);
            } else if ( varnames[comp] == "Ey" ){
                all_field_functors[lev][comp] = new CellCenterFunctor(warpx.get_pointer_Efield_aux(lev, 1), lev);
            } else if ( varnames[comp] == "Ez" ){
                all_field_functors[lev][comp] = new CellCenterFunctor(warpx.get_pointer_Efield_aux(lev, 2), lev);
            } else if ( varnames[comp] == "Bx" ){
                all_field_functors[lev][comp] = new CellCenterFunctor(warpx.get_pointer_Bfield_aux(lev, 0), lev);
            } else if ( varnames[comp] == "By" ){
                all_field_functors[lev][comp] = new CellCenterFunctor(warpx.get_pointer_Bfield_aux(lev, 1), lev);
            } else if ( varnames[comp] == "Bz" ){
                all_field_functors[lev][comp] = new CellCenterFunctor(warpx.get_pointer_Bfield_aux(lev, 2), lev);
            } else if ( varnames[comp] == "jx" ){
                all_field_functors[lev][comp] = new CellCenterFunctor(warpx.get_pointer_current_fp(lev, 0), lev);
            } else if ( varnames[comp] == "jy" ){
                all_field_functors[lev][comp] = new CellCenterFunctor(warpx.get_pointer_current_fp(lev, 1), lev);
            } else if ( varnames[comp] == "jz" ){
                all_field_functors[lev][comp] = new CellCenterFunctor(warpx.get_pointer_current_fp(lev, 2), lev);
            } else if ( varnames[comp] == "rho" ){
                all_field_functors[lev][comp] = new CellCenterFunctor(warpx.get_pointer_rho_fp(lev), lev);
            } else if ( varnames[comp] == "part_per_cell" ){
                amrex::Abort("plotting part_per_cell is not supported");
            }
        }

        AddRZModesToOutputNames();

        AddRZModesToInputsFunctors( lev );

        // At this point, varnames.size() >= all_field_functors[0].size()

        // Allocate output multifab
        // Note: default MultiFab constructor is cell-centered
        mf_avg[lev] = MultiFab(warpx.boxArray(lev),
                               warpx.DistributionMap(lev),
                               varnames.size(), 0);
    }
    // Construct Flush class. So far, only Plotfile is implemented.
    m_flush_format = new FlushFormatPlotfile;
}

void
Diagnostics::ComputeAndPack ()
{
    // First, make sure all guard cells are properly filled
    // Probably overkill/unnecessary, but safe and shouldn't happen often !!
    auto & warpx = WarpX::GetInstance();
    warpx.FillBoundaryE(warpx.getngE(), warpx.getngExtra());
    warpx.FillBoundaryB(warpx.getngE(), warpx.getngExtra());
#ifndef WARPX_USE_PSATD
    warpx.FillBoundaryAux(warpx.getngUpdateAux());
#endif
    warpx.UpdateAuxilaryData();

    warpx.FieldGather();

    // cell-center fields and store result in mf_avg.
    int icomp_dst = 0;
    for(int lev=0; lev<nlev; lev++){
        for (int icomp=0, n=all_field_functors[0].size(); icomp<n; icomp++){
            // Call all functors in all_field_functors[lev]. Each of them computes
            // a diagnostics and writes in one or more components of the output
            // multifab mf_avg[lev].
            all_field_functors[lev][icomp]->operator()(mf_avg[lev], icomp_dst);
            // update the index of the next component to fill
            icomp_dst += all_field_functors[lev][icomp]->nComp();
        }
    }
    // Check that the proper number of components of mf_avg were updated.
    AMREX_ALWAYS_ASSERT( icomp_dst == varnames.size() );
}

void
Diagnostics::Flush ()
{
    auto & warpx = WarpX::GetInstance();
    m_flush_format->WriteToFile(
        varnames, GetVecOfConstPtrs(mf_avg), warpx.Geom(), warpx.getistep(),
        warpx.gett_new(0), warpx.GetPartContainer(), nlev, file_prefix,
        m_plot_raw_fields, m_plot_raw_fields_guards, m_plot_rho, m_plot_F);
}

void
Diagnostics::FlushRaw () {}

bool
Diagnostics::DoDump (int step, bool force_flush)
{
    if (force_flush) return true;
    if ( m_period>0 && (step+1)%m_period==0 ) return true;
    return false;
}

void
Diagnostics::AddRZModesToInputsFunctors (int lev)
{
#ifdef WARPX_DIM_RZ
    auto & warpx = WarpX::GetInstance();
    int ncomp_multimodefab = warpx.get_pointer_Efield_aux(0, 0)->nComp();
    // Make sure all multifabs have the same number of components
    for (int dim=0; dim<3; dim++){
        AMREX_ALWAYS_ASSERT(
            warpx.get_pointer_Efield_aux(lev, dim)->nComp() == ncomp_multimodefab );
        AMREX_ALWAYS_ASSERT(
            warpx.get_pointer_Bfield_aux(lev, dim)->nComp() == ncomp_multimodefab );
        AMREX_ALWAYS_ASSERT(
            warpx.get_pointer_current_fp(lev, dim)->nComp() == ncomp_multimodefab );
    }

    // First index of all_field_functors[lev] where RZ modes are stored
    int icomp = all_field_functors[0].size();

    // Er, Etheta, Ez, Br, Btheta, Bz, jr, jtheta, jz
    // Each of them being a multi-component multifab
    all_field_functors[lev].resize( all_field_functors[0].size() + 9 );
    // E
    for (int dim=0; dim<3; dim++){
        // 3 components, r theta z
        all_field_functors[lev][icomp] = new
            CellCenterFunctor(warpx.get_pointer_Efield_aux(lev, dim), lev,
                              false, ncomp_multimodefab);
        icomp += 1;
    }
    // B
    for (int dim=0; dim<3; dim++){
        // 3 components, r theta z
        all_field_functors[lev][icomp] = new
            CellCenterFunctor(warpx.get_pointer_Bfield_aux(lev, dim), lev,
                              false, ncomp_multimodefab);
        icomp += 1;
    }
    // j
    for (int dim=0; dim<3; dim++){
        // 3 components, r theta z
        all_field_functors[lev][icomp] = new
            CellCenterFunctor(warpx.get_pointer_current_fp(lev, dim), lev,
                              false, ncomp_multimodefab);
        icomp += 1;
    }
    // Sum the number of components in input vector all_field_functors
    // and check that it corresponds to the number of components in varnames
    // and mf_avg
    int ncomp_from_src = 0;
    for (int i=0; i<all_field_functors[0].size(); i++){
        ncomp_from_src += all_field_functors[lev][i]->nComp();
    }
    AMREX_ALWAYS_ASSERT( ncomp_from_src == varnames.size() );
#endif
}

void
Diagnostics::AddRZModesToOutputNames (){
#ifdef WARPX_DIM_RZ
    auto & warpx = WarpX::GetInstance();
    // In cylindrical geometry, real and imag part of each mode are also
    // dumped to file separately, so they need to be added to varnames
    for (std::string field : {"E", "B", "j"}){
        for (std::string dir :  {"r", "theta", "z"}){
            varnames.push_back( field + dir + "_0_real" );
            for (int ic=1 ; ic < warpx.get_pointer_Efield_aux(0, 0)->nComp() ; ic += 2) {
                varnames.push_back( field + dir + "_" + std::to_string(ic) + "_real" );
                varnames.push_back( field + dir + "_" + std::to_string(ic) + "_imag" );
            }
        }
    }
#endif
}
