/*
FILE:     Ndb2Pdb_Remark_500s_Util.C
*/
/*
VERSION:  10.200
*/
/*
DATE:     7/29/2020
*/
/*
  Comments and Questions to: sw-help@rcsb.rutgers.edu
*/
/*
COPYRIGHT 1999-2020 Rutgers - The State University of New Jersey

This software is provided WITHOUT WARRANTY OF MERCHANTABILITY OR
FITNESS FOR A PARTICULAR PURPOSE OR ANY OTHER WARRANTY, EXPRESS OR
IMPLIED.  RUTGERS MAKE NO REPRESENTATION OR WARRANTY THAT THE
SOFTWARE WILL NOT INFRINGE ANY PATENT, COPYRIGHT OR OTHER
PROPRIETARY RIGHT.

The user of this software shall indemnify, hold harmless and defend
Rutgers, its governors, trustees, officers, employees, students,
agents and the authors against any and all claims, suits,
losses, liabilities, damages, costs, fees, and expenses including
reasonable attorneys' fees resulting from or arising out of the
use of this software.  This indemnification shall include, but is
not limited to, any and all claims alleging products liability.
*/
/*
               RCSB PDB SOFTWARE LICENSE AGREEMENT

BY CLICKING THE ACCEPTANCE BUTTON OR INSTALLING OR USING 
THIS "SOFTWARE, THE INDIVIDUAL OR ENTITY LICENSING THE  
SOFTWARE ("LICENSEE") IS CONSENTING TO BE BOUND BY AND IS 
BECOMING A PARTY TO THIS AGREEMENT.  IF LICENSEE DOES NOT 
AGREE TO ALL OF THE TERMS OF THIS AGREEMENT
THE LICENSEE MUST NOT INSTALL OR USE THE SOFTWARE.

1. LICENSE AGREEMENT

This is a license between you ("Licensee") and the Protein Data Bank (PDB) 
at Rutgers, The State University of New Jersey (hereafter referred to 
as "RUTGERS").   The software is owned by RUTGERS and protected by 
copyright laws, and some elements are protected by laws governing 
trademarks, trade dress and trade secrets, and may be protected by 
patent laws. 

2. LICENSE GRANT

RUTGERS grants you, and you hereby accept, non-exclusive, royalty-free 
perpetual license to install, use, modify, prepare derivative works, 
incorporate into other computer software, and distribute in binary 
and source code format, or any derivative work thereof, together with 
any associated media, printed materials, and on-line or electronic 
documentation (if any) provided by RUTGERS (collectively, the "SOFTWARE"), 
subject to the following terms and conditions: (i) any distribution 
of the SOFTWARE shall bind the receiver to the terms and conditions 
of this Agreement; (ii) any distribution of the SOFTWARE in modified 
form shall clearly state that the SOFTWARE has been modified from 
the version originally obtained from RUTGERS.  

2. COPYRIGHT; RETENTION OF RIGHTS.  

The above license grant is conditioned on the following: (i) you must 
reproduce all copyright notices and other proprietary notices on any 
copies of the SOFTWARE and you must not remove such notices; (ii) in 
the event you compile the SOFTWARE, you will include the copyright 
notice with the binary in such a manner as to allow it to be easily 
viewable; (iii) if you incorporate the SOFTWARE into other code, you 
must provide notice that the code contains the SOFTWARE and include 
a copy of the copyright notices and other proprietary notices.  All 
copies of the SOFTWARE shall be subject to the terms of this Agreement.  

3. NO MAINTENANCE OR SUPPORT; TREATMENT OF ENHANCEMENTS 

RUTGERS is under no obligation whatsoever to: (i) provide maintenance 
or support for the SOFTWARE; or (ii) to notify you of bug fixes, patches, 
or upgrades to the features, functionality or performance of the 
SOFTWARE ("Enhancements") (if any), whether developed by RUTGERS 
or third parties.  If, in its sole discretion, RUTGERS makes an 
Enhancement available to you and RUTGERS does not separately enter 
into a written license agreement with you relating to such bug fix, 
patch or upgrade, then it shall be deemed incorporated into the SOFTWARE 
and subject to this Agreement. You are under no obligation whatsoever 
to provide any Enhancements to RUTGERS or the public that you may 
develop over time; however, if you choose to provide your Enhancements 
to RUTGERS, or if you choose to otherwise publish or distribute your 
Enhancements, in source code form without contemporaneously requiring 
end users or RUTGERS to enter into a separate written license agreement 
for such Enhancements, then you hereby grant RUTGERS a non-exclusive,
royalty-free perpetual license to install, use, modify, prepare
derivative works, incorporate into the SOFTWARE or other computer
software, distribute, and sublicense your Enhancements or derivative
works thereof, in binary and source code form.

4. FEES.  There is no license fee for the SOFTWARE.  If Licensee
wishes to receive the SOFTWARE on media, there may be a small charge
for the media and for shipping and handling.  Licensee is
responsible for any and all taxes.

5. TERMINATION.  Without prejudice to any other rights, Licensor
may terminate this Agreement if Licensee breaches any of its terms
and conditions.  Upon termination, Licensee shall destroy all
copies of the SOFTWARE.

6. PROPRIETARY RIGHTS.  Title, ownership rights, and intellectual
property rights in the Product shall remain with RUTGERS.  Licensee 
acknowledges such ownership and intellectual property rights and will 
not take any action to jeopardize, limit or interfere in any manner 
with RUTGERS' ownership of or rights with respect to the SOFTWARE.  
The SOFTWARE is protected by copyright and other intellectual 
property laws and by international treaties.  Title and related 
rights in the content accessed through the SOFTWARE is the property 
of the applicable content owner and is protected by applicable law.  
The license granted under this Agreement gives Licensee no rights to such
content.

7. DISCLAIMER OF WARRANTY.  THE SOFTWARE IS PROVIDED FREE OF 
CHARGE, AND, THEREFORE, ON AN "AS IS" BASIS, WITHOUT WARRANTY OF 
ANY KIND, INCLUDING WITHOUT LIMITATION THE WARRANTIES THAT IT 
IS FREE OF DEFECTS, MERCHANTABLE, FIT FOR A PARTICULAR PURPOSE 
OR NON-INFRINGING.  THE ENTIRE RISK AS TO THE QUALITY AND 
PERFORMANCE OF THE SOFTWARE IS BORNE BY LICENSEE.  SHOULD THE 
SOFTWARE PROVE DEFECTIVE IN ANY RESPECT, THE LICENSEE AND NOT 
LICENSOR ASSUMES THE ENTIRE COST OF ANY SERVICE AND REPAIR.  
THIS DISCLAIMER OF WARRANTY CONSTITUTES AN ESSENTIAL PART OF 
THIS AGREEMENT.  NO USE OF THE PRODUCT IS AUTHORIZED HEREUNDER 
EXCEPT UNDER THIS DISCLAIMER.

8. LIMITATION OF LIABILITY.  TO THE MAXIMUM EXTENT PERMITTED BY
APPLICABLE LAW,  IN NO EVENT WILL LICENSOR BE LIABLE FOR ANY 
INDIRECT, SPECIAL, INCIDENTAL OR CONSEQUENTIAL DAMAGES ARISING 
OUT OF THE USE OF OR INABILITY TO USE THE SOFTWARE, INCLUDING, 
WITHOUT LIMITATION, DAMAGES FOR LOSS OF GOODWILL, WORK 
STOPPAGE, COMPUTER FAILURE OR MALFUNCTION, OR ANY AND ALL 
OTHER COMMERCIAL DAMAGES OR LOSSES, EVEN IF ADVISED OF THE
POSSIBILITY THEREOF. 
*/


// ----------------------------------------------------------------------------

// #include <stdio.h>
// #include <stdlib.h>
// #include <string.h>

// #include "Maxit.h"
// #include "PdbWrite.h"
// #include "utillib.h"


// void Maxit::_remark_500_close_contact(const std::vector<Atom*>& atoms, const double& val,
//                                       const std::string& symmetry)
// {
//        std::string remark = " ";
//        for (int i = 0; i < 2; ++i) {
//             std::string atomtyp = printAtomNameField(_ccDic, atoms[i]->atom_type(),
//                                   atoms[i]->pdb_atmnam(), atoms[0]->pdb_resnam());
//             if (i) remark += "   ";
//             remark += FormattedFieldValue(atomtyp, 3, 4, 0, true, true) + "  "
//               + FormattedFieldValue(atoms[i]->pdb_resnam(), 3, 3, 0, false, true)
//               + FormattedFieldValue(atoms[i]->pdb_chnid(), 3, 2, 0, false, true) + "  "
//               + FormattedFieldValue(atoms[i]->pdb_resnum(), 3, 4, 0, false, true)
//               + atoms[i]->ins_code_char();
//        }

//        if (!symmetry.empty())
//             remark += "   " + FormattedFieldValue(symmetry, 3, 5, 0, false, true)
//                     + "    ";
//        else remark += "            ";
//        remark += FloatToString(val, 5, 2, false, true);
//        _addNewRemark(500, remark);
// }

// void Maxit::_remark_500_bond_deviation(const std::vector<Atom*>& atoms, const double& val,
//                                        const int& mol_id)
// {
//        std::string remark = "   ";
//        if (mol_id) remark = FloatToString((double) mol_id, 2, 0, false, true) + " ";

//        for (int i = 0; i < 2; ++i) {
//             std::string atomtyp = printAtomNameField(_ccDic, atoms[i]->atom_type(),
//                                   atoms[i]->pdb_atmnam(), atoms[0]->pdb_resnam());
//             std::string ins_code = atoms[i]->ins_code_char();
//             if (i) remark += "   ";
//             remark += FormattedFieldValue(atoms[i]->pdb_resnam(), 3, 3, 0, false, true)
//                     + FormattedFieldValue(atoms[i]->pdb_chnid(), 3, 2, 0, false, true)
//                     + FormattedFieldValue(atoms[i]->pdb_resnum(), 3, 4, 0, false, true)
//                     + atoms[i]->ins_code_char() + " "
//                     + FormattedFieldValue(atomtyp, 3, 4, 0, false, true);
//        }
//        remark += "    " + FloatToString(val, 6, 3, false, true);
//        _addNewRemark(500, remark);
// }

// void Maxit::_remark_500_angle_deviation(const std::vector<Atom*>& atoms, const double& val,
//                                         const int& mol_id)
// {
//        std::string remark = "   ";
//        if (mol_id) remark = FloatToString((double) mol_id, 2, 0, false, true) + " ";

//        remark += FormattedFieldValue(atoms[1]->pdb_resnam(), 3, 3, 0, false, true)
//                + FormattedFieldValue(atoms[1]->pdb_chnid(), 3, 2, 0, false, true)
//                + FormattedFieldValue(atoms[1]->pdb_resnum(), 3, 4, 0, false, true)
//                + atoms[1]->ins_code_char();

//        for (int i = 0; i < 3; ++i) {
//             std::string atomtyp = printAtomNameField(_ccDic, atoms[i]->atom_type(),
//                                   atoms[i]->pdb_atmnam(), atoms[i]->pdb_resnam());
//             if (i) remark += "-";
//             remark += " " + FormattedFieldValue(atomtyp, 3, 4, 0, false, true) + " ";
//        }
//        remark += "ANGL. DEV. = " +  FloatToString(val, 5, 1, false, true) + " DEGREES";
//        _addNewRemark(500, remark);
// }

// void Maxit::_remark_500_Ramachandran_outliers(Atom* atom, const double& psi, const
//                                               double& phi, const int& mol_id)
// {
//        std::string remark = "   ";
//        if (mol_id) remark = FloatToString((double) mol_id, 2, 0, false, true) + " ";

//        std::string ins_code = atom->ins_code_char();
//        remark += FormattedFieldValue(atom->pdb_resnam(), 3, 3, 0, false, true)
//               + FormattedFieldValue(atom->pdb_chnid(), 3, 2, 0, false, true)
//               + FormattedFieldValue(atom->pdb_resnum(), 3, 4, 0, false, true)
//               + atom->ins_code_char() + "    "
//               + FloatToString(psi, 7, 2, false, true) + "   "
//               + FloatToString(phi, 7, 2, false, true);
//        _addNewRemark(500, remark);
// }

// void Maxit::_remark_500_non_cis_trans_torsions(const std::vector<Atom*>& atoms, const
//                                                double& val, const int& mol_id)
// {
//        std::string remark = "";
//        for (int i = 0; i < 2; ++i) {
//             if (i) remark += "    ";
//             remark += FormattedFieldValue(atoms[i]->pdb_resnam(), 3, 3, 0, false, true)
//                    + FormattedFieldValue(atoms[i]->pdb_chnid(), 3, 2, 0, false, true) + " "
//                    + FormattedFieldValue(atoms[i]->pdb_resnum(), 3, 4, 0, false, true)
//                    + atoms[i]->ins_code_char();
//        }
//        if (mol_id)
//             remark += "       " + FloatToString((double) mol_id, 3, 0, false, true)
//                     + "      ";
//        else remark += "                ";
//        remark += FloatToString(val, 7, 2, false, true);
//        _addNewRemark(500, remark);
// }

// void Maxit::_remark_500_side_chain_plane(Atom* atom, const double& val, const int&
//                                          mol_id, const std::string& details)
// {
//        std::string remark = "   ";
//        if (mol_id) remark = FloatToString((double) mol_id, 2, 0, false, true) + " ";

//        remark += FormattedFieldValue(atom->pdb_resnam(), 3, 3, 0, false, true)
//               + FormattedFieldValue(atom->pdb_chnid(), 3, 2, 0, false, true)
//               + FormattedFieldValue(atom->pdb_resnum(), 3, 4, 0, false, true)
//               + atom->ins_code_char() + "        "
//               + FloatToString(val, 4, 2, false, true) + "    " + details;
//        _addNewRemark(500, remark);
// }

// void Maxit::_remark_500_main_chain_planarity(Atom* atom, const double& val,
//                                              const int& mol_id)
// {
//        std::string remark = "   ";
//        if (mol_id) remark = FloatToString((double) mol_id, 2, 0, false, true) + " ";

//        remark += FormattedFieldValue(atom->pdb_resnam(), 3, 3, 0, false, true)
//               + FormattedFieldValue(atom->pdb_chnid(), 3, 2, 0, false, true)
//               + FormattedFieldValue(atom->pdb_resnum(), 3, 4, 0, false, true)
//               + atom->ins_code_char() + "      "
//               + FloatToString(val, 7, 2, false, true);
//        _addNewRemark(500, remark);
// }

// void Maxit::_remark_525(Atom* atom, const double& val)
// {
//        std::string remark = "   "
//            + FormattedFieldValue(atom->pdb_resnam(), 3, 3, 0, false, true) + " "
//            + atom->pdb_chnid_char()
//            + FormattedFieldValue(atom->pdb_resnum(), 3, 4, 0, false, true)
//            + atom->ins_code_char() + "       DISTANCE = "
//            + FloatToString(val, 5, 2, false, true) + " ANGSTROMS";
//        _addNewRemark(525, remark);
// }




// ----------------------------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>

#include "Maxit.h"
#include "PdbWrite.h"
#include "utillib.h"

// ============================================================
// Stereochem instrumentation
// - Prints per-hit lines to stderr (so you can redirect/grep).
// - Also prints a one-line summary at program exit (even if all 0).
//
// Usage:
//   maxit ... 2> model.maxit.txt
//   grep -Ei 'close|bond length|bond angle|planar|chirality|polymer' model.maxit.txt
// ============================================================

static long g_sc_close_contacts = 0;
static long g_sc_bond_length    = 0;
static long g_sc_bond_angle     = 0;
static long g_sc_planar         = 0;
static long g_sc_chirality      = 0; // this file doesn't generate chiral rows; stays 0 unless you add hooks elsewhere
static long g_sc_polymer        = 0; // same

static void sc_dump_summary(void) {
    // Always print: makes batch runs easy to parse even for "all good" structures
    fprintf(stderr,
        "[STEREOCHEM SUMMARY] close=%ld bond length=%ld bond angle=%ld planar=%ld chirality=%ld polymer=%ld\n",
        g_sc_close_contacts,
        g_sc_bond_length,
        g_sc_bond_angle,
        g_sc_planar,
        g_sc_chirality,
        g_sc_polymer
    );
}

// Register atexit hook automatically (GCC/Clang)
__attribute__((constructor))
static void sc_register_atexit(void) {
    atexit(sc_dump_summary);
}


void Maxit::_remark_500_close_contact(const std::vector<Atom*>& atoms, const double& val,
                                      const std::string& symmetry)
{
       std::string remark = " ";
       for (int i = 0; i < 2; ++i) {
            std::string atomtyp = printAtomNameField(_ccDic, atoms[i]->atom_type(),
                                  atoms[i]->pdb_atmnam(), atoms[0]->pdb_resnam());
            if (i) remark += "   ";
            remark += FormattedFieldValue(atomtyp, 3, 4, 0, true, true) + "  "
              + FormattedFieldValue(atoms[i]->pdb_resnam(), 3, 3, 0, false, true)
              + FormattedFieldValue(atoms[i]->pdb_chnid(), 3, 2, 0, false, true) + "  "
              + FormattedFieldValue(atoms[i]->pdb_resnum(), 3, 4, 0, false, true)
              + atoms[i]->ins_code_char();
       }

       if (!symmetry.empty())
            remark += "   " + FormattedFieldValue(symmetry, 3, 5, 0, false, true)
                    + "    ";
       else remark += "            ";
       remark += FloatToString(val, 5, 2, false, true);
       _addNewRemark(500, remark);

       // instrumentation
       g_sc_close_contacts++;
       fprintf(stderr, "[STEREOCHEM] close %s\n", remark.c_str());
}

void Maxit::_remark_500_bond_deviation(const std::vector<Atom*>& atoms, const double& val,
                                       const int& mol_id)
{
       std::string remark = "   ";
       if (mol_id) remark = FloatToString((double) mol_id, 2, 0, false, true) + " ";

       for (int i = 0; i < 2; ++i) {
            std::string atomtyp = printAtomNameField(_ccDic, atoms[i]->atom_type(),
                                  atoms[i]->pdb_atmnam(), atoms[0]->pdb_resnam());
            std::string ins_code = atoms[i]->ins_code_char();
            if (i) remark += "   ";
            remark += FormattedFieldValue(atoms[i]->pdb_resnam(), 3, 3, 0, false, true)
                    + FormattedFieldValue(atoms[i]->pdb_chnid(), 3, 2, 0, false, true)
                    + FormattedFieldValue(atoms[i]->pdb_resnum(), 3, 4, 0, false, true)
                    + atoms[i]->ins_code_char() + " "
                    + FormattedFieldValue(atomtyp, 3, 4, 0, false, true);
       }
       remark += "    " + FloatToString(val, 6, 3, false, true);
       _addNewRemark(500, remark);

       // instrumentation
       g_sc_bond_length++;
       fprintf(stderr, "[STEREOCHEM] bond length %s\n", remark.c_str());
}

void Maxit::_remark_500_angle_deviation(const std::vector<Atom*>& atoms, const double& val,
                                        const int& mol_id)
{
       std::string remark = "   ";
       if (mol_id) remark = FloatToString((double) mol_id, 2, 0, false, true) + " ";

       remark += FormattedFieldValue(atoms[1]->pdb_resnam(), 3, 3, 0, false, true)
               + FormattedFieldValue(atoms[1]->pdb_chnid(), 3, 2, 0, false, true)
               + FormattedFieldValue(atoms[1]->pdb_resnum(), 3, 4, 0, false, true)
               + atoms[1]->ins_code_char();

       for (int i = 0; i < 3; ++i) {
            std::string atomtyp = printAtomNameField(_ccDic, atoms[i]->atom_type(),
                                  atoms[i]->pdb_atmnam(), atoms[i]->pdb_resnam());
            if (i) remark += "-";
            remark += " " + FormattedFieldValue(atomtyp, 3, 4, 0, false, true) + " ";
       }
       remark += "ANGL. DEV. = " +  FloatToString(val, 5, 1, false, true) + " DEGREES";
       _addNewRemark(500, remark);

       // instrumentation
       g_sc_bond_angle++;
       fprintf(stderr, "[STEREOCHEM] bond angle %s\n", remark.c_str());
}

void Maxit::_remark_500_Ramachandran_outliers(Atom* atom, const double& psi, const
                                              double& phi, const int& mol_id)
{
       std::string remark = "   ";
       if (mol_id) remark = FloatToString((double) mol_id, 2, 0, false, true) + " ";

       std::string ins_code = atom->ins_code_char();
       remark += FormattedFieldValue(atom->pdb_resnam(), 3, 3, 0, false, true)
              + FormattedFieldValue(atom->pdb_chnid(), 3, 2, 0, false, true)
              + FormattedFieldValue(atom->pdb_resnum(), 3, 4, 0, false, true)
              + atom->ins_code_char() + "    "
              + FloatToString(psi, 7, 2, false, true) + "   "
              + FloatToString(phi, 7, 2, false, true);
       _addNewRemark(500, remark);
}

void Maxit::_remark_500_non_cis_trans_torsions(const std::vector<Atom*>& atoms, const
                                               double& val, const int& mol_id)
{
       std::string remark = "";
       for (int i = 0; i < 2; ++i) {
            if (i) remark += "    ";
            remark += FormattedFieldValue(atoms[i]->pdb_resnam(), 3, 3, 0, false, true)
                   + FormattedFieldValue(atoms[i]->pdb_chnid(), 3, 2, 0, false, true) + " "
                   + FormattedFieldValue(atoms[i]->pdb_resnum(), 3, 4, 0, false, true)
                   + atoms[i]->ins_code_char();
       }
       if (mol_id)
            remark += "       " + FloatToString((double) mol_id, 3, 0, false, true)
                    + "      ";
       else remark += "                ";
       remark += FloatToString(val, 7, 2, false, true);
       _addNewRemark(500, remark);
}

void Maxit::_remark_500_side_chain_plane(Atom* atom, const double& val, const int&
                                         mol_id, const std::string& details)
{
       std::string remark = "   ";
       if (mol_id) remark = FloatToString((double) mol_id, 2, 0, false, true) + " ";

       remark += FormattedFieldValue(atom->pdb_resnam(), 3, 3, 0, false, true)
              + FormattedFieldValue(atom->pdb_chnid(), 3, 2, 0, false, true)
              + FormattedFieldValue(atom->pdb_resnum(), 3, 4, 0, false, true)
              + atom->ins_code_char() + "        "
              + FloatToString(val, 4, 2, false, true) + "    " + details;
       _addNewRemark(500, remark);

       // instrumentation
       g_sc_planar++;
       fprintf(stderr, "[STEREOCHEM] planar %s\n", remark.c_str());
}

void Maxit::_remark_500_main_chain_planarity(Atom* atom, const double& val,
                                             const int& mol_id)
{
       std::string remark = "   ";
       if (mol_id) remark = FloatToString((double) mol_id, 2, 0, false, true) + " ";

       remark += FormattedFieldValue(atom->pdb_resnam(), 3, 3, 0, false, true)
              + FormattedFieldValue(atom->pdb_chnid(), 3, 2, 0, false, true)
              + FormattedFieldValue(atom->pdb_resnum(), 3, 4, 0, false, true)
              + atom->ins_code_char() + "      "
              + FloatToString(val, 7, 2, false, true);
       _addNewRemark(500, remark);

       // instrumentation
       g_sc_planar++;
       fprintf(stderr, "[STEREOCHEM] planar %s\n", remark.c_str());
}

void Maxit::_remark_525(Atom* atom, const double& val)
{
       std::string remark = "   "
           + FormattedFieldValue(atom->pdb_resnam(), 3, 3, 0, false, true) + " "
           + atom->pdb_chnid_char()
           + FormattedFieldValue(atom->pdb_resnum(), 3, 4, 0, false, true)
           + atom->ins_code_char() + "       DISTANCE = "
           + FloatToString(val, 5, 2, false, true) + " ANGSTROMS";
       _addNewRemark(525, remark);
}


