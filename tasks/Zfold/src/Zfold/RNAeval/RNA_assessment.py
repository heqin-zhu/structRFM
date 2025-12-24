#  example.py
#
#  Copyright 2017 Chichau Miau <zmiao@ebi.ac.uk>
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#
#

## install RNA_assessment according to https://github.com/RNA-Puzzles/RNA_assessment

import sys,os

import RNA_normalizer

RESIDUES_LIST = "RNA_assessment/data/residues.list"
ATOMS_LIST = "RNA_assessment/data/atoms.list"

def CleanFormat(f):
    os.system( "mac2unix -q %s" %f )
    os.system( "dos2unix -q %s" %f )

def normalize_structure(struct, out_file = None, index_file=None, extract_file = None):
    pdb_normalizer = RNA_normalizer.PDBNormalizer( RESIDUES_LIST, ATOMS_LIST )
    ok = pdb_normalizer.parse( struct, out_file )
    if not ok:
        sys.stderr.write("ERROR: structure not normalized!\n")
    else:
        sys.stderr.write("INFO: Normalization succeded!\n")
    if not extract_file is None:
        coords=open(index_file).read()
        extract.extract_PDB(SOLUTION_NORMAL,coords, extract_file)
        sys.stderr.write("INFO: structure extracted\n")

# PVALUE set according to Hajdin et al., RNA (7) 16, 2010, either "+" or "-"
def calc_RMSD(native_file, prediction_file, native_index=None, prediction_index=None, PVALUE = "-"):
    res_struct = RNA_normalizer.PDBStruct()
    res_struct.load( native_file, native_index )
    res_raw_seq = res_struct.raw_sequence()

    sol_struct = RNA_normalizer.PDBStruct()
    sol_struct.load( prediction_file, prediction_index )
    sol_raw_seq = sol_struct.raw_sequence()

    if( sol_raw_seq != res_raw_seq ):
        sys.stderr.write("ERROR Result sequence != Solution sequence!\n")
        sys.stderr.write("DATA Solution sequence --> '%s'\n" %sol_raw_seq )
        sys.stderr.write("DATA Result sequence   --> '%s'\n" %res_raw_seq )
        return(-1)
    # computes the RMSD
    comparer = RNA_normalizer.PDBComparer()
    rmsd = comparer.rmsd( sol_struct, res_struct )
    sys.stderr.write("INFO Partial RMSD --> %f\n" %rmsd )
    pvalue = comparer.pvalue( rmsd, len(sol_raw_seq), PVALUE )
    sys.stderr.write("INFO Partial P-Value --> %e\n" %pvalue )
    return {'RMSD': rmsd, 'RMSD_pvalue': pvalue}

def InteractionNetworkFidelity(native_file, prediction_file, native_index=None, prediction_index=None):
    res_struct = RNA_normalizer.PDBStruct()
    res_struct.load( native_file, native_index )
    res_raw_seq = res_struct.raw_sequence()

    sol_struct = RNA_normalizer.PDBStruct()
    sol_struct.load( prediction_file, prediction_index )
    sol_raw_seq = sol_struct.raw_sequence()

    if( sol_raw_seq != res_raw_seq ):
        sys.stderr.write("ERROR Result sequence != Solution sequence!\n")
        sys.stderr.write("DATA Solution sequence --> '%s'\n" %sol_raw_seq )
        sys.stderr.write("DATA Result sequence   --> '%s'\n" %res_raw_seq )
        return(-1)
    # computes the RMSD
    comparer = RNA_normalizer.PDBComparer()
    rmsd = comparer.rmsd( sol_struct, res_struct )
    INF_ALL = comparer.INF( sol_struct, res_struct, type="ALL" )
    DI_ALL = rmsd / INF_ALL
    INF_WC = comparer.INF( sol_struct, res_struct, type="PAIR_2D" )
    INF_NWC = comparer.INF( sol_struct, res_struct, type="PAIR_3D" )
    INF_STACK = comparer.INF( sol_struct, res_struct, type="STACK" )
    return {
            'RMSD': rmsd,
            'DI_ALL': DI_ALL, 
            'INF_ALL': INF_ALL, 
            'INF_WC': INF_WC, 
            'INF_NWC': INF_NWC,
            'INF_STACK': INF_STACK
           }


if __name__ == '__main__':
    # Normalize PDB format, correct residue names and atom names.
    normalize_structure('example/14_solution_0.pdb','example/14_solution_normalized.pdb')

    # calculate RMSD for RNA structures
    # require biopython
    out1 = calc_RMSD("example/14_solution_0.pdb","example/14_ChenPostExp_2.pdb","example/14_solution_0.index","example/14_ChenPostExp_2.index")

    # calculate InteractionNetworkFidelity and Deformation Index for RNA structures
    # need to have MA-annotate in the directory or set in mcannotate.py
    out2 = InteractionNetworkFidelity("example/14_solution_0.pdb","example/14_ChenPostExp_2.pdb","example/14_solution_0.index","example/14_ChenPostExp_2.index")
    print(out1)
    print(out2)
