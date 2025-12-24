#  mcannotate.py
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
#  This script calls MC-Annotate to calculate RNA 3D interactions from RNA structure. 
#  With the results of MC-Annotate, Interaction network fidelity can be measured. 
import re
import os

# !!! IMPORTANT, please set the directory of MC-Annotate before using this script.
file_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MCAnnotate_bin=f'{file_dir}/MC-Annotate'

class MCAnnotate:
    def __init__(self):
        self.mc_file = ""
        self.residues = []
        self.interactions = []
    
    def load(self, pdb_file, mc_dir):
        # defines the annotation file
        self.mc_file = "%s/%s.mcout" %(mc_dir, os.path.basename(pdb_file))
        
        # check if the annotation file exists
        if not os.path.isfile( self.mc_file ):
            # create a new annotation file
            cmd = "%s %s > %s" %(MCAnnotate_bin, pdb_file, self.mc_file)
            os.system( cmd )
        
        # parse the annotation file
        self.parse()
    
    def parse(self):
        STATE_OUT = 0
        STATE_RESIDUE = 1
        STATE_PAIR = 2
        STATE_STACK = 3
        
        pattern_pair = "^([A-Z]|\'[0-9]\'|)(\d+)-([A-Z]|\'[0-9]\'|)(\d+) : (\w+)-(\w+) ([\w\']+)/([\w\']+)(?:.*)pairing( (parallel|antiparallel) (cis|trans))"
        pattern_stack = "^([A-Z]|\'[0-9]\'|)(\d+)-([A-Z]|\'[0-9]\'|)(\d+) :.*(inward|upward|downward|outward).*"
              
        # opens and parses the annotation file
        f = open( self.mc_file, "r" )
            
        model_count = 0
        state = STATE_OUT 
        for line in f:
            line = line.strip()
    
            # for now we only care about the first model
            if( line.startswith( "Residue conformations" ) ):
                if( model_count == 0 ):
                    state = STATE_RESIDUE
                    model_count += 1
                    continue
                else:
                    break
            
            if( line.startswith( "Base-pairs" ) ):
                state = STATE_PAIR
                continue
    
            if( line.startswith( "Adjacent stackings" ) or line.startswith( "Non-Adjacent stackings" ) ):
                state = STATE_STACK
                continue
    
            if( line.endswith( "----------" ) ):
                state = STATE_OUT
                continue
            
            interaction = None
            
            if( state == STATE_RESIDUE ):
                data = line.split()
                
                if( len(data) == 5 ):
                    self.residues.append( (data[0][0], data[0][1:], data[2]) )
            
            if( state == STATE_PAIR ):
                match = re.match( pattern_pair, line )
                
                if( match != None ):
                    g = match.groups()
                    interaction = self.convert_pair( g )
            
            if( state == STATE_STACK ):
                match = re.match( pattern_stack, line )
                
                if( match != None ):
                    g = match.groups()
                    interaction = self.convert_stack( g )
    
            if( interaction != None ):
                self.interactions.append( interaction )
            
        f.close()
    
    def convert_pair( self, match ):
        int_a = match[6][0].upper()
        int_b = match[7][0].upper()
        
        result = None
        
        if( (int_a in ["W", "H", "S"]) and (int_b in ["W", "H", "S"]) ):
            chain_a = match[0].replace( "'", "" )
            pos_a = int(match[1])
            nt_a = match[4]
            
            chain_b = match[2].replace( "'", "" )
            pos_b = int(match[3])
            nt_b = match[5]
            
            int_type = "%s%s" %(int_a, int_b)
            int_orientation = match[10].lower()
            
            # define the type of pair
            pair_name_aux = "%s%s%s" %(int_orientation, int_a, int_b)
            
            if( pair_name_aux == "cisWW" ):
                pair_name = "PAIR_2D"
            else:
                pair_name = "PAIR_3D"
            
            
            # check if the smallest 'pos' is always the first 
            if( ((chain_a == chain_b) and (pos_a < pos_b)) or (chain_a < chain_b) ):
                result = (pair_name, chain_a, pos_a, nt_a, chain_b, pos_b, nt_b, int_type, int_orientation, "")
            else:
                result = (pair_name, chain_b, pos_b, nt_b, chain_a, pos_a, nt_a, int_type, int_orientation, "")
        
        return( result )
    
    def convert_stack( self, match ):
        chain_a = match[0].replace( "'", "" )
        pos_a = int(match[1])

        chain_b = match[2].replace( "'", "" )
        pos_b = int(match[3])
        
        int_type = match[4]
            
        return( "STACK", chain_a, pos_a, "", chain_b, pos_b, "", int_type, "", "" )
    

if __name__ == '__main__':
    pdb_dir = '/public/share/heqinzhu_share/structRFM/RNA3d_eval/pred/20_RNA_Puzzles/'

    import shutil
    tmp_dir = 'tmp'
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    shutil.copytree(pdb_dir, tmp_dir)
    pdb_dir = tmp_dir
    mca = MCAnnotate()
    for f in os.listdir(pdb_dir):
        f = 'PZ1.pdb'
        pdb_path = os.path.join(pdb_dir, f)
        mca.load(pdb_path, pdb_dir)
        print(pdb_path)
        for entry in mca.interactions:
            print(entry) 
        break
