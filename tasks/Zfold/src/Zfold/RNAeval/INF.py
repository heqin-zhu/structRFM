import os

from mcannotate import MCAnnotate

class MCStruct:
    def __init__(self, pdb_path, cache_dir='.MCAnno'):
        self._pdb_file = pdb_path
        os.makedirs(cache_dir, exist_ok=True)
        self._load_annotations_3D()


    def _load_annotations_3D(self):
        self._interactions = []
        mca = MCAnnotate()
        mca.load( self._pdb_file, os.path.dirname( self._pdb_file ) )
        #~ print mca.interactions
        for (type, chain_a, pos_a, nt_a, chain_b, pos_b, nt_b, extra1, extra2, extra3) in mca.interactions:
            # get the rank of the first position of the pair
            rank_a = self._get_index( chain_a, pos_a, 1 )
            rank_b = self._get_index( chain_b, pos_b, 1 )

            if( (rank_a is None) or (rank_b is None) ):
                continue
            #~ return False

            if( type == "STACK" ):
                extra = extra1
            else:
                extra = "%s%s" %(extra1, extra2)
            self._interactions.append( (type, min( rank_a, rank_b ), max( rank_a, rank_b ), extra ))


    def get_interactions(self, anno_type="ALL"):
        # ('PAIR_2D', 'A', 23, 'G', 'A', 27, 'C', 'WW', 'cis', '')
        if( type == "ALL" ):
            # "ALL": returns all interactions
            return self._interactions
        elif( type in ("PAIR") ):
            # "PAIR": returns all pairs irrespective of their type
            return list(filter( lambda x: x[0] in ("PAIR_2D", "PAIR_3D"), self._interactions ))
        elif( type in ("PAIR_2D", "PAIR_3D", "STACK") ):
            # "PAIR_2D", "PAIR_3D", "STAK": returns the interactions of the specified type
            return list(filter( lambda x: x[0] == type, self._interactions ))
        else:
            raise Exception(f'Unknown anno_type: {anno_type}')


def INF(pred, gt, tp):
    (P, TP, FP, FN) = (0, 0, 0, 0)

    for (stype, sb1, sb2, sextra) in src_struct.get_interactions( tp ):
        P += 1
        found = False
        for (ttype, tb1, tb2, textra) in trg_struct.get_interactions( type ):
            if( (stype==ttype) and (sb1==tb1) and (sb2==tb2) and (sextra==textra) ):
                found = True
                break

        if( found ):
            #print "TP>", (stype, sb1, sb2, sextra)
            TP += 1
        else:
            #print "FN>", (stype, sb1, sb2, sextra)
            FN += 1

    for (ttype, tb1, tb2, textra) in trg_struct.get_interactions( tp ):
        found = False
        for (stype, sb1, sb2, sextra) in src_struct.get_interactions( type ):
            if( (stype==ttype) and (sb1==tb1) and (sb2==tb2) and (sextra==textra) ):
                found = True
                break

        if( not found ):
            FP += 1
            #print "FP>", (ttype, tb1, tb2, textra)

    if( TP == 0 and (FP == 0 or FN == 0) ):
        INF = -1.0
    else:
        PPV = float(TP) / (float(TP) + float(FP))
        STY = float(TP) / (float(TP) + float(FN))
        INF = (PPV * STY) ** 0.5

    #print "##>", INF, P, TP, FP, FN

    return( INF )


if __name__ == '__main__':
