import os

from mcannotate import MCStruct


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
