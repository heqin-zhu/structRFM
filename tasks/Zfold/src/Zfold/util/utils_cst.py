#!/usr/bin/env /usr/bin/python
import tempfile

import os
import warnings

import numpy as np
import sys
import argparse
import math
from math import exp, log, sqrt, pi, atan


def npz2cst(args):
    global seq

    npz = np.load(args.NPZ, allow_pickle=True)
    seq = read_fasta(args.FASTA).replace('T', 'U')

    distances = npz['distance'].item()
    cont = npz['contact']

    dataP = distances["P"]
    dataC1 = distances["C3'"]
    dataC2 = distances["C1'"]
    dataC3 = distances["C4"]
    dataN = distances["N1"]

    cutoff_dist = args.dcut
    cutoff_contact = 0.6

    cstcout_path = args.tmpdir + f'/cstfile_cont.txt'
    cstout_path = args.tmpdir + f'/cstfile_dist.txt'
    splinec_tmp_dir = args.tmpdir + f'/splines_cont/'
    splines_tmp_dir = args.tmpdir + f'/splines_dist/'

    os.makedirs(splinec_tmp_dir, exist_ok=True)
    os.makedirs(splines_tmp_dir, exist_ok=True)

    cstout = open(cstout_path, "w")
    cstcout = open(cstcout_path, "w")
    dist_cst_fun1(dataP, cutoff_dist, cstout, splines_tmp_dir, "P")
    dist_cst_fun1(dataC1, cutoff_dist, cstout, splines_tmp_dir, "C3'")
    dist_cst_fun1(dataC2, cutoff_dist, cstout, splines_tmp_dir, "C1'")
    dist_cst_fun1(dataC3, cutoff_dist, cstout, splines_tmp_dir, "C4")
    dist_cst_fun1(dataN, cutoff_dist, cstout, splines_tmp_dir, "N1")
    cont_cst_fun1(cont, cutoff_contact, cstcout, splinec_tmp_dir)
    cstout.close()
    cstcout.close()
    with open(f'{args.tmpdir}/done.txt', 'w') as f:
        f.write('yes')
    ####################################################################


#    fasta file
#################################

def read_fasta(file):
    fasta = ""
    with open(file, "r") as f:
        for line in f:
            if (line[0] == ">"):
                if len(fasta) > 0:
                    warnings.warn(
                        'Submitted protein contained multiple chains. Only the first protein chain will be used')
                    break
                continue
            else:
                line = line.rstrip()
                fasta = fasta + line
    return fasta


def convert_atm_name(*atms):
    map_dict = {'C': "C4'", 'P': 'P', 'N': 'N1'}
    converted_names = []
    res_ids = []
    N_id = []
    for i, atm in enumerate(atms):
        atm_name, atm_id = map_dict[atm[0]], atm[1:]
        converted_names.append(atm_name)
        res_ids.append(atm_id)
        if atm_name == 'N':
            N_id.append(i)
    return converted_names, res_ids, N_id


def Econtact(d, atom, weight1):
    cont_def = {
        'P': 15,
        "C3'": 20,
        "C1'": 20,
        "C4": 20,
        "N1": 20,
        "N9": 20,
    }
    cont_cutoff = cont_def[atom]
    if d == 0:
        e = -weight1
    elif d == 1:
        e = -weight1
    elif d == 2:
        e = -weight1
    elif d == 3:
        e = -weight1
    elif d <= cont_cutoff:
        e = -weight1
    elif d <= 35:
        e = -0.5 * weight1 * (1 - np.sin(np.pi * (d - 0.5 - (cont_cutoff + 35) / 2) / (35 - cont_cutoff)))
    elif d <= 110:
        e = 0.5 * weight1 * (1 + np.sin(np.pi * (d - 0.5 - (110 + 35) / 2) / (110 - 35)))
    else:
        e = weight1
    e = e - weight1
    return e


def cont_cst_fun1(data, cutoff, cstout, splines_tmp_dir):
    dim = data.shape[0]

    for i in range(dim):
        for j in range(i + 1, dim):
            Prob = data[i][j]
            if (Prob < cutoff): continue
            sep = abs(i - j)
            if sep <= 12: weight1 = 7; weight2 = 1
            if sep > 12 and sep <= 24: weight1 = 6; weight2 = 0.5
            if sep > 24: weight1 = 5; weight2 = 0.25
            xs = np.r_[np.arange(4), np.linspace(3.5, 149.5, 147)]
            for atom in ["P", "C3'", "C1'", "C4", "N1"]:
                if atom == "N1":
                    if seq[i] in ['A', 'G']:
                        atm1 = 'N9'
                    if seq[j] in ['A', 'G']:
                        atm2 = 'N9'
                elif atom == "C4":
                    if seq[i] in ['A', 'G']:
                        atm1 = 'C2'
                    if seq[j] in ['A', 'G']:
                        atm2 = 'C2'
                else:
                    atm1 = atom
                    atm2 = atom

                ys = [Econtact(x, atom, weight1) for x in xs]
                name = splines_tmp_dir + "/%s_%d.%d.txt" % (atom, i + 1, j + 1)
                tmp_name = splines_tmp_dir + "/%s_%d.%d.txt" % (atom, i + 1, j + 1)
                #     print(name)
                out = open(name, 'w')
                out.write('x_axis' + '\t%7.2f' * len(xs) % tuple(xs) + '\n')
                out.write('y_axis' + '\t%7.3f' * len(ys) % tuple(ys) + '\n')
                out.close()
                # weight=1 #a constant 1 seems to be the best
                # format: AtomPair CB 13 CB 37 SPLINE tag fname expt? weight binsize
                form = 'AtomPair %4s %3d %4s %3d SPLINE TAG %s 1.0 %.3f 1 #cont\n'
                pair_i = i + 1
                pari_j = j + 1
                if (abs(i - j) > 0): cstout.write(form % (atm1, pair_i, atm2, pari_j, tmp_name, weight2))


def dist_cst_fun1(dist_npz, cutoff, cstout, splines_tmp_dir, atom):
    data = dist_npz
    dim = data.shape[0]
    EBASE = -0.5
    EREP = [10.0, 3.0, 0.5]  # Repulsion penalty at 0.0, 2.0, 3.0 Angstrom

    for i in range(dim):
        if 'N' in atom:
            if seq[i] in ['A', 'G', 'I']:
                atm1 = 'N9'
            elif seq[i] in ['C', 'U', 'T']:
                atm1 = 'N1'
            else:
                raise ValueError(f'unknow resname:{seq[i]}')
        else:
            atm1 = atom
        for j in range(dim):
            if j <= i: continue
            if abs(i - j) > 0:
                if 'N' in atom:
                    if seq[j] in ['A', 'G']:
                        atm2 = 'N9'
                    elif seq[j] in ['C', 'U']:
                        atm2 = 'N1'
                    else:
                        raise ValueError(f'unknow resname:{seq[j]}')
                else:
                    atm2 = atom
                Prob = data[i][j]
                first_bin = 1
                last_bin = 38
                first_d = 3.5
                weight = 0
                list = []
                for P in Prob[first_bin:last_bin]:
                    if (P > 0): weight += P;
                if (weight < cutoff): continue
                weight3 = 1.0
                if (weight > 0.7): weight3 = 2.0
                if (weight > 0.9): weight3 = 3.0
                p_std = Prob[first_bin:last_bin].std()
                Pnorm = [P for P in Prob[first_bin:last_bin]]
                Pnorm = [P / sum(Pnorm) for P in Pnorm]
                MEFF = 0.0001
                Pref = Pnorm[-1] + MEFF
                xs = []
                ys = []
                DCUT = 39.5
                ALPHA = 0.
                for k, P in enumerate(Pnorm):
                    d = first_d + k
                    dnorm = (d / DCUT) ** ALPHA
                    # dnorm = 1.0
                    E = -2. * (log((P + Pref) / (2. * Pref)))
                    xs.append(d)
                    ys.append(E)
                xs = [0.0, 2.0, 3.0] + xs
                ys = [y + EBASE for y in ys]
                y0 = max(ys[0], 0.0)  # baseline of repulsion energy
                ys = [y0 + EREP[0], y0 + EREP[1], y0 + EREP[2]] + ys  # add repulsion on top of

                name = splines_tmp_dir + "/%s_%d.%d.txt" % (atom, i + 1, j + 1)
                tmp_name = splines_tmp_dir + "/%s_%d.%d.txt" % (atom, i + 1, j + 1)
                #     print(name)
                out = open(name, 'w')
                out.write('x_axis' + '\t%7.2f' * len(xs) % tuple(xs) + '\n')
                out.write('y_axis' + '\t%7.3f' * len(ys) % tuple(ys) + '\n')
                out.close()
                # weight=1 #a constant 1 seems to be the best
                # format: AtomPair CB 13 CB 37 SPLINE tag fname expt? weight binsize
                form = 'AtomPair %4s %3d %4s %3d SPLINE TAG %s 1.0 %.3f 1  #%.3f 0.0 0.0 %.3f\n'
                pair_i = i + 1
                pari_j = j + 1
                if (abs(i - j) > 0): cstout.write(
                    form % (atm1, pair_i, atm2, pari_j, tmp_name, weight3, p_std, weight3))


########################################################################
#   angle (3-1) cnc; ncp
#   theta 0..pi
#########################################################################

def theta_cst_fun1(theta_npz, cutoff, cstout, splines_tmp_dir, count=None):
    first_bin = 1
    last_bin = 13
    first_d = 5
    bin_size = 15

    for angle in theta_npz:
        data = theta_npz[angle]
        dim = data.shape[0]
        converted_names, res_ids, N_id = convert_atm_name(*angle.split('-'))
        for i in range(dim):
            for j in range(dim):
                if i != j:
                    Prob = data[i][j]

                    weight = 0
                    for P in Prob[first_bin:last_bin]:
                        if (P > 0): weight += P
                    if (weight < cutoff):
                        continue
                    elif count is not None:
                        count += 1
                    p_std = Prob[first_bin:last_bin].std()
                    Pnorm = [P for P in Prob[first_bin:last_bin]]
                    Pnorm = [P / sum(Pnorm) for P in Pnorm]
                    MEFF = 0.0001
                    Pref = Pnorm[-1] + MEFF
                    xs = []
                    ys = []
                    for k, P in enumerate(Pnorm):
                        d = first_d + k * bin_size
                        E = -log((P + MEFF))
                        xs.append(d * np.pi / 180)
                        ys.append(E)

                    name = splines_tmp_dir + "/%s_%d.%d.txt" % (angle, i + 1, j + 1)
                    tmp_name = splines_tmp_dir + "/%s_%d.%d.txt" % (angle, i + 1, j + 1)
                    #     print(name)
                    out = open(name, 'w')
                    out.write('x_axis' + '\t%7.2f' * len(xs) % tuple(xs) + '\n')
                    out.write('y_axis' + '\t%7.3f' * len(ys) % tuple(ys) + '\n')
                    out.close()

                    for nid in N_id:
                        resname = seq[eval(f'res_ids[{nid}]') - 1]
                        if resname in ['A', 'G']:
                            converted_names[nid] = 'N9'
                        elif resname in ['C', 'U']:
                            converted_names[nid] = 'N1'
                        else:
                            raise ValueError(f'unknown resname:{resname}')

                    atm1, atm2, atm3 = converted_names
                    atm1_resid, atm2_resid, atm3_resid = res_ids
                    if max(eval(atm1_resid), eval(atm2_resid), eval(atm3_resid)) + 1 > dim:
                        continue

                    form = f'Angle {atm1} {eval(atm1_resid) + 1} {atm2} {eval(atm2_resid) + 1} {atm3} {eval(atm3_resid) + 1} SPLINE TAG {tmp_name} 1.0 1.0 0.26 #{p_std:.3f} 0.0 0.0 {weight:.3f}\n'
                    if (abs(i - j) > 0): cstout.write(form)
    return count


def psi_cst_fun1(psi_npz, cutoff, cstout, splines_tmp_dir, count=None):
    first_bin = 1
    last_bin = 25
    first_d = -175
    bin_size = 15

    for angle in psi_npz:
        data = psi_npz[angle]
        dim = data.shape[0]
        converted_names, res_ids, N_id = convert_atm_name(*angle.split('-'))

        for i in range(dim):
            for j in range(dim):
                if i != j:
                    Prob = data[i][j]

                    weight = 0
                    for P in Prob[first_bin:last_bin]:
                        if (P > 0): weight += P
                    if (weight < cutoff):
                        continue
                    elif count is not None:
                        count += 1
                    p_std = Prob[first_bin:last_bin].std()
                    Pnorm = [P for P in Prob[first_bin:last_bin]]
                    Pnorm = [P / sum(Pnorm) for P in Pnorm]
                    MEFF = 0.0001
                    Pref = Pnorm[-1] + MEFF
                    xs = []
                    ys = []
                    for k, P in enumerate(Pnorm):
                        d = first_d + k * bin_size
                        E = -log((P + MEFF))
                        xs.append(d * np.pi / 180)
                        ys.append(E)

                    name = splines_tmp_dir + "/%s_%d.%d.txt" % (angle, i + 1, j + 1)
                    tmp_name = splines_tmp_dir + "/%s_%d.%d.txt" % (angle, i + 1, j + 1)
                    out = open(name, 'w')
                    out.write('x_axis' + '\t%7.2f' * len(xs) % tuple(xs) + '\n')
                    out.write('y_axis' + '\t%7.3f' * len(ys) % tuple(ys) + '\n')
                    out.close()

                    for nid in N_id:
                        resname = seq[eval(f'res_ids[{nid}]') - 1]
                        if resname in ['A', 'G']:
                            converted_names[nid] = 'N9'
                        elif resname in ['C', 'U']:
                            converted_names[nid] = 'N1'
                        else:
                            raise ValueError(f'unknown resname:{resname}')

                    atm1, atm2, atm3, atm4 = converted_names
                    atm1_resid, atm2_resid, atm3_resid, atm4_resid = res_ids
                    if max(eval(atm1_resid), eval(atm2_resid), eval(atm3_resid), eval(atm4_resid)) + 1 > dim:
                        continue

                    form = f'Dihedral {atm1} {eval(atm1_resid) + 1} {atm2} {eval(atm2_resid) + 1} {atm3} {eval(atm3_resid) + 1} {atm4} {eval(atm4_resid) + 1} SPLINE TAG {tmp_name} 1.0 1.0 0.26 #{p_std:.3f} 0.0 0.0 {weight:.3f}\n'
                    if (abs(i - j) > 0): cstout.write(form)
    return count


def bond_angle_cst_fun1(theta_npz, cutoff, cstout, splines_tmp_dir, intra_weight=10):
    first_bin = 1
    last_bin = 13
    first_d = 73.75
    bin_size = 7.5

    for angle in theta_npz:
        data = theta_npz[angle]
        dim = data.shape[0]
        converted_names, res_ids, N_id = convert_atm_name(*angle.split('_'))

        for i in range(dim):
            Prob = data[i]

            weight = 0
            for P in Prob[first_bin:last_bin]:
                if (P > 0): weight += P
            if (weight < cutoff): continue
            p_std = Prob[first_bin:last_bin].std()
            Pnorm = [P for P in Prob[first_bin:last_bin]]
            Pnorm = [P / sum(Pnorm) for P in Pnorm]
            MEFF = 0.0001
            Pref = Pnorm[-1] + MEFF
            xs = []
            ys = []
            # DCUT = 39.5
            for k, P in enumerate(Pnorm):
                d = first_d + k * bin_size
                E = -log((P + MEFF))
                xs.append(d * np.pi / 180)
                ys.append(E)

            name = splines_tmp_dir + "/%s_%d.txt" % (angle, i + 1)
            tmp_name = splines_tmp_dir + "/%s_%d.txt" % (angle, i + 1)
            #     print(name)
            out = open(name, 'w')
            out.write('x_axis' + '\t%7.2f' * len(xs) % tuple(xs) + '\n')
            out.write('y_axis' + '\t%7.3f' * len(ys) % tuple(ys) + '\n')
            out.close()

            for nid in N_id:
                resname = seq[eval(f'res_ids[{nid}]') - 1]
                if resname in ['A', 'G']:
                    converted_names[nid] = 'N9'
                elif resname in ['C', 'U']:
                    converted_names[nid] = 'N1'
                else:
                    raise ValueError(f'unknown resname:{resname}')

            atm1, atm2, atm3 = converted_names
            atm1_resid, atm2_resid, atm3_resid = res_ids
            if max(eval(atm1_resid), eval(atm2_resid), eval(atm3_resid)) + 1 > dim or min(eval(atm1_resid),
                                                                                          eval(atm2_resid),
                                                                                          eval(atm3_resid)) < 0:
                continue

            form = f'Angle {atm1} {eval(atm1_resid) + 1} {atm2} {eval(atm2_resid) + 1} {atm3} {eval(atm3_resid) + 1} SPLINE TAG {tmp_name} 1.0 {intra_weight:.3f} {bin_size * np.pi / 180:.2f} #{p_std:.3f} 0.0 0.0 {weight:.3f} #1d\n'
            cstout.write(form)


def bond_dihedral_cst_fun1(psi_npz, cutoff, cstout, splines_tmp_dir, intra_weight=10):
    first_bin = 1
    last_bin = 25
    first_d = -175
    bin_size = 15

    for angle in psi_npz:
        data = psi_npz[angle]
        dim = data.shape[0]
        converted_names, res_ids, N_id = convert_atm_name(*angle.split('_'))

        for i in range(dim):
            Prob = data[i]

            weight = 0
            for P in Prob[first_bin:last_bin]:
                if (P > 0): weight += P
            if (weight < cutoff): continue
            p_std = Prob[first_bin:last_bin].std()
            Pnorm = [P for P in Prob[first_bin:last_bin]]
            Pnorm = [P / sum(Pnorm) for P in Pnorm]
            MEFF = 0.0001
            Pref = Pnorm[-1] + MEFF
            xs = []
            ys = []
            for k, P in enumerate(Pnorm):
                d = first_d + k * bin_size

                E = -log((P + MEFF))
                xs.append(d * np.pi / 180)
                ys.append(E)

            name = splines_tmp_dir + "/%s_%d.txt" % (angle, i + 1)
            tmp_name = splines_tmp_dir + "/%s_%d.txt" % (angle, i + 1)
            #     print(name)
            out = open(name, 'w')
            out.write('x_axis' + '\t%7.2f' * len(xs) % tuple(xs) + '\n')
            out.write('y_axis' + '\t%7.3f' * len(ys) % tuple(ys) + '\n')
            out.close()

            for nid in N_id:
                resname = seq[eval(f'res_ids[{nid}]') - 1]
                if resname in ['A', 'G']:
                    converted_names[nid] = 'N9'
                elif resname in ['C', 'U']:
                    converted_names[nid] = 'N1'
                else:
                    raise ValueError(f'unknown resname:{resname}')

            atm1, atm2, atm3, atm4 = converted_names
            atm1_resid, atm2_resid, atm3_resid, atm4_resid = res_ids
            if max(eval(atm1_resid), eval(atm2_resid), eval(atm3_resid), eval(atm4_resid)) + 1 > dim or min(
                    eval(atm1_resid), eval(atm2_resid), eval(atm3_resid), eval(atm4_resid)) < 0:
                continue
            # weight=1 # remove "#" before weight if a constant 1 is needed

            # form = 'Angle %s %d %s %d %s %d SPLINE TAG %s 1.0 1.0 0.26 #0.0 0.0 0.0 %.3f\n'
            # form % (atm1, i + 1, atm2, i + 1, atm3, j + 1, name, weight)
            form = f'Dihedral {atm1} {eval(atm1_resid) + 1} {atm2} {eval(atm2_resid) + 1} {atm3} {eval(atm3_resid) + 1} {atm4} {eval(atm4_resid) + 1} SPLINE TAG {tmp_name} 1.0 {intra_weight:.3f} 0.26 #{p_std:.3f} 0.0 0.0 {weight:.3f} #1d\n'
            cstout.write(form)
