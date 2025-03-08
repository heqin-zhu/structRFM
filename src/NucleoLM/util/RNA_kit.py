import numpy as np

from .misc import get_file_name


def read_fasta(path):
    '''
    Read iterator of all sequences from a fasta file.

    Parameters
    ----------
    path: str
        Path of fasta file.

    Returns
    -------
    yield: (str, str)
        (seq_name, seq)
    '''
    seq_name = None
    with open(path) as fp:
        for line in fp.readlines():
            line = line.strip(' \n\r\t')
            if line.startswith('#') or line=='':
                continue
            elif line.startswith('>'):
                seq_name = line[1:]
            else:
                yield seq_name, line


def write_fasta(path:str, name_seq_pairs:[(str, str)])->None:
    with open(path, 'w') as fp:
        for name, seq in name_seq_pairs:
            fp.write(f'>{name}\n{seq}\n')


def read_SS(path:str, return_index:bool=False):
    '''
    Read secondary structure from bpseq/ct/dbn file.

    Parameters
    ----------
    path: str
        path of ss file, endswith '.bpseq', '.ct', '.dbn'
    return_index : bool
        indicate whether this function returns indexes of all bases

    Returns
    -------
    seq: str, length L
        Containing RNA bases: AUGC.
    connects: [int], length L
        The i-th base connects to `connects[i-1]`-th base, 1-indexed, 0 for no connection.
    index: [int], length L
        list of base indexes, if valid, = list(range(1, 1+L))
    '''
    low_path = path.lower()
    if low_path.endswith('.bpseq'):
        return read_bpseq(path, return_index)
    elif low_path.endswith('.ct'):
        return read_ct(path, return_index)
    elif low_path.endswith('.dbn'):
        read_dbn(path, return_index)
    else:
        raise Exception(f'[Error] Unkown file type: {path}')


def read_dbn(path:str, return_index:bool=False):
    with open(path) as fp:
        seq = None
        line = fp.readline().strip('\r\n ')
        if set(line.upper()).issubset(set('AUGC')):
            seq = line
            line = fp.readline().strip('\r\n ')
        connects = dbn2connects(line)
        if return_index:
            return seq, connects, list(range(1, 1+len(connects)))
        else:
            return seq, connects


def read_bpseq(path:str, return_index:bool=False):
    '''
    Read bpseq file.

    Parameters
    ----------
    path: str
        Path of bpseq file.
    return_index: bool
        indicate whether this function returns indexes of all bases

    Returns
    -------
    seq: str, length L
        Containing RNA bases: AUGC.
    connects: [int], length L
        The i-th base connects to `connects[i-1]`-th base, 1-indexed, 0 for no connection.
    index: [int], length L
        list of base indexes, if valid, = list(range(1, 1+L))
    '''
    bases = []
    connects = []
    indexes = []
    with open(path) as f:
        for line in f.readlines():
            if not line.startswith('#'):
                idx, base, conn = [item for item in line.strip('\n\t\r ').split() if item]
                bases.append(base)
                connects.append(conn)
                indexes.append(int(idx))
    connects = [int(i) for i in connects]
    if return_index:
        return ''.join(bases), connects, indexes
    else:
        return ''.join(bases), connects


def read_ct(path:str, return_index:bool=False):
    '''
    Read ct file.

    Parameters
    ----------
    path: str
        Path of ct file.
    return_index: bool
        indicate whether this function returns indexes of all bases

    Returns
    -------
    seq: str, length L
        Containing RNA bases: AUGC.
    connects: [int], length L
        The i-th base connects to `connects[i-1]`-th base, 1-indexed, 0 for no connection.
    index: [int], length L
        list of base indexes, if valid, = list(range(1, 1+L))
    '''
    bases = []
    connects = []
    indexes = []
    ct = 0
    with open(path) as f:
        for i, line in enumerate(f.readlines()):
            if line.startswith('#'):
                ct +=1
                continue
            if ct == i:
                continue
            items = [item for item in line.strip('\n\t\r ').split() if item]
            if len(items)!=6 or int(items[0])!=i-ct:
                break
            idx, base, _, _, conn, _ = items
            bases.append(base)
            connects.append(conn)
            indexes.append(int(idx))
    connects = [int(i) for i in connects]
    if return_index:
        return ''.join(bases), connects, indexes
    else:
        return ''.join(bases), connects


def write_SS(path:str, seq:str, connects:[int], out_type='bpseq')->None:
    '''
    Write secondary structure to bpseq/ct/dbn.

    Parameters
    ----------
    path: str
        Dest path.
    seq: str, length L
        Containing RNA bases: AUGC.
    connects: [int], length L
        The i-th base connects to `connects[i-1]`-th base, 1-indexed, 0 for no connection.
    out_type: str
        bpseq, ct, dbn
    '''
    if 'bpseq' in out_type.lower():
        write_bpseq(path, seq, connects)
    elif 'ct' in out_type.lower():
        write_ct(path, seq, connects)
    elif 'dbn' in out_type.lower():
        write_dbn(path, seq, connects)
    else:
        raise Exception(f'[Error] Unkown output secondary structure file type: {out_type}')


def write_dbn(path:str, seq:str, connects:[int])->None:
    '''
    Write secondary structure to dbn file.

    Parameters
    ----------
    path: str
        Dest path.
    seq: str, length L
        Containing RNA bases: AUGC.
    connects: [int], length L
        The i-th base connects to `connects[i-1]`-th base, 1-indexed, 0 for no connection.
    '''
    with open(path, 'w') as f:
        dbn = connects2dbn(connects)
        f.write(f'{seq}\n{dbn}\n')


def write_ct(path:str, seq:str, connects:[int])->None:
    '''
    Write secondary structure to ct file.

    Parameters
    ----------
    path: str
        Dest path.
    seq: str, length L
        Containing RNA bases: AUGC.
    connects: [int], length L
        The i-th base connects to `connects[i-1]`-th base, 1-indexed, 0 for no connection.
    '''
    with open(path, 'w') as f:
        for i, (base, connect) in enumerate(zip(seq, connects)):
            f.write(f'{i+1} {base.upper()} {i} {i+2} {connect} {i+1}\n')


def write_bpseq(path:str, seq:str, connects:[int], comments=None)->None:
    '''
    Write secondary structure to bpseq file.

    Parameters
    ----------
    path: str
        Dest path.
    seq: str, length L
        Containing RNA bases: AUGC.
    connects: [int], length L
        The i-th base connects to `connects[i-1]`-th base, 1-indexed, 0 for no connection.
    '''
    with open(path, 'w') as f:
        if comments is not None:
            f.write(f'# {comm}\n')
        for i, (base, connect) in enumerate(zip(seq, connects)):
            f.write(f'{i+1} {base.upper()} {connect}\n')


def dispart_nc_pairs(seq:str, connects:[int])->[int]:
    '''
    Dispart canonical and noncanonical pairs in connects.

    Parameters
    ----------
    seq: str
        RNA base seq
    connects: [int], length L
        The i-th base connects to `connects[i-1]`-th base, 1-indexed, 0 for no connection.

    Returns
    -------
    connects: [int], length L
    nc_connects: [int], length L
    '''
    seq = seq.upper()
    canonical_pairs = {'AU', 'UA', 'GC', 'CG', 'GU', 'UG'}
    conns = [0] * len(seq)
    nc_conns = [0] * len(seq)
    for idx, (base, conn) in enumerate(zip(seq, connects)):
        if conn!=0 and idx<conn-1:
            pair = base + seq[conn-1]
            if pair in canonical_pairs:
                conns[idx] = conn
                conns[conn-1] = idx+1
            else:
                nc_conns[idx] = conn
                nc_conns[conn-1] = idx+1
    return conns, nc_conns


def merge_connects(connects:[int], nc_connects:[int])->[int]:
    '''
    Merge canonical and noncanonical pairs in connects.

    Parameters
    ----------
    connects: [int], length L
        The i-th base connects to `connects[i-1]`-th base, 1-indexed, 0 for no connection.

    Returns
    -------
    connects: [int], length L
    '''
    n = len(connects)
    n1 = len(nc_connects)
    assert n==n1, f'length mismatch: {n}!={n1}'
    ret_connects = connects[:]
    dic = {i+1: conn for i, conn in enumerate(connects)}
    for i, conn in enumerate(nc_connects):
        if conn!=0 and i<conn:
            if connects[i]==0 and connects[conn-1]==0:
                ret_connects[i] = conn
                ret_connects[conn-1] = i+1
    return ret_connects


def remove_lone_pairs(connects:[int], loop_len:int=3)->[int]:
    '''
    Remove lone pairs in connects

    Parameters
    ----------
    connects: [int], length L
        The i-th base connects to `connects[i-1]`-th base, 1-indexed, 0 for no connection.
    loop_len: int
        interval of lone pair and other pair

    Returns
    -------
    connects: [int], length L
        The i-th base connects to `connects[i-1]`-th base, 1-indexed, 0 for no connection.
    '''
    L = len(connects)
    ret_conn = [0] * L
    for idx, conn in enumerate(connects):
        if conn!=0 and idx<conn-1:
            isLone = True
            for center, conj in [(idx, conn-1), (conn-1, idx)]:
                for dire in [-1, 1]:
                    neighbor = center + dire
                    if 0 <= neighbor < L:
                        if connects[neighbor]-1 in [conj-i*dire for i in range(1, loop_len+1)]:
                            isLone = False
                            break
                if not isLone:
                    break
            if not isLone:
                ret_conn[idx] = conn
                ret_conn[conn-1] = idx+1
    return ret_conn


def dbn2connects(dbn:str)->[int]:
    '''
    Convert dbn to connects.

    Parameters
    ----------
    dbn: str
        Dot-bracket notation of secondary structure, including '.()[]{}', with valid brackets.

    Returns
    -------
    connects: [int], length L
        The i-th base connects to `connects[i-1]`-th base, 1-indexed, 0 for no connection.
    '''
    alphabet = ''.join([chr(ord('A')+i) for i in range(26)])
    alphabet_low = alphabet.lower()
    syms = ('([{<' + alphabet)[::-1]
    syms_conj = (')]}>' + alphabet_low)[::-1]
    left2right = {p: q for p, q in zip(syms, syms_conj)}
    right2left = {p: q for p, q in zip(syms_conj, syms)}

    pair_dic = {}
    stack_dic = {char: [] for char in left2right}
    for i, char in enumerate(dbn):
        idx = i+1
        if char=='.':
            pair_dic[idx] = 0
        elif char in left2right:
            stack_dic[char].append((idx, char))
        elif char in right2left:
            cur_stack = stack_dic[right2left[char]]
            if len(cur_stack)==0:
                raise Exception(f'[Error] Invalid brackets: {dbn}')
            p, ch = cur_stack.pop()
            pair_dic[p] = idx
            pair_dic[idx] = p
        else:
            raise Exception(f'[Error] Unknown DBN representation: dbn[{i}]={char}: {dbn}')
    if any(stack for k, stack in stack_dic.items()):
        raise Exception(f'[Error] Brackets dismatch: {dbn}')
    connects = [pair_dic[i] for i in range(1, 1+ len(dbn))]
    return connects


def connects2dbn(connects:[int])->str:
    '''
    Convert connects to dbn. [Warning] Can't deal with pseudo knot.

    Parameters
    ----------
    connects: [int], length L
        The i-th base connects to `connects[i-1]`-th base, 1-indexed, 0 for no connection.

    Returns
    -------
    dbn: str
        Dot-bracket notation of secondary structure, including '.()[]{}', with valid brackets.
    '''
    alphabet = ''.join([chr(ord('A')+i) for i in range(26)])
    alphabet_low = alphabet.lower()
    syms = ('([{<' + alphabet)[::-1]
    syms_conj = (')]}>' + alphabet_low)[::-1]
    syms_set = set(syms)
    syms_conj_set = set(syms_conj)
    ret = ['.']*len(connects)
    for i, conn in enumerate(connects):
        pi = conn-1
        if pi != -1 and pi>=i:
            counts = [0] * len(syms)
            for j in range(i+1, pi):
                sym = ret[j]
                if sym in syms_set:
                    ct_idx = syms.index(sym)
                    counts[ct_idx] +=1
                elif sym in syms_conj_set:
                    ct_idx = syms_conj.index(sym)
                    counts[ct_idx] -=1
            for idx, ct in enumerate(counts):
                if ct==0:
                    ret[i] = syms[idx]
                    ret[pi] = syms_conj[idx]
    return ''.join(ret)


def arr2connects(arr)->[int]:
    '''
    Convert contact map to connects.

    Parameters
    ----------
    arr: numpy.ndarray
        LxL matrix where the arr[i,j]=1 represents paired bases, otherwise 0

    Returns
    -------
    connects: [int], length L
        The i-th base connects to `connects[i-1]`-th base, 1-indexed, 0 for no connection.
    '''
    connects = arr.argmax(axis=1)
    connects[connects!=0] +=1
    if connects[0]!=0:
        connects[connects[0]-1] = 1
    return connects.tolist()


def connects2arr(connects:[int]):
    '''
    Convert connects to contact map.

    Parameters
    ----------
    connects: [int], length L
        The i-th base connects to `connects[i-1]`-th base, 1-indexed, 0 for no connection.

    Returns
    -------
    arr: numpy.ndarray
        LxL matrix where the arr[i,j]=1 represents paired bases, otherwise 0
    '''
    L = len(connects)
    ret = np.zeros((L, L))
    for num, conn in zip(range(1, L+1), connects):
        if conn!=0:
            ret[num-1][conn-1] = ret[conn-1][num-1] = 1
    return ret


def valid_ss(seq:str, connects:[int], indexes:[int]=None)->bool:
    return len(seq)>4 and len(seq) == len(connects) and min(connects)>=0 and max(connects)<=len(seq) and (indexes is None or indexes==list(range(1, 1+len(seq))))


def is_valid_bracket(s, ignore_unknown=False):
    chars = set('.()[]{}')
    left2right = {p: q for p, q in ['()', '[]', '{}']}
    right2left = {q: p for p, q in ['()', '[]', '{}']}
    count_dic = {ch: 0 for ch in chars}

    for i, char in enumerate(s):
        count_dic[char] += 1
        if char in '.([{':
            pass
        elif char in ')]}':
            if count_dic[char] > count_dic[right2left[char]]:
                return False
        else:
            if not ignore_unknown:
                raise Exception(f'[Error] Unknown brackets repr: {s}')
    return all(count_dic[char]==count_dic[left2right[char]] for char in '([{')


def arr2scores(arr, connects, mode)->[float]:
    '''
    NOTICE! Deprecated, discarded!

    Convert contact map to prob scores.

    Parameters
    ----------
    arr: numpy.ndarray
        LxL matrix where the arr[i,j]=1 represents paired bases, otherwise 0
    connects: [int], length L
        The i-th base connects to `connects[i-1]`-th base, 1-indexed, 0 for no connection.
    mode: str
        prob, energy, softmax

    Returns
    -------
    scores: [float], length L
        range [0, 1]
    '''
    def get_pred_score(xs, i, mode):
        if i is None:
            if xs.max()==xs.min()==0:
                return 0
            else:
                return 1
        if mode == 'prob':
            prob = xs[i]
            if prob<=0:
                return 0.0
            if prob>1:
                return 1.0
            else:
                return prob
        elif mode == 'energy':
            RT = 2.4788 # kj/mol
            xs = -xs/RT
            x = xs[i]
            sm_exp = (np.exp(xs)).sum()
            return np.exp(x)/sm_exp
        elif mode == 'softmax':
            x = xs[i]
            sm_exp = (np.exp(xs)).sum()
            return np.exp(x)/sm_exp
    scores = []
    for i in range(len(arr)):
        score = get_pred_score(arr[i], None if connects[i]==0 else connects[i]-1, mode)
        scores.append(score)
    return np.array(scores).tolist()


def parse_stockholm(lines):
    '''
    Stockholm format (e.g. Rfam seed file).

    Parameters
    ----------
    lines : [str]
        lines read from file

    Returns
    -------
    ret: dict
        {'headers': {tag: value}, 'seqs': {name: seq}, 'SS': dbn}
    '''
    ret = {'headers': {}, 'seqs': {}, 'SS': ''}
    for line in lines:
        line = line.strip('\r\n')
        parts = [part for part in line.split() if part]
        if line.startswith('#=GF'):
            ret['headers'][parts[1]] = parts[2]
        elif not line.startswith('#'):
            if len(parts) == 2:
                seq_name, seq = parts
                seq_id = '_'.join(seq_name.split('/'))
                if seq_id in ret['seqs']:
                    ret['seqs'][seq_id] += seq
                else:
                    ret['seqs'][seq_id] = seq
        elif line.startswith('#=GC SS_cons'):
            ret['SS'] += parts[2]
    return ret


def read_stockholm(path):
    '''
    Stockholm format (e.g. Rfam seed file).
    Other method to read: scikit-bio
    ``
        from skbio import Protein, TabularMSA
        msa = TabularMSA.read(fp, constructor=Protein)
    ```

    Parameters
    ----------
    path: str
        stockholm file path

    Returns
    -------
    ret: dict
        {'headers': {tag: value}, 'seqs': {name: seq}, 'SS': dbn}
    '''
    with open(path) as fp:
        return parse_stockholm(fp.readlines())


def process_stockholm_SS(seq, dbn):
    '''
        _：表示间隔区域。
        -：表示缺失或未配对的区域。
        :：表示这些位置的碱基保守性较高或成对概率较高。
        ~: 
        The RNA SS letters are taken from WUSS (Washington University Secondary Structure) notation. Matching nested parentheses characters <>, (), [], or {} indicate a basepair. The symbols '.', ',' and ';' indicate unpaired regions. Matched upper and lower case characters from the English alphabet indicate pseudoknot interactions. The 5' nucleotide within the knot should be in uppercase and the 3' nucleotide lowercase.
    '''
    base_set = set('AUGC')
    unpaired_sym = set('.,_-:~')
    seq = seq.upper()
    ref_dbn = ['.' if sym in unpaired_sym else sym for sym in dbn]
    connects = dbn2connects(ref_dbn)
    # firstly, process ref_dbn, unpair unkown base pairs
    for idx, base in enumerate(seq):
        if base not in base_set: # discard
            if connects[idx]!=0: # if have pair, unpair it
                ref_dbn[idx] = ref_dbn[connects[idx]-1] = '.'
    # then, get final seq and dbn
    seq_lst = []
    dbn_lst = []
    for idx, base in enumerate(seq):
        if base in base_set:
            seq_lst.append(base)
            dbn_lst.append(ref_dbn[idx])
    return ''.join(seq_lst), ''.join(dbn_lst)


def compute_confidence(mat1, mat2, scale_k:float=1.5220298, scale_b:float=-0.08571319, ceiling_01:bool=True)->float:
    return cos_row(mat1, mat2, scale_k, scale_b, ceiling_01)


def cos_row(mat1, mat2, scale_k:float=1, scale_b:float=0, ceiling_01:bool=True)->float:
    '''
    Compute confidence index according to contact maps before and after postprocessing.

    Parameters
    ----------
    mat: np.ndarray
        Contact map.

    Returns
    -------
    CI: float
        confidence index
    '''
    CEILING = 0.98
    FLOOR = 0.0
    inner_prod = 0
    norm1 = norm2 = 0
    for arr1, arr2 in zip(mat1, mat2):
        # np.linalg.norm, np.linalg.det
        norm1 += np.dot(arr1, arr1)
        norm2 += np.dot(arr2, arr2)
        inner_prod += np.dot(arr1, arr2)
    if norm1 == 0 and norm2 == 0: # NOTICE! zero vector
        return CEILING # don't use 1
    elif norm1 == 0 or norm2 == 0:
        return FLOOR
    else:
        CI = inner_prod/((norm1*norm2)**0.5)
        # The linear coeffs are for rescale CI to [0, 1], which is the range of F1
        if scale_k and scale_b is not None:
            CI = scale_k*(CI.item()) + scale_b
        if ceiling_01:
            return min(CEILING, max(FLOOR, CI))
        else:
            return CI
