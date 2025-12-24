### Additional evaluation    

## #1
def stereochemical(pdb_path):
    ''' stereochemical quality
    # https://sw-tools.rcsb.org/apps/MAXIT/source.html
    # tar -xvf maxit-v10-linux64.tar.gz
    # cd maxit-v10-linux64
    # chmod +x maxit
    # export PATH=$PWD:$PATH
    '''
    cache_dir = '.RNAeval_cache'
    os.makedirs(cache_dir, exist_ok=True)
    out_path = os.path.join(cache_dir, os.path.basename(pdb_path)+'.maxit.txt')
    err_path = os.path.join(cache_dir, os.path.basename(pdb_path)+'.maxit.err')
    os.system(f'maxit -input {pdb_path} > {out_path} 2> {err_Path}')
    os.system(f"grep -Ei 'close|bond length|bond angle|planar|chirality|polymer' {out_path}")
    raise NotImplementedError
    return 


def knot_artifact(pdb_path, ntrials=200):
    '''
    # 3. Knotted artifacts in predicted 3D RNA structures
    - https://github.com/ilbsm/CASP15_knotted_artifacts
    - pip install topoly
    '''
    import topoly
    from topoly import alexander
    coords = topoly.read_xyz(pdb_path, atoms=["P","O5'","C5'","C4'","C3'","O3'"])
    result = alexander(coords, ntrials=ntrials) # random 闭合, compute Alexander polynomial
    return result ## <50%: unknot, if knot, output knot type, such as "3_1" for trefoil knot


def entanglement():
    # RNA 3D structure entanglement
    # https://www.cs.put.poznan.pl/mantczak/spider.zip
    raise NotImplementedError
