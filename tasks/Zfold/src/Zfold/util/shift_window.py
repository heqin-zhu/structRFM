import numpy as np


def shift_window(L:int, window:int=150, shift:int=50):
    '''
    Generate shift windows. Approximate (L/shift)**2 windows.

    Parameters
    ----------
    L: int
        seq length
    window: int
    shift: int

    Returns
    -------
    ret: dict
        {
           'selections': [mask], 
           'intervals': [interval_list], 
           'count_1d': np.array((L)),
           'count_2d': np.array((L, L)),
          }
    '''
    grids = np.arange(0, L - window + shift, shift)
    num_grids = len(grids)
    idxs = np.arange(L)
    weight = 1
    ret = {
           'selections': [], 
           'intervals': [], 
           'sel_idx_2d': [], 
           'count_1d': np.zeros((L)),
           'count_2d': np.zeros((L, L)),
          }
    for i in range(num_grids):
        for j in range(i, num_grids):
            start_1 = grids[i]
            end_1 = min(grids[i] + window, L)
            start_2 = grids[j]
            end_2 = min(grids[j] + window, L)
            intervals = []
            if end_1 >= start_2:
                intervals.append((start_1, end_2))
            else:
                intervals.append((start_1, end_1))
                intervals.append((start_2, end_2))
            sel = np.zeros((L)).astype(np.bool_)
            sel[start_1:end_1] = True
            sel[start_2:end_2] = True
            ret['selections'].append(sel)
            ret['intervals'].append(intervals)
            sel_idxs = idxs[sel]
            sel_idx_2d = np.ix_(sel_idxs, sel_idxs)
            ret['sel_idx_2d'].append(sel_idx_2d)
            ret['count_1d'][sel_idxs] += weight
            ret['count_2d'][sel_idx_2d] += weight
            # mat = mat[sel][:, sel]
    return ret


if __name__ == '__main__':
    L = 200
    ret = shift_window(L)
    print(len(ret['selections']))
    print(len(ret['intervals']))
    # print(ret['intervals'])
    print(ret['count_1d'])
    print(ret['count_2d'])
