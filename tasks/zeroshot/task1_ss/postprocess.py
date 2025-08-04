import torch
import torch.nn.functional as F


def get_base_index():
    return {'A': 0, 'U': 1, 'C': 2, 'G': 3}


def constraint_matrix_batch(x, loop_min_len=2, is_nc=False):
    base_index = get_base_index()
    base_a = x[:, :, base_index['A']]
    base_u = x[:, :, base_index['U']]
    base_c = x[:, :, base_index['C']]
    base_g = x[:, :, base_index['G']]
    batch = base_a.shape[0]
    length = base_a.shape[1]

    # canonical pairs
    au = torch.matmul(base_a.view(batch, length, 1), base_u.view(batch, 1, length))
    au_ua = au + torch.transpose(au, -1, -2)
    cg = torch.matmul(base_c.view(batch, length, 1), base_g.view(batch, 1, length))
    cg_gc = cg + torch.transpose(cg, -1, -2)
    ug = torch.matmul(base_u.view(batch, length, 1), base_g.view(batch, 1, length))
    ug_gu = ug + torch.transpose(ug, -1, -2)
    ret = au_ua + cg_gc + ug_gu # batch x L x L

    ## non-canonical pairs
    if is_nc:
        ac = torch.matmul(base_a.view(batch, length, 1), base_c.view(batch, 1, length))
        ac_ca = ac + torch.transpose(ac, -1, -2)
        ag = torch.matmul(base_a.view(batch, length, 1), base_g.view(batch, 1, length))
        ag_ga = ag + torch.transpose(ag, -1, -2)
        uc = torch.matmul(base_u.view(batch, length, 1), base_c.view(batch, 1, length))
        uc_cu = uc + torch.transpose(uc, -1, -2)
        aa = torch.matmul(base_a.view(batch, length, 1), base_a.view(batch, 1, length))
        uu = torch.matmul(base_u.view(batch, length, 1), base_u.view(batch, 1, length))
        cc = torch.matmul(base_c.view(batch, length, 1), base_c.view(batch, 1, length))
        gg = torch.matmul(base_g.view(batch, length, 1), base_g.view(batch, 1, length))
        ret += ac_ca + ag_ga + uc_cu + aa + uu + cc + gg

    # remove sharp loop 
    for b in range(batch):
        for i in range(length):
            for j in range(i, length):
                if j-i<loop_min_len+1: # i [i+1, i+2...] i+loop_len+1=j:  [] reprs loop region
                    ret[b, i, j] = ret[b, j, i] = 0
    return ret


def contact_a(a_hat, m):
    a = a_hat * a_hat
    a = (a + torch.transpose(a, -1, -2)) / 2
    a = a * m
    return a


def sign(x):
    return (x > 0).type(x.dtype)


def soft_sign(x):
    k = 1
    return 1.0/(1.0+torch.exp(-2*k*x))


def apply_constraints(u, x, lr_min, lr_max, num_itr, rho=0.0, with_l1=False, s=2.2, is_nc=False):
    """
    :param u: utility matrix, u is assumed to be symmetric, in batch
    :param x: RNA sequence, in batch
    :param lr_min: learning rate for minimization step
    :param lr_max: learning rate for maximization step (for lagrangian multiplier)
    :param num_itr: number of iterations
    :param rho: sparsity coefficient
    :param with_l1:
    :return:
    """
    m = constraint_matrix_batch(x, is_nc=is_nc).float()
    # u with threshold
    # equivalent to sigmoid(u) > 1/(e^(-2.2)+1)
    # u = (u > 2.2).type(torch.FloatTensor) * u
    u = soft_sign(u - s) * u

    # initialization
    a_hat = (torch.sigmoid(u)) * soft_sign(u - s).detach()
    lmbd = F.relu(torch.sum(contact_a(a_hat, m), dim=-1) - 1).detach()

    # gradient descent
    for t in range(num_itr):
        grad_a = (lmbd * soft_sign(torch.sum(contact_a(a_hat, m), dim=-1) - 1)).unsqueeze_(-1).expand(u.shape) - u / 2
        grad = a_hat * m * (grad_a + torch.transpose(grad_a, -1, -2))
        a_hat -= lr_min * grad
        lr_min = lr_min * 0.99

        if with_l1:
            a_hat = F.relu(torch.abs(a_hat) - rho * lr_min)

        lmbd_grad = F.relu(torch.sum(contact_a(a_hat, m), dim=-1) - 1)
        lmbd += lr_max * lmbd_grad
        lr_max = lr_max * 0.99
    out_a = contact_a(a_hat, m)
    return out_a


def postprocess(pred_batch, seq_onehot, nc_map=None, return_nc=False, return_score=False):
    ''' nc: non-canonical '''
    def get_mat_score(pred_ori, pred_post):
        ret = []
        for row_ori, row_post in zip(pred_ori, pred_post):
            val = row_ori[row_post.argmax()]
            new_row = row_ori[:]
            new_row[new_row>val] = new_row.min()
            ret.append(new_row)
        return ret

    ret_pred = []
    ret_pred_nc = []
    ret_score = []
    ret_score_nc = []
    for i in range(pred_batch.shape[0]):
        pred = apply_constraints(pred_batch[i:i+1], seq_onehot[i:i+1], 0.01, 0.1, 100, 1.6, True, 1.5)
        ret_pred.append(pred[0])

        if return_score:
            ret_score.append(get_mat_score(pred_batch[i], pred[0]))
        if return_nc:
            pred_nc = apply_constraints(pred_batch[i:i+1], seq_onehot[i:i+1], 0.01, 0.1, 100, 1.6, True, 1.5, is_nc=True)
            pred_nc =  nc_map * pred_nc
            ret_pred_nc.append(pred_nc[0])
            if return_score:
                ret_score_nc.append(get_mat_score(pred_batch[i], pred_nc[0]))
    return ret_pred, ret_pred_nc, ret_score, ret_score_nc
