import numpy as np
from sklearn.metrics import jaccard_similarity_score
from keras import backend

SMOOTH = 1.e-12

def jaccard_coef(y_true, y_pred):
    # __author__ = Vladimir Iglovikov
    intersection = backend.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = backend.sum(y_true + y_pred, axis=[0, -1, -2])

    jac = (intersection + SMOOTH) / (sum_ - intersection + SMOOTH)

    return backend.mean(jac)


def jaccard_coef_int(y_true, y_pred):
    # __author__ = Vladimir Iglovikov
    y_pred_pos = backend.round(backend.clip(y_pred, 0, 1))

    intersection = backend.sum(y_true * y_pred_pos, axis=[0, -1, -2])
    sum_ = backend.sum(y_true + y_pred_pos, axis=[0, -1, -2])
    jac = (intersection + SMOOTH) / (sum_ - intersection + SMOOTH)
    return backend.mean(jac)


def calc_jacc(model, X_val, y_val):

    prd = model.predict(X_val, batch_size=4)
    avg, trs = [], []

    for i in range(y_val.shape[1]):
        t_msk = y_val[:, i, :, :]
        t_prd = prd[:, i, :, :]
        t_msk = t_msk.reshape(-1, y_val.shape[3])
        t_prd = t_prd.reshape(-1, y_val.shape[3])

        m, b_tr = 0, 0
        N_tr = 100
        for tr in np.arange(N_tr + 1, dtype=float)/N_tr:
            pred_binary_mask = t_prd > tr

            jk = jaccard_similarity_score(t_msk, pred_binary_mask)
            if jk > m:
                m = jk
                b_tr = tr
        print i, m, b_tr
        avg.append(m)
        trs.append(b_tr)

    score = sum(avg) / 10.0
    return score, trs, prd
