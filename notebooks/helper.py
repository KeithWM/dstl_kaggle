import itertools
import numpy as np
import h5py

N_CLASSES = 10

def image_iterator(scope=False):
    for id0, id1, id2 in itertools.product(range(6010, 6190, 10), range(5), range(5)):
        id = '{:d}_{:d}_{:d}'.format(id0, id1, id2)
        if not scope or id in scope:
            yield id

def split_data(band, return_id=False):
    get = h5py.File('/data/dstl/trn_val_{}.hdf5'.format(band), 'r')
    X_val = get['X_val'][:].transpose([0, 3, 1, 2])
    y_val = get['y_val'][:].transpose([0, 3, 1, 2])
    X_trn = get['X_trn'][:].transpose([0, 3, 1, 2])
    y_trn = get['y_trn'][:].transpose([0, 3, 1, 2])
    if return_id:
        id_val = get['id_val']
        id_trn = get['id_trn']
        choice = get['choice']
        get.close()
        return X_val, y_val, X_trn, y_trn, id_val, id_trn, choice
    else:
        get.close()
        return X_val, y_val, X_trn, y_trn

def get_patches(band, isz, classes=range(N_CLASSES), get=None):
    if get is None:
        get = h5py.File('/data/dstl/trn_val_{}.hdf5'.format(band), 'r')
    X_val = get['X_val'][:].transpose([0, 3, 1, 2])
    y_val = get['y_val'][:].transpose([0, 3, 1, 2])[:, classes, ...]
    X_trn = get['X_trn'][:].transpose([0, 3, 1, 2])
    y_trn = get['y_trn'][:].transpose([0, 3, 1, 2])[:, classes, ...]

    X_scale = 2**11 # the images are in 11 bit format
    X_val = (2*X_val - X_scale).astype(float)/float(X_scale)
    X_trn = (2*X_trn - X_scale).astype(float)/float(X_scale)

    i_val = np.random.randint(X_val.shape[-2] + 1 - isz)
    j_val = np.random.randint(X_val.shape[-1] + 1 - isz)

    i_trn = np.random.randint(X_val.shape[-2] + 1 - isz)
    j_trn = np.random.randint(X_val.shape[-1] + 1 - isz)
    
    get.close()

    return X_val[..., i_val:i_val+isz, j_val:j_val+isz], y_val[..., i_val:i_val+isz, j_val:j_val+isz], \
            X_trn[..., i_trn:i_trn+isz, j_trn:j_trn+isz], y_trn[..., i_trn:i_trn+isz, j_trn:j_trn+isz]

def get_more_patches(band, isz, repetitions=1, classes=range(N_CLASSES)):
    get = h5py.File('/data/dstl/trn_val_{}.hdf5'.format(band), 'r')
    N_classes = len(classes)

    N_val = get['X_val'][:].shape[0]
    N_trn = get['X_trn'][:].shape[0]
    depth = get['X_val'][:].shape[3]

    X_vals = np.zeros((repetitions * N_val, depth, isz, isz), dtype=float)
    y_vals = np.zeros((repetitions * N_val, N_classes, isz, isz), dtype=float)
    X_trns = np.zeros((repetitions * N_trn, depth, isz, isz), dtype=float)
    y_trns = np.zeros((repetitions * N_trn, N_classes, isz, isz), dtype=float)
    for rep in range(repetitions):
        X_val, y_val, X_trn, y_trn = get_patches(band, isz, classes=classes, get=get)

        X_vals[rep*N_val:(rep+1)*N_val, ...] = X_val
        y_vals[rep*N_val:(rep+1)*N_val, ...] = y_val
        X_trns[rep*N_trn:(rep+1)*N_trn, ...] = X_trn
        y_trns[rep*N_trn:(rep+1)*N_trn, ...] = y_trn

    return X_vals, y_vals, X_trns, y_trns

def classes_string(classes):
    return ''.join(['{:d}'.format(c) for c in classes])

if __name__ == "__main__":
    for im_id in image_iterator():
        print im_id, ' ',

    print '\n'
    for im_id in image_iterator(('6110_1_1', '6120_0_3', 'foo', 6010, '6010_0_0_bar')):
        print im_id, ' ',
    print '\n'

    X_val, y_val, X_trn, y_trn = get_patches('M', 12, classes=(0,1,2))
    print y_val.shape

    X_val, y_val, X_trn, y_trn = get_more_patches('M', 12, classes=(0,1,2))
    print y_val.shape