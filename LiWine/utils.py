import scipy.io as scio
import numpy as np
from numpy import linalg
from scipy.sparse import csc_matrix
from scoring import evaluate
from gensim.models import KeyedVectors


def load_data(prefix='.', dataset='ppi', init=True, init_method='SVD', context='N1'):


    filename = '{}/data/{}.mat'.format(prefix, dataset)
    print(filename)
    
    mat = scio.loadmat(filename)
    network = mat['network']
    labels = mat['group'].tocoo()
    row = labels.row
    col = labels.col
    labels = {}
    for i in range(len(row)):
        if(row[i] not in labels):
            labels[row[i]] = set([])
        labels[row[i]].add(col[i])

   
    if context == 'N2':
        network = network * network + network
    normalized_term = np.array(network.sum(axis=1)) + 1e-12


    U, V = None, None
    w2v_method = ['deepwalk', 'graphsage', 'n2v', 'verse']
    mat_method = ['pawine', 'AROPE']
    if init == True:

        if init_method in w2v_method:
            embeddings_file = '{}/init/{}/{}_{}.embeddings'.format(prefix, dataset, dataset, init_method)
            model = KeyedVectors.load_word2vec_format(embeddings_file, binary=False)
            U = V = np.asarray([model[str(node)] for node in range(len(labels))])
        elif init_method in mat_method or (init_method == 'BPR' and (dataset == 'citeseer' or dataset == 'wiki')):
            U_filename = '{}/init/{}/U0_{}.mat'.format(prefix, dataset, init_method)
            U = V = scio.loadmat(U_filename)['U'].astype(float)
        else:
            U_filename = '{}/init/{}/U0_{}128.mat'.format(prefix, dataset, init_method)
            V_filename = '{}/init/{}/V0_{}128.mat'.format(prefix, dataset, init_method)
            if dataset == 'ppi' and init_method == 'BPR':
                U = scio.loadmat(U_filename)['U0'].astype(float)
                V = scio.loadmat(V_filename)['V0'].astype(float)
            else:
                U = scio.loadmat(U_filename)['U'].astype(float)
                V = scio.loadmat(V_filename)['V'].astype(float)

    print(U.max(), U.min(), U.mean(), U.var())
    return network, labels, normalized_term, U, V


def test_init(prefix='..', dataset='ppi'):
    filename = '{}/data/{}.mat'.format(prefix, dataset)
    mat = scio.loadmat(filename)
    network = mat['network']
    labels = mat['group'].tocoo()
    row = labels.row
    col = labels.col
    labels = {}
    for i in range(len(row)):
        if (row[i] not in labels):
            labels[row[i]] = set([])
        labels[row[i]].add(col[i])

    U, V = None, None
    U_filename = '{}/init/{}/U0_SVD128.mat'.format(prefix, dataset)
    V_filename = '{}/init/{}/V0_SVD128.mat'.format(prefix, dataset)
    U = scio.loadmat(U_filename)['U'].astype(float)
    V = scio.loadmat(V_filename)['V'].astype(float)
    evaluate(U, labels, dataset)

def gen_svd(prefix='.', dataset='citeseer'):
    print(dataset)
    graph = np.loadtxt("{}/data/{}/graph.txt".format(prefix, dataset)).astype(int)
    labels = np.loadtxt("{}/data/{}/group.txt".format(prefix, dataset)).astype(int)

    network = np.zeros((len(labels), len(labels)))
    network[graph[:, 0], graph[:, 1]] = 1.0

    U,_, V = linalg.svd(network)
    savefile = '{}/init/{}/{}.mat'
    U = {'U': U[:,:128]}
    V = {'V': V[:128, :].T}
    scio.savemat(savefile.format(prefix, dataset, 'U0_SVD128'), U)
    scio.savemat(savefile.format(prefix, dataset, 'V0_SVD128'), V)


