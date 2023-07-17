import numpy as np
import jax.numpy as jnp
import jax.scipy as jsp
from scipy.linalg import solve_discrete_are
from examples.osc_mass import static_canon_osqp
import cvxpy as cp
import yaml
from l2ws.launcher import Workspace
import os
from examples.solve_script import direct_osqp_setup_script
import scipy.linalg as la
import urllib.request
import os
import gzip
import matplotlib.pyplot as plt
import hydra
from emnist import extract_training_samples


def run(run_cfg):
    example = "mnist"
    data_yaml_filename = 'data_setup_copied.yaml'

    # read the yaml file
    with open(data_yaml_filename, "r") as stream:
        try:
            setup_cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            setup_cfg = {}

    # set the seed
    np.random.seed(setup_cfg['seed'])

    lambd = setup_cfg['lambd']
    blur_size = setup_cfg['blur_size']

    # static_dict = dict(A=A, lambd=lambd, ista_step=ista_step)

    # get the blur matrix
    B = vectorized2DBlurMatrix(28, 28, blur_size)

    # get P, A
    P = B.T @ B
    m, n = 784, 784
    A = np.eye(n)

    rho_vec, sigma = jnp.ones(m), 1
    # rho_vec = rho_vec.at[l == u].set(1000)
    M = P + sigma * jnp.eye(n) + A.T @ jnp.diag(rho_vec) @ A

    factor = jsp.linalg.lu_factor(M)

    m, n = A.shape
    static_dict = dict(factor=factor, P=P, A=A, rho=rho_vec)

    # we directly save q now
    static_flag = True
    algo = 'osqp'
    workspace = Workspace(algo, run_cfg, static_flag, static_dict, example, 
                          custom_visualize_fn=custom_visualize_fn)

    # run the workspace
    workspace.run()



def setup_probs(setup_cfg):
    cfg = setup_cfg
    N_train, N_test = cfg.N_train, cfg.N_test
    N = N_train + N_test
    lambd = cfg.lambd
    emnist = cfg.get('emnist', True)

    # np.random.seed(setup_cfg['seed'])
    # m_orig, n_orig = setup_cfg['m_orig'], setup_cfg['n_orig']
    # A = jnp.array(np.random.normal(size=(m_orig, n_orig)))
    # evals, evecs = jnp.linalg.eigh(A.T @ A)
    # ista_step = 1 / evals.max()
    # lambd = setup_cfg['lambd']

    np.random.seed(cfg.seed)

    # save output to output_filename
    output_filename = f"{os.getcwd()}/data_setup"

    # P, A, cones, q_mat, theta_mat_jax, A_tensor = multiple_random_sparse_pca(
    #     n_orig, cfg.k, cfg.r, N, factor=False)
    # P_sparse, A_sparse = csc_matrix(P), csc_matrix(A)
    # b_mat = generate_b_mat(A, N, p=.1)
    # m, n = A.shape

    # setup the training

    lambd = setup_cfg['lambd']
    blur_size = setup_cfg['blur_size']

    # load the mnist images
    x_train, x_test = get_mnist(emnist=emnist)

    # get the blur matrix
    B = vectorized2DBlurMatrix(28, 28, blur_size)

    # get P, A
    P = B.T @ B
    m, n = 784, 784
    A = np.eye(n)

    q_mat = jnp.zeros((N, 2 * m + n))
    q_mat = q_mat.at[:, m + n:].set(jnp.inf) #q_mat.at[:, m + n:].set(1)

    theta_mat = jnp.zeros((N, n))

    # blur img
    blurred_imgs = []
    for i in range(N):
        blurred_img = jnp.reshape(B @ x_train[i, :], (28, 28))
        blurred_img_vec = jnp.ravel(blurred_img)
        q_mat = q_mat.at[i, :n].set(-B.T @ blurred_img_vec + lambd)
        theta_mat = theta_mat.at[i, :].set(blurred_img_vec)
        blurred_imgs.append(blurred_img)

        # create cvxpy problem with TV regularization

        # get P, A, q, l, u with cvxpy osqp canonicalization
        # lambd = 1e-6
        # P, A, c, l, u = mnist_canon(A, lambd, blurred_img)
        # mnist_canon(A, lambd, blurred_img)
        


    # blur the images


    # create the cvxpy problem with parameter


    # create theta_mat and q_mat
    # q_mat = jnp.vstack([q_mat_train, q_mat_test])
    # theta_mat = jnp.vstack([theta_mat_train, theta_mat_test])


    # osqp_setup_script(theta_mat, q_mat, P, A, output_filename, z_stars=z_stars)
    P = B.T @ B
    z_stars = direct_osqp_setup_script(theta_mat, q_mat, P, A, output_filename, z_stars=None)

    if not os.path.exists('images'):
        os.mkdir('images')

    # save blurred images
    for i in range(5):
        blurred_img = blurred_imgs[i]
        f, axarr = plt.subplots(1,3)
        axarr[0].imshow(blurred_img, cmap=plt.get_cmap('gray'), label='blurred')
        axarr[0].set_title('blurred')

        sol_img = np.reshape(z_stars[i, :784], (28,28))
        axarr[1].imshow(sol_img, cmap=plt.get_cmap('gray'), label='solution')
        axarr[1].set_title('solution')

        orig_img = np.reshape(x_train[i, :], (28, 28))
        axarr[2].imshow(orig_img, cmap=plt.get_cmap('gray'), label='original')
        axarr[2].set_title('original')

        # plt.legend()
        plt.savefig(f"images/blur_img_{i}.pdf")




def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte.gz' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte.gz' % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

    return images, labels


def get_mnist(emnist=True):
    orig_cwd = hydra.utils.get_original_cwd()
    if emnist:
        images, labels = extract_training_samples('letters')
        images = images[:20000, :, :]
        x_train = np.reshape(images, (images.shape[0], 784)) / 255
        x_test = None

    else:
        # Load MNIST dataset
        # x_train, y_train = load_mnist('mnist_data', kind='train')
        # x_test, y_test = load_mnist('mnist_data', kind='t10k')
        x_train, y_train = load_mnist(f"{orig_cwd}/examples/mnist_data", kind='train')
        x_test, y_test = load_mnist(f"{orig_cwd}/examples/mnist_data", kind='t10k')

        # Normalize pixel values
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255
    return x_train, x_test




def mnist_canon(A, lambd, blurred_img):
    # create cvxpy prob
    b = np.ravel(blurred_img)
    m_img, n_img = blurred_img.shape
    x = cp.Variable((m_img, n_img))

    # tv = 0
    # for i in range(m_img - 1):
    #     for j in range(n_img - 1):
    #         tv += cp.abs(x[i, j + 1] - x[i, j]) + cp.abs(x[i + 1, j] - x[i, j])

    # prob = cp.Problem(cp.Minimize(lambd * tv + cp.sum_squares(A @ cp.vec(x) - b))) 
    constraints = [x >= 0, x <= 1]
    prob = cp.Problem(cp.Minimize(lambd * cp.sum(x) + cp.sum_squares(A @ cp.vec(x) - b)), constraints)
    data, chain, inverse_data = prob.get_problem_data(cp.OSQP)

    # A = np.vstack([data['A'].todense(), data['F'].todense()])
    # u = np.hstack([data['b'], data['G']])
    # l = np.hstack([data['b'], -np.inf * np.ones(data['G'].size)])
    # P = data['P'].todense()
    # c = data['q']

    prob.solve(solver=cp.OSQP, verbose=True, max_iter=10, adaptive_rho=False, eps_abs=1e0) #, normalize=False,
                    #  scale=1,
                    #  rho_x=1,
                    #  adaptive_scale=False,
                    #  max_iters=50)
                    #  eps_rel=1e-4,
                    #  eps_abs=1e-4)
                    #  max_iters=1000)#, max_iter=50)
    img = np.reshape(x.value, (28,28))

    f, axarr = plt.subplots(1,2)
    axarr[0].imshow(blurred_img, cmap=plt.get_cmap('gray'))
    axarr[1].imshow(img.T, cmap=plt.get_cmap('gray'))
    # plt.show()

    # import pdb
    # pdb.set_trace()

    # get prob data
    # return jnp.array(P), jnp.array(A), jnp.array(c), jnp.array(l), jnp.array(u)
    import pdb
    pdb.set_trace()
    return jnp.array(P), jnp.array(A), jnp.array(c), jnp.array(l), jnp.array(u)

# # Create directory to store MNIST dataset
# if not os.path.exists('mnist_data'):
#     os.makedirs('mnist_data')

# # Download training set images and labels
# urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
#                            'mnist_data/train-images-idx3-ubyte.gz')
# urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
#                            'mnist_data/train-labels-idx1-ubyte.gz')

# # Download test set images and labels
# urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
#                            'mnist_data/t10k-images-idx3-ubyte.gz')
# urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
#                            'mnist_data/t10k-labels-idx1-ubyte.gz')


def blurMatrix(m, width=3):
    # width should be odd
    halflen = int(np.ceil((width-1)/2))
    
    r, c = np.zeros(m), np.zeros(m)
    c[:1+halflen] = 1/width
    r[:1+halflen] = 1/width
    
    return la.toeplitz(c, r)

def vectorized2DBlurMatrix(m, n, width=3):
    # This function returns B corresponding to blurring the columns and rows of an image independently using averaging filters with the same filter width, and
    # represents this process as a matrix operating on the mn vector obtained by vectorizing the image
    Bcols = blurMatrix(m, width) #+ .1 * np.eye(n)
    Brows = blurMatrix(n, width) #+ .1 * np.eye(n)
    return np.kron(Brows, Bcols)


def custom_visualize_fn(z_all, z_stars, z_no_learn, z_nn, thetas, iterates, visual_path, num=20, 
                        quantiles=[.01, .05, .1, .2, .25, .3, .35, .4, .45, .45, .5, .55, .6, .65, .7, .75, .8, .85, .9, .95, .99]):
    """
    assume len(iterates) == 1 for now
        point is to compare no-learning vs learned for after iterates number of steps

    plots
    1. optimal
    2. blurred
    3. no-learning
    4. nearest neighbor
    5. learned

    do this for multiple images
    """
    # assert len(iterates) == 1
    # iter_num = iterates[0]
    # num = x_no_learn.shape[0]
    lim = np.min([num, z_no_learn.shape[0]])
    for i in range(lim):
        # create the folder
        if not os.path.exists(f"{visual_path}/first_few/"):
            os.mkdir(f"{visual_path}/first_few/")
        if not os.path.exists(f"{visual_path}/first_few/blur_img_{i}"):
            os.mkdir(f"{visual_path}/first_few/blur_img_{i}")
            
        for j in range(len(iterates)):
            steps = iterates[j]
            filename = f"{visual_path}/first_few/blur_img_{i}/steps_{steps}.pdf"
            x_star = z_stars[i, :784]
            blurred_img_vec = thetas[i, :784]
            x_no_learn = z_no_learn[i, steps, :784]
            x_nn = z_nn[i, steps, :784]
            x_learn = z_all[i, steps, :784]
            plot_mnist_img(x_star, blurred_img_vec, x_no_learn, x_nn, x_learn, filename)

    # plot the quantiles
    # first get the nn distances, a vector of length (N) 
    lim = z_nn.shape[0]
    # distances = np.linalg.norm(z_stars[:lim, :] - z_nn[:, 0, :], axis=1) / np.linalg.norm(z_nn[:, 0, :], axis=1)
    distances = np.linalg.norm(z_stars[:lim, :] - z_nn[:, 0, :], axis=1)
    mult_percentiles = [.1, .5, .9, .99]
    mult_percentiles_indices = []

    argsort_dist = np.argsort(distances)
    for i in range(len(quantiles)):
        quantile = quantiles[i]
        if not os.path.exists(f"{visual_path}/quantiles/"):
            os.mkdir(f"{visual_path}/quantiles/")
        if not os.path.exists(f"{visual_path}/quantiles/quantile_{quantile}"):
            os.mkdir(f"{visual_path}/quantiles/quantile_{quantile}")

        # find the test index that corresponds to the specific quantile
        quantile_index = int(quantile * distances.size)
        index = argsort_dist[quantile_index]

        if quantile in mult_percentiles:
            mult_percentiles_indices.append(index)

        for j in range(len(iterates)):
            steps = iterates[j]
            filename = f"{visual_path}/quantiles/quantile_{quantile}/steps_{steps}.pdf"
            x_star = z_stars[index, :784]
            blurred_img_vec = thetas[index, :784]
            x_no_learn = z_no_learn[index, steps, :784]
            x_nn = z_nn[index, steps, :784]
            x_learn = z_all[index, steps, :784]
            plot_mnist_img(x_star, blurred_img_vec, x_no_learn, x_nn, x_learn, filename)

    # plot the percentiles for [.1, .5, .9, .99]
    indices = np.array(mult_percentiles_indices)
    if not os.path.exists(f"{visual_path}/quantiles/quantile_mult"):
        os.mkdir(f"{visual_path}/quantiles/quantile_mult")
    for j in range(len(iterates)):
        steps = iterates[j]
        filename = f"{visual_path}/quantiles/quantile_mult/steps_{steps}.pdf"
        x_stars = z_stars[indices, :784]
        blurred_img_vecs = thetas[indices, :784]
        x_no_learns = z_no_learn[indices, steps, :784]
        x_nns = z_nn[indices, steps, :784]
        x_learns = z_all[indices, steps, :784]
        plot_mult_mnist_img(x_stars, blurred_img_vecs, x_no_learns, x_nns, x_learns, filename)


def plot_mult_mnist_img(x_stars, blurred_img_vecs, x_no_learns, x_nns, x_learns, filename):
    num = x_stars.shape[0]
    f, axarr = plt.subplots(num, 5)

    axarr[0, 0].set_title('optimal\n')
    axarr[0, 1].set_title('blurred\n')
    axarr[0, 2].set_title('cold-start\n')
    axarr[0, 3].set_title('nearest \n neighbor')
    axarr[0, 4].set_title('learned\n')

    # import pdb
    # pdb.set_trace()
    for i in range(num):
        # get the clean image (optimal solution) from x_stars
        opt_img = np.reshape(x_stars[i, :], (28,28))
        axarr[i, 0].imshow(opt_img, cmap=plt.get_cmap('gray'))
        axarr[i, 0].axis('off')

        # get the blurred image from theta
        blurred_img = np.reshape(blurred_img_vecs[i, :], (28,28))
        axarr[i, 1].imshow(blurred_img, cmap=plt.get_cmap('gray'))
        axarr[i, 1].axis('off')

        # cold-start
        cold_start_img = np.reshape(x_no_learns[i, :], (28,28))
        axarr[i, 2].imshow(cold_start_img, cmap=plt.get_cmap('gray'))
        axarr[i, 2].axis('off')

        # nearest neighbor
        nearest_neighbor_img = np.reshape(x_nns[i, :], (28,28))
        axarr[i, 3].imshow(nearest_neighbor_img, cmap=plt.get_cmap('gray'))
        axarr[i, 3].axis('off')

        # learned
        learned_img = np.reshape(x_learns[i, :], (28,28))
        axarr[i, 4].imshow(learned_img, cmap=plt.get_cmap('gray'))
        axarr[i, 4].axis('off')

    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.clf()



def plot_mnist_img(x_star, blurred_img_vec, x_no_learn, x_nn, x_learn, filename):
    f, axarr = plt.subplots(1, 5)

    # get the clean image (optimal solution) from x_stars
    opt_img = np.reshape(x_star, (28,28))
    axarr[0].imshow(opt_img, cmap=plt.get_cmap('gray'))
    axarr[0].set_title('optimal\n')
    axarr[0].axis('off')

    # get the blurred image from theta
    blurred_img = np.reshape(blurred_img_vec, (28,28))
    axarr[1].imshow(blurred_img, cmap=plt.get_cmap('gray'))
    axarr[1].set_title('blurred\n')
    axarr[1].axis('off')

    # cold-start
    cold_start_img = np.reshape(x_no_learn, (28,28))
    axarr[2].imshow(cold_start_img, cmap=plt.get_cmap('gray'))
    axarr[2].set_title('cold-start\n')
    axarr[2].axis('off')

    # nearest neighbor
    nearest_neighbor_img = np.reshape(x_nn, (28,28))
    axarr[3].imshow(nearest_neighbor_img, cmap=plt.get_cmap('gray'))
    axarr[3].set_title('nearest \n neighbor')
    axarr[3].axis('off')

    # learned
    learned_img = np.reshape(x_learn, (28,28))
    axarr[4].imshow(learned_img, cmap=plt.get_cmap('gray'))
    axarr[4].set_title('learned\n')
    axarr[4].axis('off')

    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.clf()
