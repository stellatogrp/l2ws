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
from PIL import Image
from functools import partial
from scipy.spatial import distance_matrix


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

    # # old
    blur_size = setup_cfg['blur_size']

    deblur_or_denoise = setup_cfg['deblur_or_denoise']

    # # get the blur matrix
    dataset = setup_cfg['dataset']
    img_size = setup_cfg['mri_size'] if dataset == 'mri' else 28
    B = vectorized2DBlurMatrix(img_size, img_size, blur_size)

    # # get P, A
    if deblur_or_denoise == 'deblur':
        P = B.T @ B * setup_cfg.get('obj_const', 1)
        m, n = img_size * img_size, img_size * img_size
        A = np.eye(n)
        A = setup_cfg.get('A_scale', 1) * A

        rho_vec, sigma = jnp.ones(m), 1
        # rho_vec = rho_vec.at[l == u].set(1000)
    else:
        # new
        P, A, prob, img_param, num_eq = mnist_canon(B, setup_cfg['lambd'])
        m, n = A.shape
        rho_vec = jnp.ones(m)
        rho_vec = rho_vec.at[:num_eq].set(1000)

    sigma = 1
    M = P + sigma * jnp.eye(n) + A.T @ jnp.diag(rho_vec) @ A
    factor = jsp.linalg.lu_factor(M)

    m, n = A.shape
    static_dict = dict(factor=factor, P=P, A=A, rho=rho_vec)

    # we directly save q now
    static_flag = True
    algo = 'osqp'
    deblur_or_denoise = setup_cfg['deblur_or_denoise']
    
    vis_fn = partial(custom_visualize_fn, figsize=img_size**2, deblur_or_denoise=deblur_or_denoise)
    workspace = Workspace(algo, run_cfg, static_flag, static_dict, example, 
                          custom_visualize_fn=vis_fn)

    # run the workspace
    workspace.run()



def setup_probs(setup_cfg):
    cfg = setup_cfg
    N_train, N_test = cfg.N_train, cfg.N_test
    N = N_train + N_test
    lambd = cfg.lambd
    # emnist = cfg.get('emnist', True)
    dataset = cfg.get('dataset')
    mri_size = setup_cfg['mri_size']
    img_size = mri_size if dataset == 'mri' else 28
    deblur_or_denoise = cfg.get('deblur_or_denoise')

    np.random.seed(cfg.seed)

    # save output to output_filename
    output_filename = f"{os.getcwd()}/data_setup"

    # setup the training
    lambd = setup_cfg['lambd']
    lambd2 = setup_cfg['lambd2']
    blur_size = setup_cfg['blur_size']
    

    # load the mnist images
    # x_train, x_test = get_mnist(emnist=emnist)
    x_train, x_test = get_dataset(dataset, mri_size=mri_size, num=N)

    # distances = distance_matrix(
    #             x_train,
    #             x_train)
    # import pdb
    # pdb.set_trace()

    # get the blur matrix
    B = vectorized2DBlurMatrix(img_size, img_size, blur_size)

    # get P, A
    obj_const = cfg.get('obj_const', 1)
    P = B.T @ B * obj_const
    m, n = img_size * img_size, img_size * img_size
    A = np.eye(n)
    A_scale = cfg.get('A_scale', 1)
    A = A * A_scale

    if deblur_or_denoise == 'denoise':
        # first get P, A
        # the following line is for deblurring and denoising
        # P, A, prob, img_param, num_eq = mnist_canon(B, lambd)
        P, A, prob, img_param, num_eq = mnist_canon(np.eye(784, 784), lambd, lambd2)
        m, n = A.shape

        # create theta_mat
        noise_matrix = cfg['noise_std_dev'] * np.array(np.random.normal(size=(N, 784)))
        # theta_mat = np.clip((B @ x_train[:N, :].T).T + noise_matrix, a_min=0, a_max=1)
        theta_mat = np.clip((B @ x_train[:N, :].T).T + noise_matrix, a_min=0, a_max=1)
        q_mat = get_q_mat(theta_mat, prob, img_param, m, n, B)

    elif deblur_or_denoise == 'deblur':
        q_mat = jnp.zeros((N, 2 * m + n))
        q_mat = q_mat.at[:, m + n:].set(1 * A_scale) #q_mat.at[:, m + n:].set(jnp.inf) #
        theta_mat = jnp.zeros((N, n))

        # noisy_img =  np.reshape(B @ x_train[0, :], (28, 28)) + .00 * np.random.normal(size=(28, 28))
        # mnist_canon(B, lambd, noisy_img)

        # blur img
        # blurred_imgs = []
        for i in range(N):
            noise = cfg['noise_std_dev'] * jnp.array(np.random.normal(size=(img_size, img_size)))
            blurred_img = jnp.reshape(B @ x_train[i, :], (img_size, img_size)) + noise
            # blurred_img = jnp.reshape(x_train[i, :], (28, 28)) + noise
            blurred_img_vec = jnp.ravel(blurred_img)

            q_mat = q_mat.at[i, :n].set((-B.T @ blurred_img_vec + lambd) * obj_const)
            theta_mat = theta_mat.at[i, :].set(blurred_img_vec)
            # blurred_imgs.append(blurred_img)

    z_stars = direct_osqp_setup_script(theta_mat, q_mat, P, A, output_filename, z_stars=None)
    
    if not os.path.exists('images'):
        os.mkdir('images')

    # save blurred images
    for i in range(5):
        blurred_img = np.reshape(theta_mat[i, :], (img_size, img_size))
        f, axarr = plt.subplots(1,3)
        axarr[0].imshow(blurred_img, cmap=plt.get_cmap('gray'), label='blurred')
        axarr[0].set_title('blurred')

        # sol_img = np.reshape(z_stars[i, :784], (28,28))
        # sol_img = np.reshape(z_stars[i, 1458:1458 + 784], (28,28))
        if dataset == 'mri':
            sol_img = np.reshape(z_stars[i, :img_size * img_size], (img_size, img_size))
        else:
            if deblur_or_denoise == 'denoise':
                sol_img = np.reshape(z_stars[i, 1458:1458 + 784], (28,28))
            elif deblur_or_denoise == 'deblur':
                sol_img = np.reshape(z_stars[i, :784], (28,28))
        axarr[1].imshow(sol_img, cmap=plt.get_cmap('gray'), label='solution')
        axarr[1].set_title('solution')

        orig_img = np.reshape(x_train[i, :], (img_size, img_size))
        axarr[2].imshow(orig_img, cmap=plt.get_cmap('gray'), label='original')
        axarr[2].set_title('original')

        # plt.legend()
        plt.savefig(f"images/blur_img_{i}.pdf")


def get_q_mat(img_matrix, prob, img_param, m, n, B):
    """
    change this so that b_matrix, b_param is passed in
        instead of A_tensor, A_param

    I think this should work now
    """
    N = img_matrix.shape[0]
    q_mat = jnp.zeros((N, 2 * m + n))

    # set the parameter
    img_param.value = img_matrix[0, :]

    # get the problem data
    data, _, __ = prob.get_problem_data(cp.OSQP)
    c, b = data['q'], data['b']
    g = data['G']

    q_mat = q_mat.at[:, :n].set(c)
    q_mat = q_mat.at[:, n + 784:n + m].set(-jnp.inf)
    q_mat = q_mat.at[:, n + m + 784:].set(g)

    q_mat = q_mat.at[:, n:n + 784].set((img_matrix.T).T)
    q_mat = q_mat.at[:, n + m:n + m + 784].set((img_matrix.T).T)

    for i in range(N):
        print(i)
        # # set the parameter
        img_param.value = img_matrix[i, :]

        # # get the problem data
        data, _, __ = prob.get_problem_data(cp.OSQP)

        # n = c.size
        c, b = data['q'], data['b']
        g = data['G']
        # import pdb
        # pdb.set_trace()
        
    #     u = np.concatenate([b, g])
    #     l = np.concatenate([b, -np.inf * np.ones(g.size)])

    #     # q_mat = q_mat.at[i, :n].set(c)
    #     q_mat = q_mat.at[i, n:n + m].set(l)
    #     q_mat = q_mat.at[i, n + m:].set(u)
    return q_mat


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


def get_mri(mri_size, num=11000):
    orig_cwd = hydra.utils.get_original_cwd()
    x_test = None
    img_x, img_y = mri_size, mri_size
    folder_path = f"{orig_cwd}/examples/mri_data"
    # folder_path = f"{orig_cwd}/examples/mri_data2"
    files = os.listdir(folder_path)
    N = len(files)
    x_train = np.zeros((N, img_x * img_y))
    # for filename in files:
    # 
    num_files = min(len(files), num)

    for i in range(num_files):
        print(i)
        filename = files[i]
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            img = Image.open(image_path).convert('L')  # Convert to grayscale
            img = img.resize((img_x, img_y))  # Resize to desired dimensions
            
            img_mat = np.array(img)
            # img_cut = img_mat[10:-10, 10:-10]
            
            
            img_vector = img_mat.flatten()
            # image_matrix.append(img_vector)
            x_train[i, :] = img_vector / 255
            # import pdb
            # pdb.set_trace()
    
    # x_train = np.array(image_matrix)
    return x_train, x_test


def get_dataset(dataset, mri_size=50, num=11000):
    if dataset == 'mnist':
        return get_mnist(emnist=False)
    elif dataset == 'emnist':
        return get_mnist(emnist=True)
    elif dataset == 'mri':
        return get_mri(mri_size, num=num)


def tv(value, *args):
    from cvxpy.atoms.affine.reshape import reshape
    from cvxpy.atoms.affine.sum import sum
    from cvxpy.atoms.affine.vstack import vstack
    from cvxpy.atoms.norm import norm
    from cvxpy.expressions.expression import Expression
    """Total variation of a vector, matrix, or list of matrices.

    Uses L1 norm of discrete gradients for vectors and
    L2 norm of discrete gradients for matrices.

    Parameters
    ----------
    value : Expression or numeric constant
        The value to take the total variation of.
    args : Matrix constants/expressions
        Additional matrices extending the third dimension of value.

    Returns
    -------
    Expression
        An Expression representing the total variation.
    """
    value = Expression.cast_to_const(value)
    if value.ndim == 0:
        raise ValueError("tv cannot take a scalar argument.")
    # L1 norm for vectors.
    elif value.ndim == 1:
        return norm(value[1:] - value[0:value.shape[0]-1], 1)
    # L2 norm for matrices.
    else:
        rows, cols = value.shape
        args = map(Expression.cast_to_const, args)
        values = [value] + list(args)
        diffs = []
        for mat in values:
            diffs += [
                mat[0:rows-1, 1:cols] - mat[0:rows-1, 0:cols-1],
                mat[1:rows, 0:cols-1] - mat[0:rows-1, 0:cols-1],
            ]
        length = diffs[0].shape[0]*diffs[1].shape[1]
        stacked = vstack([reshape(diff, (1, length)) for diff in diffs])
        return sum(norm(stacked, p=1, axis=0))


def mnist_canon(A, lambd, lambd2=0):#, blurred_img):
    # create cvxpy prob
    # b = np.ravel(blurred_img)
    img_param = cp.Parameter(784)
    # m_img, n_img = blurred_img.shape
    m_img, n_img = 28, 28
    x = cp.Variable((m_img, n_img))

    # tv = 0
    # for i in range(m_img - 1):
    #     for j in range(n_img - 1):
    #         tv += cp.abs(x[i, j + 1] - x[i, j]) + cp.abs(x[i + 1, j] - x[i, j])

    # prob = cp.Problem(cp.Minimize(lambd * tv + cp.sum_squares(A @ cp.vec(x) - b))) 
    constraints = [x >= 0, x <= 1]
    # prob = cp.Problem(cp.Minimize(lambd * cp.sum(x) + cp.sum_squares(A @ cp.vec(x) - b)), constraints)
    # import pdb
    # pdb.set_trace
    prob = cp.Problem(cp.Minimize(lambd * tv(x) + lambd2 * cp.sum(x) + cp.sum_squares(A @ cp.vec(x) - img_param)), constraints)
    # data, chain, inverse_data = prob.get_problem_data(cp.OSQP)
    # import pdb
    # pdb.set_trace()
    img_param.value = 3.3 * np.random.rand(784)
    soln = prob.solve()
    data, chain, inverse_data = prob.get_problem_data(cp.OSQP)
    soln = chain.solve_via_data(prob, data)
    # unpacks the solution returned by SCS into `problem`
    prob.unpack_results(soln, chain, inverse_data)

    Amat = np.vstack([data['A'].todense(), data['F'].todense()])
    b = data['b']
    num_eq = b.size
    # x_indices = [i for i, var in enumerate(prob.variables()) if id(var) == id(x)]
    # import pdb
    # pdb.set_trace()

    return data['P'].todense(), np.array(Amat), prob, img_param, num_eq
    

    # A = np.vstack([data['A'].todense(), data['F'].todense()])
    # u = np.hstack([data['b'], data['G']])
    # l = np.hstack([data['b'], -np.inf * np.ones(data['G'].size)])
    # P = data['P'].todense()
    # c = data['q']

    prob.solve(solver=cp.OSQP, verbose=True)#, adaptive_rho=False, eps_abs=1e0) 
    # prob.solve(solver=cp.SCS,
    #            normalize=False,
    #                  scale=1,
    #                  rho_x=1,
    #                  adaptive_scale=False,
    #                  max_iters=500,
    #                  verbose=True)
                    #  eps_rel=1e-4,
                    #  eps_abs=1e-4)
                    #  max_iters=1000)#, max_iter=50)
    img = np.reshape(x.value, (28,28))

    f, axarr = plt.subplots(1,2)
    axarr[0].imshow(blurred_img, cmap=plt.get_cmap('gray'))
    axarr[1].imshow(img.T, cmap=plt.get_cmap('gray'))
    plt.show()

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


def custom_visualize_fn(z_all, z_stars, z_no_learn, z_nn, thetas, iterates, visual_path, 
                        figsize=784,
                        num=20, 
                        deblur_or_denoise='deblur',
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
            if deblur_or_denoise == 'deblur':
                x_star = z_stars[i, :figsize]
                blurred_img_vec = thetas[i, :figsize]
                x_no_learn = z_no_learn[i, steps, :figsize]
                x_nn = z_nn[i, steps, :figsize]
                x_learn = z_all[i, steps, :figsize]
            elif deblur_or_denoise == 'denoise':
                base = 1458
                x_star = z_stars[i, base:base + figsize]
                blurred_img_vec = thetas[i,  :figsize]
                x_no_learn = z_no_learn[i, steps,  base:base + figsize]
                x_nn = z_nn[i, steps,  base:base + figsize]
                x_learn = z_all[i, steps,  base:base + figsize]
            plot_mnist_img(x_star, blurred_img_vec, x_no_learn, x_nn, x_learn, filename, figsize=figsize)

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
            x_star = z_stars[index, :figsize]
            blurred_img_vec = thetas[index, :figsize]
            x_no_learn = z_no_learn[index, steps, :figsize]
            x_nn = z_nn[index, steps, :figsize]
            x_learn = z_all[index, steps, :figsize]
            plot_mnist_img(x_star, blurred_img_vec, x_no_learn, x_nn, x_learn, filename, figsize=figsize)

    # plot the percentiles for [.1, .5, .9, .99]
    indices = np.array(mult_percentiles_indices)
    if not os.path.exists(f"{visual_path}/quantiles/quantile_mult"):
        os.mkdir(f"{visual_path}/quantiles/quantile_mult")
    for j in range(len(iterates)):
        steps = iterates[j]
        filename = f"{visual_path}/quantiles/quantile_mult/steps_{steps}.pdf"
        x_stars = z_stars[indices, :figsize]
        blurred_img_vecs = thetas[indices, :figsize]
        x_no_learns = z_no_learn[indices, steps, :figsize]
        x_nns = z_nn[indices, steps, :figsize]
        x_learns = z_all[indices, steps, :figsize]
        plot_mult_mnist_img(x_stars, blurred_img_vecs, x_no_learns, x_nns, x_learns, filename, figsize=figsize)


def plot_mult_mnist_img(x_stars, blurred_img_vecs, x_no_learns, x_nns, x_learns, filename, figsize=784):
    num = x_stars.shape[0]
    f, axarr = plt.subplots(num, 5 + 1)

    start = 1
    axarr[0, 0].set_title('percentile\n')
    axarr[0, 0 + start].set_title('optimal\n')
    axarr[0, 1 + start].set_title('blurred\n')
    axarr[0, 2 + start].set_title('cold-start\n')
    axarr[0, 3 + start].set_title('nearest \n neighbor')
    axarr[0, 4 + start].set_title('learned\n')

    axarr[0, 0].text(.5, .5, r'$10^{\rm{th}}$', ha='center', va='center')#, rotation=0) #, size='large')
    axarr[1, 0].text(.5, .5, r'$50^{\rm{th}}$', ha='center', va='center') #, rotation=0)
    axarr[2, 0].text(.5, .5, r'$90^{\rm{th}}$', ha='center', va='center') #, rotation=0)
    axarr[3, 0].text(.5, .5, r'$99^{\rm{th}}$', ha='center', va='center') #, rotation=0)

    import matplotlib.ticker as ticker

    # import pdb
    # pdb.set_trace()
    img_size = int(np.sqrt(figsize))
    for i in range(num):
        # get the clean image (optimal solution) from x_stars
        opt_img = np.reshape(x_stars[i, :], (img_size, img_size))
        axarr[i, start + 0].imshow(opt_img, cmap=plt.get_cmap('gray'))
        axarr[i, 0].axis('off')
        # axarr[i, 0].tick_params(axis='both', which='both', length=0)
        # axarr[i, 0].axes.xaxis.set_ticklabels([])
        # axarr[i, 0].axes.yaxis.set_ticklabels([])
        # axarr[i, 0].xticks('off')
        # axarr[i, 0].yticks('off')

        # get the blurred image from theta
        blurred_img = np.reshape(blurred_img_vecs[i, :], (img_size, img_size))
        axarr[i, start + 1].imshow(blurred_img, cmap=plt.get_cmap('gray'))
        axarr[i, 1].axis('off')

        # cold-start
        cold_start_img = np.reshape(x_no_learns[i, :], (img_size, img_size))
        axarr[i, start + 2].imshow(cold_start_img, cmap=plt.get_cmap('gray'))
        axarr[i, 2].axis('off')

        # nearest neighbor
        nearest_neighbor_img = np.reshape(x_nns[i, :], (img_size, img_size))
        axarr[i, start + 3].imshow(nearest_neighbor_img, cmap=plt.get_cmap('gray'))
        axarr[i, 3].axis('off')

        # learned
        learned_img = np.reshape(x_learns[i, :], (img_size, img_size))
        axarr[i, start + 4].imshow(learned_img, cmap=plt.get_cmap('gray'))
        axarr[i, 4].axis('off')

        axarr[i, 5].axis('off')

    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.clf()



def plot_mnist_img(x_star, blurred_img_vec, x_no_learn, x_nn, x_learn, filename, figsize=784):
    img_size = int(np.sqrt(figsize))

    f, axarr = plt.subplots(1, 5)

    # get the clean image (optimal solution) from x_stars
    opt_img = np.reshape(x_star, (img_size, img_size))
    axarr[0].imshow(opt_img, cmap=plt.get_cmap('gray'))
    axarr[0].set_title('optimal\n')
    axarr[0].axis('off')

    # get the blurred image from theta
    blurred_img = np.reshape(blurred_img_vec, (img_size, img_size))
    axarr[1].imshow(blurred_img, cmap=plt.get_cmap('gray'))
    axarr[1].set_title('blurred\n')
    axarr[1].axis('off')

    # cold-start
    cold_start_img = np.reshape(x_no_learn, (img_size, img_size))
    axarr[2].imshow(cold_start_img, cmap=plt.get_cmap('gray'))
    axarr[2].set_title('cold-start\n')
    axarr[2].axis('off')

    # nearest neighbor
    nearest_neighbor_img = np.reshape(x_nn, (img_size, img_size))
    axarr[3].imshow(nearest_neighbor_img, cmap=plt.get_cmap('gray'))
    axarr[3].set_title('nearest \n neighbor')
    axarr[3].axis('off')

    # learned
    learned_img = np.reshape(x_learn, (img_size, img_size))
    axarr[4].imshow(learned_img, cmap=plt.get_cmap('gray'))
    axarr[4].set_title('learned\n')
    axarr[4].axis('off')

    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.clf()
