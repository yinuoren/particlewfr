import random
# datsets override methods override generic

#
# datasets
#
adult = dict(
    dataset='adult',
    dim=(88,),
    objectives=['BinaryCrossEntropyLoss', 'deo'],
    epochs=50,
    use_scheduler=False,
    # for fully-connected models
    arch = [60, 25],
    # for cosmos
    lamda=.01,
    alpha=.5,
)

credit = dict(
    dataset='credit',
    dim=(90,),
    objectives=['BinaryCrossEntropyLoss', 'deo'],
    epochs=50,
    use_scheduler=False,
    # for fully-connected models
    arch = [60, 25],
    # for cosmos
    lamda=.01,
    alpha=[.1, .5],
)

compass = dict(
    dataset='compass',
    dim=(20,),
    objectives=['BinaryCrossEntropyLoss', 'deo'],
    epochs=50,
    use_scheduler=False,
    # for fully-connected models
    arch = [60, 25],
    # for cosmos
    lamda=.01,
    alpha=.5,
)

mslr = dict(
    dataset='mslr',
    dim=(131,),
    # objectives=['InnerProductUtility', 'InnerProductUtility'],
    objectives=['ListNetLoss','ListNetLoss','ListNetLoss','ListNetLoss','ListNetLoss','ListNetLoss'],
    epochs=500,
    use_scheduler=False,
    reference_point=[0., 0., 0., 0., 0., 0.],
    # for fully-connected models
    arch = [32],
    # this dataset has multiple labels but outputs only one logit
    mtl=False,
    # for cosmos
    lamda=1.,
    alpha=1.,
    # for argmo
    warm = 1,
    p_lr = 5e-4,
    const = .0001,
    # for score computation
    k_ndcg = 10,
    # for particle
    alpha2 = 100,  # dominance magnitude
    beta = 1e-3,    # repulsion magnitude
    G_type = 'gaussian', # 'gaussian' or 'coulomb' or 'lj' or 'cauchy'
    gamma = 0,   # noise level
    M = 30., # birth-death magnitude
    width = 1e-4, # width for the kernel
)

multi_mnist = dict(
    dataset='multi_mnist',
    dim=(1, 36, 36),
    objectives=['CrossEntropyLoss', 'CrossEntropyLoss'],
    train_eval_every=1,
    # this is an image dataset
    tabular=False,
    # for cosmos
    lamda=2,
    alpha=1.2,
    # for particle
    alpha2 = 5e-2,  # dominance magnitude
    beta = 1e-3,    # repulsion magnitude
    G_type = 'gaussian', # 'gaussian' or 'coulomb' or 'lj' or 'cauchy'
    gamma = 1e-3,   # noise level
    M = 1., # birth-death magnitude
    width = 1., # width for the kernel
    # initialization='zero', # 'zero' or 'random'
)

multi_fashion = dict(
    dataset='multi_fashion',
    dim=(1, 36, 36),
    objectives=['CrossEntropyLoss', 'CrossEntropyLoss'],
    # this is an image dataset
    tabular=False,
    # for cosmos
    lamda=2,
    alpha=1.2,
)

multi_fashion_mnist = dict(
    dataset='multi_fashion_mnist',
    dim=(1, 36, 36),
    objectives=['CrossEntropyLoss', 'CrossEntropyLoss'],
    # this is an image dataset
    tabular=False,
    # for cosmos
    lamda=8,
    alpha=1.2,
)

celeba = dict(
    dataset='celeba',
    dim=(3, 64, 64),
    task_ids=[16, 22],                                    # easy tasks
    # task_ids=[25, 27],                                    # hard tasks
    # task_ids=[16, 22, 24],                                # 3 objectives
    # task_ids=[26, 22, 24, 26],                            # 4 objectives
    # task_ids=[random.randint(0, 39) for _ in range(10)],  # 10 random tasks
    objectives=['BinaryCrossEntropyLoss' for _ in range(10)],
    epochs=30,
    use_scheduler=False,
    train_eval_every=0,     # do it in parallel manually
    eval_every=0,           #
    checkpoint_every=1,
    batch_size=32,
    model_name='efficientnet-b4',   # we also experimented with 'resnet-18', try it.
    lr=5e-4,
    # this is an image dataset
    tabular=False,
    # for cosmos
    n_test_rays=1,
    lamda=3,
    alpha=1,
)

fonseca = dict(
    dataset = 'fonseca',
    dim = (0,),
    arch = [128],
    objectives = ['Fonseca1', 'Fonseca2'],
    epochs = 400,
    use_scheduler = False,
    train_eval_every=10,
    eval_every=0,
    lr = 1e-3,
    #This is an explicit example
    explicit=True,
    param_dim = 100,
    # for cosmos
    lamda = 1,
    alpha = 1,
    # for argmo
    warm = 5,
    # for particle
    alpha2 = 5e-2,  # dominance magnitude
    beta = 1e-3,    # repulsion magnitude
    G_type = 'gaussian', # 'gaussian' or 'coulomb' or 'lj' or 'cauchy'
    gamma = 1e-3,   # noise level
    M = 1., # birth-death magnitude
    width = 1., # width for the kernel
    initialization='zero', # 'zero' or 'random'
)

ZDT3 = dict(
    dataset = 'ZDT3',
    dim = (0,),
    arch = [128, 1024],
    objectives = ['ZDT3_1', 'ZDT3_2'],
    epochs = 3000,
    use_scheduler = False,
    train_eval_every= 50,
    eval_every=0,
    n_test_rays=50,
    lr = 1e-4,
    clip = [0., 1.],
    n_particles = 50,
    #This is an explicit example
    explicit=True,
    param_dim = 30,
    # for cosmos
    lamda = .5,
    alpha = .5,
    # for argmo
    warm = 200,
    p_lr = 0,
    const = 0.1,
    # for particle
    alpha2 = 5e-1,  # dominance magnitude
    beta = 1e-4,    # repulsion magnitude
    G_type = 'coulomb', # 'gaussian' or 'coulomb' or 'lj' or 'cauchy'
    gamma = 1e-4,   # noise level
    M = 10., # birth-death magnitude
    width = 1., # width for the kernel
    initialization = 'random', # 'zero' or 'random'
)

DTLZ7 = dict(
    dataset = 'DTLZ7',
    dim = (0,),
    arch = [128, 1024],
    objectives = ['DTLZ7_1', 'DTLZ7_2', 'DTLZ7_3'],
    epochs = 3000,
    use_scheduler = False,
    train_eval_every= 50,
    eval_every=0,
    n_test_rays=200,
    lr = 1e-4,
    clip = [0., 1.],
    n_particles = 200,
    #This is an explicit example
    explicit=True,
    param_dim = 30,
    # for cosmos
    lamda = .5,
    alpha = [.5, .5, .5],
    # for argmo
    warm = 200,
    p_lr = 0,
    const = 0.1,
    # for particle
    alpha2 = 5e-1,  # dominance magnitude
    beta = 1e-4,    # repulsion magnitude
    G_type = 'coulomb', # 'gaussian' or 'coulomb' or 'lj' or 'cauchy'
    gamma = 1e-4,   # noise level
    M = 10., # birth-death magnitude
    width = 1., # width for the kernel
    initialization = 'random', # 'zero' or 'random'
    reference_point=[1., 1., 10.],
)

#
# methods
#
paretoMTL = dict(
    method='ParetoMTL',
    num_starts=5,
    scheduler_gamma=0.5,
    scheduler_milestones=[15,30,45,60,75,90],
)

cosmos = dict(
    method='cosmos',
    lamda=2,        # Default for multi-mnist
    alpha=1.2,      #
)

mgda = dict(
    method='mgda',
    lr=1e-4,
    approximate_norm_solution=False,
    normalization_type='loss+',
    use_scheduler=False,
)

SingleTaskSolver = dict(
    method='SingleTask',
    num_starts=2,   # two times for two objectives (sequentially)
)

uniform_scaling = dict(
    method='uniform',
)

hyperSolver_ln = dict(
    method='hyper_ln',
    lr=1e-4,
    epochs=150,
    alpha=.2,   # dirichlet sampling
    use_scheduler=False,
    internal_solver='linear', # 'epo' or 'linear'
)

hyperSolver_epo = dict(
    method='hyper_epo',
    lr=1e-4,
    epochs=150,
    alpha=.2,   # dirichlet sampling
    use_scheduler=False,
    internal_solver='epo',
)

argmo_kernel = dict(
    method='argmo',
    n_particles=15,
    p_lr=5e-3, # 5e-3 for kernel-based
    rv_method='kernel',
    warm=1,
    num_rv_adp=1,
    const=1e-2, # width for the kernel
)

argmo_hv = dict(
    method='argmo',
    n_particles=12,
    p_lr=5e-4,
    rv_method='hv',
    warm=1,
    num_rv_adp=1,
    const=1e-2, # width for the kernel
)

particle = dict(
    method='particle',
    n_particles=12,
    normalization_type='loss+',
    alpha2 = 5e-3,  # dominance magnitude
    beta = 1e-4,    # repulsion magnitude
    G_type = 'gaussian', # 'gaussian' or 'coulomb' or 'lj' or 'cauchy'
    gamma = 0.,   # noise level
    M = 1., # birth-death magnitude
    width = 1., # width for the kernel
)


#
# Common settings
#
generic = dict(    
    # Seed.
    seed=1,
    
    # Directory for logging the results
    logdir='results',

    # dataloader worker threads
    num_workers=0,

    # Number of test preference vectors for Pareto front generating methods    
    n_test_rays=12,

    # Evaluation period for val and test sets (0 for no evaluation)
    eval_every=5,

    # Evaluation period for train set (0 for no evaluation)
    train_eval_every=5,

    # Checkpoint period (0 for no checkpoints)
    checkpoint_every=0,

    # Use a multi-step learning rate scheduler with defined gamma and milestones
    use_scheduler=True,
    scheduler_gamma=0.1,
    scheduler_milestones=[20,40,80,90],

    # Number of train rays for methods that follow a training preference (ParetoMTL and MGDA)
    num_starts=1,

    # Training parameters
    lr=1e-3,
    batch_size=512,
    epochs=100,

    # Reference point for hyper-volume calculation
    reference_point=[1., 1.],
    
    # Is it a multi-task learning problem? (Outputting multiple logits) only valid for datasets with multiple labels
    mtl=True,
    
    # Is it a tabular dataset? (Not an image dataset)
    tabular=True,
    
    # Does it involve neural networks?
    explicit=False,
    
    # Need parameters be clipped?
    clip = None,
)
