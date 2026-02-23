import numpy as np
import matplotlib.pyplot as plt
import time

# ===============================================Parameter Setting ============================================================
# for both algorithm
# low = [-10,-10]
# high = [10,10]
low = [-500,-500,-500]
high = [500,500,500]
str_size = 100
random_state=69
iter_tol_max = 30
num_var = 3
# for GA
pop_size = 100
# for SA
init_temp=500.0
final_temp=1e-5
num_sim = 10
alpha=0.8
# ===================================================================================

# def func(x1, x2):
#     v1 = 0
#     v2 = 0
#     for i in range(1,6):
#         v1 = v1 + i * np.cos((i + 1) * x1 + i)
#         v2 = v2 + i * np.cos((i + 1) * x2 + i)
#     v = -v1 * v2
#     return v

def func(x1,x2,x3):
    v1 = x1 * np.sin(np.sqrt(abs(x1))) + x2 * np.sin(np.sqrt(abs(x2))) + x3 * np.sin(np.sqrt(abs(x3)))
    return -(418.9829 * 3 - v1)

def init(m, n):
    return np.random.randint(2, size=(m,n))

def decode(ss, a, b):
    n = ss.shape[1]
    x = []
    for s in ss:
        bin_to_int = np.array([int(j) << i for i,j in enumerate(s[::-1])]).sum()
        int_to_x = a + bin_to_int * (b - a) / (2**n - 1)
        x.append(int_to_x)
    return np.array(x)

# ============================================================= Genetic Algorithm ======================================
def selection(pop, sample_size, fitness):
    m,n = pop.shape
    new_pop = pop.copy()

    for i in range(m):
        rand_id = np.random.choice(m, size=max(1, int(sample_size*m)), replace=False)
        max_id = rand_id[fitness[rand_id].argmax()]
        new_pop[i] = pop[max_id].copy()

    return new_pop

def crossover(pop, pc):
    m,n = pop.shape
    new_pop = pop.copy()

    for i in range(0, m-1, 2):
        if np.random.uniform(0, 1) < pc:
            pos = np.random.randint(0, n-1)
            new_pop[i, pos+1:] = pop[i+1, pos+1:].copy()
            new_pop[i+1, pos+1:] = pop[i, pos+1:].copy()

    return new_pop

def mutation(pop, pm):
    m,n = pop.shape
    new_pop = pop.copy()
    mutation_prob = (np.random.uniform(0, 1, size=(m,n)) < pm).astype(int)
    return (mutation_prob + new_pop) % 2

def print_result(gen_num, pop, fitness, x, num_var):
    m = pop.shape[0]
    print('=' * 68)

    for i in range(m):
        print(f'# {i+1}\t{pop[i]}   fitness: {fitness[i]:0.4f} ',end = '')
        for j in range(num_var):
            print(f'{x[j][i]:0.4f}',end = '')
    print(f'Average fitness: {fitness.mean():0.4f}')
    print('=' * 68)

    print(f'Generation {gen_num}: max fitness {fitness.max():0.5f} at x = (',end = '')
    print(f'{x[0][fitness.argmax()]:0.4f}', end='')
    for j in range(1,num_var):
        print(f',{x[j][fitness.argmax()]:0.4f}', end='')
    print(')\n\n')

def GeneticAlgorithm(func, pop_size, str_size, low, high, ps=0.25, pc=1.0, pm=0.1, max_iter=1000, iter_tol_max = 10, random_state=None, num_var=2):

    np.random.seed(random_state)
    pop = init(pop_size, str_size*num_var)

    x = []
    x_hist = []
    for j in range(num_var):
        start = j * str_size
        end = (j + 1) * str_size
        x.append(decode(pop[:,start:end], low[j], high[j]))
        x_hist.append(np.array([]))
    fitness = func(*x)
    for j in range(num_var):
        x_hist[j] = np.concat([x_hist[j], [x[j][fitness.argmax()]]])

    x_best_pop = x
    f_best_pop = fitness
    pop_best = pop
    best = [fitness.max()]
    best_all = fitness.max()
    print_result(1, pop, fitness, x, num_var)
    i = 0
    i_best = 0
    stag = 0
    while i < max_iter and stag < iter_tol_max:
        pop = selection(pop, ps, fitness)
        pop = crossover(pop, pc)
        pop = mutation(pop, pm)
        x = []
        for j in range(num_var):
            start = j * str_size
            end = (j + 1) * str_size - 1
            x.append(decode(pop[:, start:end], low[j], high[j]))
        fitness = func(*x)
        best_curr = fitness.max()
        best.append(best_curr)
        if best_curr - best_all < 1e-4:
            stag += 1
        else:
            stag = 0
            x_best_pop = x
            f_best_pop = fitness
            pop_best = pop
            i_best = i
            best_all = best_curr

        i += 1
        if i % 5 == 0:
            for j in range(num_var):
                x_hist[j] = np.concat([x_hist[j], [x[j][fitness.argmax()]]])

    print_result(i, pop, fitness, x, num_var)
    print_result(i_best, pop_best, f_best_pop, x_best_pop, num_var)
    if i % 5 != 0:
        for j in range(num_var):
            x_hist[j] = np.concat([x_hist[j], [x[j][fitness.argmax()]]])
    if i == max_iter:
        print(i, 'maximum iteration reached!')
        print('Solution not found. Try increasing max_iter for better result.')
    else:
        print(f'Solution found at iteration {i_best}')

    return f_best_pop, x_best_pop, x_hist, best, i_best, pop_size

def plot_result(func, xs, x_hist, fs, best, i, m, num_var, low, high, time):
    if num_var == 1:
        plt.figure(figsize=(8, 6))
        t = np.linspace(low, high, 100)
        zval = func(t)
        plt.plot(t, zval, color='b')
        plt.scatter(x_hist[0], func(x_hist[0]), color = 'red', s = 20, edgecolors = 'k')

        plt.xlabel('$x$')
        plt.ylabel('$f$')
        plt.title(f'History of the Best Solution(with interval of 5 iterations)\nOptimal Solution of Genetic Algorithm: x={xs[0][fs.argmax()]:.4f}')

    if num_var == 2:
        fig1 = plt.figure(figsize=(8, 6))
        t = np.linspace(low, high, 100)
        xval, yval = np.meshgrid(t, t)
        zval = func(xval,yval)
        ax1 = fig1.add_subplot(111, projection='3d')

        ax1.plot_surface(xval, yval, zval, color='b', rstride=1, cstride=1, alpha=0.5)
        ax1.scatter(x_hist[0],x_hist[1], func(*x_hist), color = 'red', s = 80, edgecolors = 'k')

        ax1.set_xlabel('$x$')
        ax1.set_ylabel('$y$')
        ax1.set_zlabel('$f$')
        ax1.set_title(f'History of the Best Solution(with intervals of 5 iterations)\nOptimal Solution of Genetic Algorithm: ({xs[0][fs.argmax()]:.4f},{xs[1][fs.argmax()]:.4f})')

    plt.figure(figsize=(14, 6))
    x = np.arange(np.size(best)) * m
    plt.plot(x, best, color='c')
    plt.xlim(0)
    plt.xlabel('Iteration x Individuals in Each Iteration')
    plt.ylabel('Best Fitness')
    plt.title(f'Best Fitness in population vs Sum of Number of Solutions(Genetic Algorithm)\nBest Fitness at Iteration {i}: {fs.max():.7f}\nNumber of Individuals: {m}\nSum of Time: {round(time,4)}')

    plt.tight_layout()

# ====================================== Simulated Annealing =========================================================
def SimulatedAnnealing(func, str_size, num_sim, low, high, max_iter=10000, init_temp=100.0, final_temp=1e-5, alpha=0.90, iter_tol_max = 10,random_state=None, num_var=2):
    np.random.seed(random_state)
    iter_tol_max = iter_tol_max * 20
    solu = init(num_sim,str_size*num_var)
    x = []
    x_hist = []
    for j in range(num_var):
        start = j * str_size
        end = (j + 1) * str_size
        x.append(decode(solu[:, start:end], low[j], high[j]))
        x_hist.append(np.array([]))
    fitness = []

    i = 0
    stag = 0
    temp = init_temp*np.ones(num_sim)
    current_solu = solu.copy()
    current_fitness = func(*x)
    for j in range(num_var):
        x_hist[j] = np.concat([x_hist[j], [x[j][current_fitness.argmax()]]])

    i_best = 0
    x_best = []
    for j in range(num_var):
        x_best.append(x[j][current_fitness.argmax()])
    fit_best = current_fitness.max()
    fitness.append(fit_best)

    while i < max_iter and stag < iter_tol_max:

        new_solu = current_solu.copy()
        flip_idx = np.random.randint(0, num_var * str_size - 1, size = int(np.ceil(str_size/5)))
        new_solu[:, flip_idx] = 1 - new_solu[:, flip_idx]

        x_new = []
        for j in range(num_var):
            start = j * str_size
            end = (j + 1) * str_size
            x_new.append(decode(new_solu[:, start:end], low[j], high[j]))
        new_fitness = func(*x_new)

        for k in range(num_sim):
            delta = new_fitness[k] - current_fitness[k]

            if delta > 0 or np.random.random() < np.exp(delta / temp[k]):
                current_solu[k] = new_solu[k]
                current_fitness[k] = new_fitness[k]
                for j in range(num_var):
                    x[j][k] = x_new[j][k]

            temp[k] = max(final_temp, temp[k] * alpha)
        fitness.append(current_fitness.max())
        if current_fitness.max() - fit_best < 1e-5:
            stag += 1
        else:
            stag = 0
            for j in range(num_var):
                x_best[j] = float(round((x[j][current_fitness.argmax()]),5))
            fit_best = current_fitness.max()
            i_best = i
        i += 1
        if i % 50 == 0:
            for j in range(num_var):
                x_hist[j] = np.concat([x_hist[j], [x[j][current_fitness.argmax()]]])

    print('=' * 68)
    print(f'Generation {i_best}: fitness {fit_best:0.5f} at (x,y) = {x_best}\n')

    if i == max_iter:
        print(i, 'maximum iteration reached!')
        print('Solution not found. Try increasing max_iter for better result.')
    else:
        print('Solution found at iteration', i_best)

    return fitness, x_best, x_hist, i_best, num_sim

def plot_result_1(func, xs, x_hist, fitness, i,m, num_var,  low, high, time):

    if num_var == 2:
        fig2 = plt.figure(figsize=(8, 6))
        t = np.linspace(low, high, 100)
        xval, yval = np.meshgrid(t, t)
        zval = func(xval,yval)

        ax1 = fig2.add_subplot(111, projection='3d')
        ax1.plot_surface(xval, yval, zval, color='b', rstride=1, cstride=1, alpha=0.6)

        ax1.scatter(*x_hist, func(*x_hist), color = 'red', s = 80, edgecolors = 'k')

        ax1.set_xlabel('$x$')
        ax1.set_ylabel('$y$')
        ax1.set_zlabel('$f$')
        ax1.set_title(f'History of the Best Solution(with intervals of 50 iterations)\nOptimal Solution of Simulated Annealing: ({xs[0]:.5f},{xs[1]:.5f})')

    if num_var == 1:
        plt.figure(figsize=(8, 6))
        t = np.linspace(low, high, 100)
        zval = func(t)

        plt.plot(t, zval, color='b')
        plt.scatter(*x_hist, func(*x_hist), color = 'red', s = 80, edgecolors = 'k')

        plt.xlabel('$x$')
        plt.ylabel('$f$')
        plt.title(f'History of the Best Solution(with intervals of 50 iterations)\nOptimal Solution of Simulated Annealing: ({xs[0]:.5f})')

    plt.figure(figsize=(14, 6))
    x = np.arange(np.size(fitness))*m
    plt.plot(x,fitness, color='c')
    plt.xlim(0)
    plt.xlabel('Iteration x Individuals in Each Iteration')
    plt.ylabel('Best Fitness')
    plt.title(f'Fitness vs Sum of Number of Solutions(Simulated Annealing)\nOptimal Fitness at Iteration {i}: {max(fitness):.7f}\nSum of Time: {round(time, 4)}')

    plt.tight_layout()

start1 = time.time()
fs, xs, x_hist, best, i, m = GeneticAlgorithm(func, pop_size=pop_size, str_size=str_size, low=low, high=high, iter_tol_max = iter_tol_max, random_state=random_state,num_var = num_var)
end1 = time.time()
plot_result(func, xs, x_hist, fs, best, i, m, num_var = num_var, low=low, high=high, time = end1 - start1)

start2 = time.time()
fitness, xs, x_hist, i, m = SimulatedAnnealing(func, init_temp=init_temp, final_temp=final_temp, alpha=alpha, str_size = str_size, num_sim = num_sim, low=low, high=high, iter_tol_max = iter_tol_max, random_state=random_state,num_var = num_var)
end2 = time.time()
plot_result_1(func, xs, x_hist, fitness, i,m, num_var = num_var, low=low, high=high, time = end2 - start2)

plt.show()
