import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime


def init(m, n):
    return np.random.randint(2, size=(m, n))


def decode(ss, a, b):
    n = ss.shape[1]
    x = []
    for s in ss:
        bin_to_int = np.array([int(j) << i for i, j in enumerate(s[::-1])]).sum()
        int_to_x = a + bin_to_int * (b - a) / (2 ** n - 1)
        x.append(int_to_x)
    return np.array(x)


def selection(pop, sample_size, fitness):
    m, n = pop.shape
    new_pop = pop.copy()
    for i in range(m):
        rand_id = np.random.choice(m, size=max(1, int(sample_size * m)), replace=False)
        max_id = rand_id[fitness[rand_id].argmax()]
        new_pop[i] = pop[max_id].copy()
    return new_pop


def crossover(pop, pc):
    m, n = pop.shape
    new_pop = pop.copy()
    for i in range(0, m - 1, 2):
        if np.random.uniform(0, 1) < pc:
            pos = np.random.randint(0, n - 1)
            new_pop[i, pos + 1:] = pop[i + 1, pos + 1:].copy()
            new_pop[i + 1, pos + 1:] = pop[i, pos + 1:].copy()
    return new_pop


def mutation(pop, pm):
    m, n = pop.shape
    new_pop = pop.copy()
    mutation_prob = (np.random.uniform(0, 1, size=(m, n)) < pm).astype(int)
    return (mutation_prob + new_pop) % 2


def GeneticAlgorithm(func, pop_size, str_size, low, high, ps=0.25, pc=1.0, pm=0.1, max_iter=1000, iter_tol_max=10,
                     random_state=None, num_var=2):
    np.random.seed(random_state)
    pop = init(pop_size, str_size * num_var)

    x = []
    x_hist = []
    for j in range(num_var):
        start = j * str_size
        end = (j + 1) * str_size
        x.append(decode(pop[:, start:end], low[j], high[j]))
        x_hist.append(np.array([]))

    fitness = func(*x)
    for j in range(num_var):
        x_hist[j] = np.concat([x_hist[j], [x[j][fitness.argmax()]]])

    x_best_pop = x
    f_best_pop = fitness
    pop_best = pop
    best = [fitness.max()]
    best_all = fitness.max()

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

    if i % 5 != 0:
        for j in range(num_var):
            x_hist[j] = np.concat([x_hist[j], [x[j][fitness.argmax()]]])

    return f_best_pop, x_best_pop, x_hist, best, i_best, pop_size


def SimulatedAnnealing(func, str_size, num_sim, low, high, max_iter=10000, init_temp=100.0, final_temp=1e-5, alpha=0.90,
                       iter_tol_max=10, random_state=None, num_var=2):
    np.random.seed(random_state)
    iter_tol_max = iter_tol_max * 20
    solu = init(num_sim, str_size * num_var)
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
    temp = init_temp * np.ones(num_sim)
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
        flip_idx = np.random.randint(0, num_var * str_size - 1, size=int(np.ceil(str_size / 5)))
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
                x_best[j] = float(round((x[j][current_fitness.argmax()]), 5))
            fit_best = current_fitness.max()
            i_best = i
        i += 1
        if i % 50 == 0:
            for j in range(num_var):
                x_hist[j] = np.concat([x_hist[j], [x[j][current_fitness.argmax()]]])

    return fitness, x_best, x_hist, i_best, num_sim


def plot_result(func, xs, x_hist, fs, best, i, m, num_var, low, high, elapsed_val, base_save_path):
    fig1 = plt.figure(figsize=(8, 6))

    l_val = low[0] if isinstance(low, list) else low
    h_val = high[0] if isinstance(high, list) else high

    if num_var == 1:
        t = np.linspace(l_val, h_val, 100)
        zval = func(t)
        plt.plot(t, zval, color='b', rstride=1, cstride=1, alpha=0.3)
        plt.scatter(x_hist[0], func(x_hist[0]), color='red', s=80, edgecolors='k', linewidths=1.5)
        plt.xlabel('$x$')
        plt.ylabel('$f$')
        plt.title(
            f'History of the Best Solution(with interval of 5 iterations)\nOptimal Solution of Genetic Algorithm: x={xs[0][fs.argmax()]:.4f}')

    elif num_var == 2:
        t = np.linspace(l_val, h_val, 100)
        xval, yval = np.meshgrid(t, t)
        zval = func(xval, yval)
        ax1 = fig1.add_subplot(111, projection='3d')
        ax1.plot_surface(xval, yval, zval, color='b', rstride=1, cstride=1, alpha=0.5)
        ax1.scatter(x_hist[0], x_hist[1], func(*x_hist), color='red', s=80, edgecolors='k', linewidths=1.5)
        ax1.set_xlabel('$x$')
        ax1.set_ylabel('$y$')
        ax1.set_zlabel('$f$')
        ax1.set_title(
            f'History of the Best Solution(with intervals of 5 iterations)\nOptimal Solution of Genetic Algorithm: ({xs[0][fs.argmax()]:.4f},{xs[1][fs.argmax()]:.4f})')
    else:
        plt.text(0.5, 0.5, f'Trajectory visualization only supported for 1 or 2 variables.\nNum vars: {num_var}',
                 horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
        plt.title(f'History of the Best Solution\nOptimal Solution of Genetic Algorithm')

    path_1 = f"{base_save_path}_1.png"
    plt.tight_layout()
    plt.savefig(path_1, dpi=150)
    plt.close(fig1)

    fig2 = plt.figure(figsize=(14, 6))
    x = np.arange(np.size(best)) * m
    plt.plot(x, best, color='c')
    plt.xlim(0)
    plt.xlabel('Iteration x Individuals in Each Iteration')
    plt.ylabel('Best Fitness')
    plt.title(
        f'Best Fitness in population vs Sum of Number of Solutions(Genetic Algorithm)\nBest Fitness at Iteration {i}: {fs.max():.7f}\nNumber of Individuals: {m}\nSum of Time: {round(elapsed_val, 4)}')

    path_2 = f"{base_save_path}_2.png"
    plt.tight_layout()
    plt.savefig(path_2, dpi=150)
    plt.close(fig2)

    return [path_1, path_2]


def plot_result_1(func, xs, x_hist, fitness, i, m, num_var, low, high, elapsed_val, base_save_path):
    fig1 = plt.figure(figsize=(8, 6))

    l_val = low[0] if isinstance(low, list) else low
    h_val = high[0] if isinstance(high, list) else high

    if num_var == 2:
        t = np.linspace(l_val, h_val, 100)
        xval, yval = np.meshgrid(t, t)
        zval = func(xval, yval)
        ax1 = fig1.add_subplot(111, projection='3d')
        ax1.plot_surface(xval, yval, zval, color='b', rstride=1, cstride=1, alpha=0.6)
        ax1.scatter(*x_hist, func(*x_hist), color='red', s=80, edgecolors='k', linewidths=1.5)
        ax1.set_xlabel('$x$')
        ax1.set_ylabel('$y$')
        ax1.set_zlabel('$f$')
        ax1.set_title(
            f'History of the Best Solution(with intervals of 5 iterations)\nOptimal Solution of Simulated Annealing: ({xs[0]:.5f},{xs[1]:.5f})')

    elif num_var == 1:
        t = np.linspace(l_val, h_val, 100)
        zval = func(t)
        xval = t
        plt.plot(xval, zval, color='b', rstride=1, cstride=1, alpha=0.6)
        plt.scatter(*x_hist, func(*x_hist), color='red', s=80, edgecolors='k', linewidths=1.5)
        plt.xlabel('$x$')
        plt.ylabel('$f$')
        plt.title(
            f'History of the Best Solution(with intervals of 5 iterations)\nOptimal Solution of Simulated Annealing: ({xs[0]:.5f})')
    else:
        plt.text(0.5, 0.5, f'Trajectory visualization only supported for 1 or 2 variables.\nNum vars: {num_var}',
                 horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
        plt.title(f'History of the Best Solution\nOptimal Solution of Simulated Annealing')

    path_1 = f"{base_save_path}_1.png"
    plt.tight_layout()
    plt.savefig(path_1, dpi=150)
    plt.close(fig1)

    fig2 = plt.figure(figsize=(14, 6))
    x = np.arange(np.size(fitness)) * m
    plt.plot(x, fitness, color='c')
    plt.xlim(0)
    plt.xlabel('Iteration x Individuals in Each Iteration')
    plt.ylabel('Best Fitness')
    plt.title(
        f'Fitness vs Sum of Number of Solutions(Simulated Annealing)\nOptimal Fitness at Iteration {i}: {max(fitness):.7f}\nSum of Time: {round(elapsed_val, 4)}')

    path_2 = f"{base_save_path}_2.png"
    plt.tight_layout()
    plt.savefig(path_2, dpi=150)
    plt.close(fig2)

    return [path_1, path_2]


class OptimizationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Function Minimization Solver (GA / SA)")
        self.root.geometry("1000x900")

        self.save_dir = "optimization_results"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        notebook = ttk.Notebook(root)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)

        config_frame = tk.Frame(notebook)
        result_frame = tk.Frame(notebook)

        notebook.add(config_frame, text="Algorithm Configuration")
        notebook.add(result_frame, text="Results")

        self.result_text = scrolledtext.ScrolledText(result_frame, font=("Consolas", 10))
        self.result_text.pack(fill="both", expand=True, padx=10, pady=10)

        # Create Scrollable Frame for Configuration
        canvas = tk.Canvas(config_frame)
        scrollbar = ttk.Scrollbar(config_frame, orient="vertical", command=canvas.yview)
        self.scrollable_frame = tk.Frame(canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Bind mouse wheel to scroll
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # --- Content inside Scrollable Frame ---

        input_frame = tk.LabelFrame(self.scrollable_frame, text="Objective Function Settings", padx=10, pady=10)
        input_frame.pack(fill="x", padx=10, pady=5)

        tk.Label(input_frame, text="Function expression f(x1, x2, x3...):").grid(row=0, column=0, sticky="w")
        self.expr_entry = tk.Entry(input_frame, width=70)
        self.expr_entry.grid(row=0, column=1, padx=5, pady=5, columnspan=3)
        self.expr_entry.insert(0, "x1**2 + x2**2")

        tk.Label(input_frame, text="Hint: Click Σ/Π to insert sum/prod with range. Edit as needed.",
                 font=("Arial", 8), fg="gray").grid(row=1, column=1, sticky="w", columnspan=3)

        func_btn_frame = tk.Frame(input_frame)
        func_btn_frame.grid(row=2, column=1, columnspan=3, pady=5, sticky="w")

        func_buttons = [
            ("sin(", "sin("), ("cos(", "cos("), ("tan(", "tan("),
            ("exp(", "exp("), ("log(", "log("), ("log10(", "log10("),
            ("sqrt(", "sqrt("), ("abs(", "abs("), ("pow(", "pow("),
            ("**2", "**2"), ("**3", "**3"), ("**0.5", "**0.5"),
            ("pi", "pi"), ("e", "e"), ("(", "("), (")", ")"),
            ("+", "+"), ("-", "-"), ("*", "*"), ("/", "/"),
            (",", ",")
        ]

        for idx, (label, value) in enumerate(func_buttons):
            row = idx // 8
            col = idx % 8
            btn = tk.Button(func_btn_frame, text=label, width=5,
                            command=lambda v=value: self.insert_text(v))
            btn.grid(row=row, column=col, padx=2, pady=2)

        sum_btn = tk.Button(func_btn_frame, text="Σ", width=5, bg="#FFCDD2", font=("Arial", 11, "bold"),
                            command=self.insert_sum)
        sum_btn.grid(row=3, column=0, padx=2, pady=5)

        prod_btn = tk.Button(func_btn_frame, text="Π", width=5, bg="#FFCDD2", font=("Arial", 11, "bold"),
                             command=self.insert_prod)
        prod_btn.grid(row=3, column=1, padx=2, pady=5)

        var_frame = tk.LabelFrame(self.scrollable_frame, text="Variable Bounds", padx=10, pady=10)
        var_frame.pack(fill="x", padx=10, pady=5)

        top_ctrl = tk.Frame(var_frame)
        top_ctrl.pack(fill="x", pady=(0, 10))
        tk.Label(top_ctrl, text="Number of variables:").pack(side=tk.LEFT)
        self.var_count = tk.IntVar(value=2)
        var_spin = tk.Spinbox(top_ctrl, from_=1, to=10, textvariable=self.var_count, width=5, command=self.update_vars)
        var_spin.pack(side=tk.LEFT, padx=5)
        tk.Button(top_ctrl, text="Update Inputs", command=self.update_vars).pack(side=tk.LEFT, padx=10)

        self.vars_container = tk.Frame(var_frame)
        self.vars_container.pack(fill="both", expand=True)
        self.entries = []
        self.var_buttons = []
        self.update_vars()

        solver_frame = tk.LabelFrame(self.scrollable_frame, text="Select Solver & Parameters", padx=10, pady=10)
        solver_frame.pack(fill="x", padx=10, pady=5)

        self.solver = tk.StringVar(value="GA")
        tk.Radiobutton(solver_frame, text="Genetic Algorithm (GA)", variable=self.solver,
                       value="GA", command=self.toggle_params).pack(side=tk.LEFT, padx=20)
        tk.Radiobutton(solver_frame, text="Simulated Annealing (SA)", variable=self.solver,
                       value="SA", command=self.toggle_params).pack(side=tk.LEFT, padx=20)

        self.param_frame = tk.LabelFrame(self.scrollable_frame, text="Algorithm Parameters", padx=10, pady=10)
        self.param_frame.pack(fill="x", padx=10, pady=5)

        self.param_widgets = {}
        self.create_param_inputs()

        action_frame = tk.Frame(self.scrollable_frame, pady=10)
        action_frame.pack()

        run_btn = tk.Button(action_frame, text="Run Minimization & Plot", command=self.run_optimization,
                            bg="#4CAF50", fg="white", font=("Arial", 12, "bold"), padx=20, pady=5)
        run_btn.pack()

        self.toggle_params()

    def create_param_inputs(self):
        for widget in self.param_frame.winfo_children():
            widget.destroy()
        self.param_widgets = {}

        ga_params = [
            ("pop_size", "Population Size", 100),
            ("str_size", "String Length", 100),
            ("max_iter", "Max Iterations", 1000),
            ("pc", "Crossover Rate", 1.0),
            ("pm", "Mutation Rate", 0.1),
            ("ps", "Selection Pressure", 0.25),
            ("iter_tol_max", "Stagnation Tolerance", 30)
        ]

        sa_params = [
            ("num_sim", "Number of Solutions", 10),
            ("str_size", "String Length", 100),
            ("max_iter", "Max Iterations", 10000),
            ("init_temp", "Initial Temperature", 500.0),
            ("final_temp", "Final Temperature", 1e-5),
            ("alpha", "Cooling Rate (Alpha)", 0.8),
            ("iter_tol_max", "Stagnation Tolerance", 30)
        ]

        params = ga_params if self.solver.get() == "GA" else sa_params

        cols = 3
        for idx, (key, label, default) in enumerate(params):
            row = idx // cols
            col = (idx % cols) * 4

            tk.Label(self.param_frame, text=f"{label}:").grid(row=row, column=col, sticky="e", padx=5, pady=5)
            entry = tk.Entry(self.param_frame, width=10)
            entry.insert(0, str(default))
            entry.grid(row=row, column=col + 1, padx=5, pady=5)
            self.param_widgets[key] = entry

    def toggle_params(self):
        self.create_param_inputs()

    def insert_text(self, text):
        self.expr_entry.insert(tk.INSERT, text)
        self.expr_entry.focus()

    def insert_sum(self):
        n = self.var_count.get()
        sum_expr = f"sum(i for i in range(1, {n + 1}))"
        self.expr_entry.insert(tk.INSERT, sum_expr)
        self.expr_entry.focus()

    def insert_prod(self):
        n = self.var_count.get()
        prod_expr = f"prod(i for i in range(1, {n + 1}))"
        self.expr_entry.insert(tk.INSERT, prod_expr)
        self.expr_entry.focus()

    def update_vars(self):
        for widget in self.vars_container.winfo_children():
            widget.destroy()
        self.entries.clear()
        self.var_buttons = []

        n = self.var_count.get()

        bounds_frame = tk.LabelFrame(self.vars_container, text="Bounds Settings", padx=5, pady=5)
        bounds_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=5, pady=5)

        cols = 2
        for i in range(n):
            row = i // cols
            col = (i % cols) * 4

            tk.Label(bounds_frame, text=f"x{i + 1}: [", font=("Arial", 10, "bold")).grid(row=row, column=col,
                                                                                         sticky="e")

            low_e = tk.Entry(bounds_frame, width=8)
            low_e.insert(0, "-500")
            low_e.grid(row=row, column=col + 1, padx=2)

            tk.Label(bounds_frame, text=",").grid(row=row, column=col + 2)

            high_e = tk.Entry(bounds_frame, width=8)
            high_e.insert(0, "500")
            high_e.grid(row=row, column=col + 3, padx=2)

            tk.Label(bounds_frame, text="]").grid(row=row, column=col + 4, sticky="w")

            self.entries.append((low_e, high_e))

        var_btn_frame = tk.LabelFrame(self.vars_container, text="Variable Buttons (click to insert)", padx=5, pady=5)
        var_btn_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=5, pady=5)

        for i in range(n):
            btn = tk.Button(var_btn_frame, text=f"x{i + 1}", width=6, height=1,
                            command=lambda v=f"x{i + 1}": self.insert_text(v),
                            bg="#E3F2FD", font=("Arial", 10, "bold"))
            btn.pack(side=tk.LEFT, padx=3, pady=3)
            self.var_buttons.append(btn)

        self.vars_container.columnconfigure(0, weight=1)
        self.vars_container.columnconfigure(1, weight=1)

    def parse_function(self, expr, num_vars):
        var_names = [f"x{i + 1}" for i in range(num_vars)]
        safe_dict = {
            "__builtins__": None,
            "np": np,
            "sin": np.sin, "cos": np.cos, "tan": np.tan,
            "exp": np.exp, "log": np.log, "log10": np.log10,
            "sqrt": np.sqrt, "abs": abs, "pow": pow,
            "pi": np.pi, "e": np.e,
            "sum": sum,
            "prod": np.prod,
            "range": range,
        }
        for name in var_names:
            safe_dict[name] = 0

        def func(*args):
            local_vars = {var_names[i]: args[i] for i in range(num_vars)}
            try:
                clean_expr = expr.replace("Σ", ",").replace("Π", ",")
                result = eval(clean_expr, safe_dict, local_vars)
                return np.array(result, dtype=float)
            except Exception as e:
                raise ValueError(f"Formula evaluation error: {e}")

        return func

    def get_params(self):
        params = {}
        for key, entry in self.param_widgets.items():
            try:
                val = entry.get()
                if '.' in val or 'e' in val.lower():
                    params[key] = float(val)
                else:
                    params[key] = int(val)
            except ValueError:
                params[key] = 0
        return params

    def run_optimization(self):
        try:
            expr = self.expr_entry.get().strip()
            if not expr:
                messagebox.showerror("Error", "Please enter an objective function expression!")
                return

            num_vars = self.var_count.get()
            low_bounds = []
            high_bounds = []
            var_names = [f"x{i + 1}" for i in range(num_vars)]

            for i, (low_e, high_e) in enumerate(self.entries):
                try:
                    low = float(low_e.get())
                    high = float(high_e.get())
                except ValueError:
                    messagebox.showerror("Error", f"Bounds for variable x{i + 1} must be numbers.")
                    return
                low_bounds.append(low)
                high_bounds.append(high)

            raw_func = self.parse_function(expr, num_vars)

            def fitness_func(*x):
                try:
                    val = raw_func(*x)
                    return -val
                except Exception:
                    return -1e9 * np.ones_like(x[0])

            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END,
                                    f"Running {self.solver.get()} solver (converting minimization to maximization)...\n")
            self.root.update()

            solver_name = self.solver.get()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"{self.save_dir}/{solver_name}_result_{timestamp}"

            start_time = time.time()
            saved_files = []
            params = self.get_params()
            elapsed_time = 0.0

            if solver_name == "GA":
                fs, xs, x_hist, best, i, m = GeneticAlgorithm(
                    fitness_func,
                    pop_size=params.get('pop_size', 100),
                    str_size=params.get('str_size', 100),
                    low=low_bounds,
                    high=high_bounds,
                    ps=params.get('ps', 0.25),
                    pc=params.get('pc', 1.0),
                    pm=params.get('pm', 0.1),
                    max_iter=params.get('max_iter', 1000),
                    iter_tol_max=params.get('iter_tol_max', 30),
                    random_state=None,
                    num_var=num_vars
                )
                end_time = time.time()
                elapsed_time = end_time - start_time

                min_val = -fs.max()
                best_vars = [float(xs[j][fs.argmax()]) for j in range(num_vars)]
                saved_files = plot_result(raw_func, xs, x_hist, fs, best, i, m, num_vars, low_bounds, high_bounds,
                                          elapsed_time, base_filename)

            elif solver_name == "SA":
                fitness, xs, x_hist, i, m = SimulatedAnnealing(
                    fitness_func,
                    str_size=params.get('str_size', 100),
                    num_sim=params.get('num_sim', 10),
                    low=low_bounds,
                    high=high_bounds,
                    max_iter=params.get('max_iter', 10000),
                    init_temp=params.get('init_temp', 500.0),
                    final_temp=params.get('final_temp', 1e-5),
                    alpha=params.get('alpha', 0.8),
                    iter_tol_max=params.get('iter_tol_max', 30),
                    random_state=None,
                    num_var=num_vars
                )
                end_time = time.time()
                elapsed_time = end_time - start_time

                min_val = -max(fitness)
                best_vars = xs
                saved_files = plot_result_1(raw_func, xs, x_hist, fitness, i, m, num_vars, low_bounds, high_bounds,
                                            elapsed_time, base_filename)

            res_msg = f"=== {solver_name} Optimization Completed ===\n"
            res_msg += f"Optimal Solution Position:\n"
            for name, val in zip(var_names, best_vars):
                res_msg += f"  {name} = {val:.8f}\n"
            res_msg += f"Minimum Function Value = {min_val:.8f}\n"
            res_msg += f"Convergence Iteration: {i}\n"
            res_msg += f"Time Elapsed: {elapsed_time:.4f}s\n\n"
            res_msg += f"[Success] Saved 2 plots:\n"
            for f_path in saved_files:
                res_msg += f"  - {os.path.abspath(f_path)}\n"

            self.result_text.insert(tk.END, res_msg)
            messagebox.showinfo("Done",
                                f"Optimization complete!\nMinimum Value: {min_val:.6f}\n2 Plots saved successfully.")

        except Exception as e:
            error_msg = f"Runtime Error: {str(e)}"
            messagebox.showerror("Runtime Error", error_msg)
            self.result_text.insert(tk.END, error_msg + "\n")


if __name__ == "__main__":
    root = tk.Tk()
    app = OptimizationApp(root)
    root.mainloop()