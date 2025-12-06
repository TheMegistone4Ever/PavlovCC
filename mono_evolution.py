import os

import matplotlib.pyplot as plt
import numpy as np
import pacmap
import torch
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

SEED = 1810
DPI = 200
FIG_SIZE_2D = (20, 16)
FIG_SIZE_3D = (20, 16)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class BatchLPSolver:
    def __init__(self, A, b, c, y_min, alpha, t0, D):
        self.device = A.device
        self.A = A
        self.b = b
        self.c = c
        self.y_min = y_min
        self.alpha = alpha
        self.t0 = t0
        self.D = D

        self.num_products = A.shape[1]
        self.num_assigned = alpha.shape[0]

        self.b_expanded = self.b.unsqueeze(0)
        self.c_expanded = self.c.unsqueeze(0)
        self.y_min_expanded = self.y_min.unsqueeze(0)
        self.D_expanded = self.D.unsqueeze(0)
        self.t0_expanded = self.t0.unsqueeze(0)
        self.alpha_expanded = self.alpha.unsqueeze(0)

    def solve(self, omegas, n_steps=1000, lr=0.1):
        batch_size = omegas.shape[0]

        init_y_val = self.y_min.max().item() + 10.0
        y_data = torch.full((batch_size, self.num_products), init_y_val, device=self.device)
        y = y_data.clone().detach().requires_grad_(True)

        z_data = torch.ones((batch_size, self.num_assigned), device=self.device) * 10.0
        z = z_data.clone().detach().requires_grad_(True)

        optimizer = torch.optim.Adam([y, z], lr=lr)

        mu = 1.0
        decay = 0.99

        for _ in range(n_steps):
            optimizer.zero_grad()

            obj = (omegas * z).sum(dim=1) - (self.c_expanded * y).sum(dim=1)

            res_res = self.b_expanded - y @ self.A.T
            y_assigned = y[:, :self.num_assigned]
            assigned_min_res = y_assigned - self.y_min_expanded
            y_pos_res = y
            z_pos_res = z
            time_res = (self.D_expanded + z) - (self.t0_expanded + self.alpha_expanded * y_assigned)

            barrier = -torch.log(torch.clamp(res_res, min=1e-8)).sum(dim=1) \
                      - torch.log(torch.clamp(assigned_min_res, min=1e-8)).sum(dim=1) \
                      - torch.log(torch.clamp(y_pos_res, min=1e-8)).sum(dim=1) \
                      - torch.log(torch.clamp(z_pos_res, min=1e-8)).sum(dim=1) \
                      - torch.log(torch.clamp(time_res, min=1e-8)).sum(dim=1)

            loss = (obj + mu * barrier).mean()

            if torch.isnan(loss):
                break

            loss.backward()
            optimizer.step()
            mu *= decay

        return y.detach(), z.detach()


class MonoEvolutionAlgorithm:
    def __init__(self, lp_data, pop_size, n_generations, beta1, beta2, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        self.A = torch.tensor(lp_data['production_matrix'], dtype=torch.float32, device=self.device)
        self.b = torch.tensor(lp_data['b'], dtype=torch.float32, device=self.device)
        self.c = torch.tensor(lp_data['c'], dtype=torch.float32, device=self.device)
        self.y_min = torch.tensor(lp_data['y_assigned'], dtype=torch.float32, device=self.device)
        self.D = torch.tensor(lp_data['directive_terms'], dtype=torch.float32, device=self.device)
        self.t0 = torch.tensor(lp_data['t0'], dtype=torch.float32, device=self.device)
        self.alpha = torch.tensor(lp_data['alpha'], dtype=torch.float32, device=self.device)
        self.f_coeffs = torch.tensor(lp_data['f_coeffs'], dtype=torch.float32, device=self.device)

        self.pop_size = pop_size
        self.n_generations = n_generations
        self.beta1 = beta1
        self.beta2 = beta2

        self.num_assigned = self.alpha.shape[0]
        self.lp_solver = BatchLPSolver(self.A, self.b, self.c, self.y_min, self.alpha, self.t0, self.D)

        self.history_best_f = []
        self.history_avg_f = []
        self.history_best_omega = []
        self.history_avg_omega = []

    def _penalty_function(self, z):
        return self.f_coeffs.unsqueeze(0) * (z ** 2)

    def _mono_evolution_operator(self, omegas, z_sol):
        f_vals = self._penalty_function(z_sol)
        means = f_vals.mean(dim=1, keepdim=True)

        new_omegas = omegas.clone()

        # TODO: Vectorized Implementation
        bad_mask = f_vals > means
        good_mask = f_vals <= means

        max_f, _ = f_vals.max(dim=1, keepdim=True)
        gamma = f_vals / torch.clamp(max_f, min=1e-8)

        new_omegas[bad_mask] = omegas[bad_mask] * (1.0 + self.beta1 * gamma[bad_mask])

        ratio = f_vals / torch.clamp(means, min=1e-8)
        decrease_factor = 1.0 - self.beta2 * (1.0 - ratio)
        new_omegas[good_mask] = omegas[good_mask] * decrease_factor[good_mask]

        return new_omegas

    def run(self):
        omegas = torch.rand((self.pop_size, self.num_assigned), device=self.device) + 0.1

        for generation in range(self.n_generations):
            y_sol, z_sol = self.lp_solver.solve(omegas)

            f_vals = self._penalty_function(z_sol)
            total_fitness = f_vals.sum(dim=1)

            best_val, best_idx = torch.min(total_fitness, dim=0)
            avg_val = total_fitness.mean()

            self.history_best_f.append(best_val.item())
            self.history_avg_f.append(avg_val.item())

            current_best_omega = omegas[best_idx].detach().cpu().numpy()
            current_avg_omega = omegas.mean(dim=0).detach().cpu().numpy()

            self.history_best_omega.append(current_best_omega)
            self.history_avg_omega.append(current_avg_omega)

            if generation % 5 == 0:
                print(f"Gen {generation}: Best F={best_val.item():.4f}, Avg F={avg_val.item():.4f}")

            sorted_indices = torch.argsort(total_fitness)
            omegas = omegas[sorted_indices]
            z_sol = z_sol[sorted_indices]

            parents_a = omegas[:self.pop_size // 2]
            parents_b = omegas[torch.randperm(self.pop_size // 2, device=self.device)]
            children_crossover = (parents_a + parents_b) / 2.0

            mutated_omegas = self._mono_evolution_operator(omegas, z_sol)

            n_elites = int(self.pop_size * 0.2)
            n_cross = int(self.pop_size * 0.4)
            n_mutate = self.pop_size - n_elites - n_cross

            next_gen = []
            next_gen.append(omegas[:n_elites])
            next_gen.append(children_crossover[:n_cross])
            next_gen.append(mutated_omegas[n_elites:n_elites + n_mutate])

            omegas = torch.cat(next_gen, dim=0)
            omegas = torch.clamp(omegas, min=0.01)

        self._plot_results()

    def _plot_results(self):
        best_omegas = np.array(self.history_best_omega)
        avg_omegas = np.array(self.history_avg_omega)
        diff_omegas = best_omegas - avg_omegas
        generations = np.arange(self.n_generations)

        base_dir = "results"
        os.makedirs(base_dir, exist_ok=True)

        # 1. Convergence
        plt.figure(figsize=FIG_SIZE_2D, dpi=DPI)
        plt.plot(self.history_best_f, label=r'Best F(omega)', linewidth=2)
        plt.plot(self.history_avg_f, label=r'Avg F(omega)', linestyle='--', linewidth=2)
        plt.xlabel('Generation', fontsize=14)
        plt.ylabel('Objective Value', fontsize=14)
        plt.title('Algorithm Convergence', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(base_dir, 'convergence.png'))
        plt.close()

        # 2. Heatmaps
        vmin = min(best_omegas.min(), avg_omegas.min())
        vmax = max(best_omegas.max(), avg_omegas.max())

        fig, axes = plt.subplots(2, 1, figsize=(16, 12), dpi=DPI)

        im1 = axes[0].imshow(best_omegas.T, aspect='auto', cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
        axes[0].set_title(r'Evolution of Best omega Vector Components', fontsize=16)
        axes[0].set_ylabel('Component Index', fontsize=12)

        im2 = axes[1].imshow(avg_omegas.T, aspect='auto', cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
        axes[1].set_title(r'Evolution of Average omega Vector Components', fontsize=16)
        axes[1].set_ylabel('Component Index', fontsize=12)
        axes[1].set_xlabel('Generation', fontsize=12)

        cbar = fig.colorbar(im1, ax=axes.ravel().tolist(), label='Component Value')
        cbar.ax.tick_params(labelsize=10)

        plt.savefig(os.path.join(base_dir, 'heatmaps.png'))
        plt.close()

        # 3. Manifold Learning (32 Plots)
        self._generate_manifolds(best_omegas, avg_omegas, diff_omegas, generations, base_dir)

    def _generate_manifolds(self, best, avg, diff, gens, base_dir):
        # Prepare Data
        combined = np.vstack([best, avg])
        n_samples = len(best)

        reducers = {
            "PCA": lambda n: PCA(n_components=n, random_state=SEED),
            "t-SNE": lambda n: TSNE(n_components=n, perplexity=min(30, n_samples - 1), random_state=SEED),
            "UMAP": lambda n: umap.UMAP(n_components=n, random_state=SEED),
            "PaCMAP": lambda n: pacmap.PaCMAP(n_components=n, n_neighbors=min(10, n_samples - 1), random_state=SEED)
        }

        # Loop: 4 Methods
        for method_name, reducer_factory in reducers.items():
            method_dir = os.path.join(base_dir, method_name)
            os.makedirs(method_dir, exist_ok=True)

            # Loop: 2 Dimensions
            for dim in [2, 3]:
                dim_dir = os.path.join(method_dir, f"{dim}D")
                os.makedirs(dim_dir, exist_ok=True)

                print(f"Computing {method_name} {dim}D...")

                try:
                    # Fit Combined
                    red = reducer_factory(dim)
                    emb_combined = red.fit_transform(combined)
                    emb_best = emb_combined[:n_samples]
                    emb_avg = emb_combined[n_samples:]

                    # Fit Diff (requires separate manifold learning as topology differs)
                    red_diff = reducer_factory(dim)
                    emb_diff = red_diff.fit_transform(diff)

                    # --- PLOT 1: Best Trajectory ---
                    self._save_plot(emb_best, gens, f"{method_name} - Best omega", dim,
                                    os.path.join(dim_dir, "best_trajectory.png"))

                    # --- PLOT 2: Average Trajectory ---
                    self._save_plot(emb_avg, gens, f"{method_name} - Average omega", dim,
                                    os.path.join(dim_dir, "avg_trajectory.png"))

                    # --- PLOT 3: Combined View ---
                    self._save_combined_plot(emb_best, emb_avg, gens, f"{method_name} - Combined", dim,
                                             os.path.join(dim_dir, "combined_view.png"))

                    # --- PLOT 4: Difference Trajectory (Best - Avg) ---
                    self._save_plot(emb_diff, gens, f"{method_name} - Difference (Best - Avg)", dim,
                                    os.path.join(dim_dir, "diff_trajectory.png"))

                except Exception as e:
                    print(f"Skipping {method_name} {dim}D: {e}")

    def _save_plot(self, data, gens, title, dim, path):
        fig = plt.figure(figsize=FIG_SIZE_3D if dim == 3 else FIG_SIZE_2D, dpi=DPI)

        if dim == 3:
            ax = fig.add_subplot(111, projection='3d')
            sc = ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=gens, cmap='plasma', s=60, alpha=0.9)
            ax.plot(data[:, 0], data[:, 1], data[:, 2], color='gray', alpha=0.4)
            ax.set_zlabel('Comp 3')
        else:
            ax = fig.add_subplot(111)
            sc = ax.scatter(data[:, 0], data[:, 1], c=gens, cmap='plasma', s=60, alpha=0.9)
            ax.plot(data[:, 0], data[:, 1], color='gray', alpha=0.4)

        ax.set_title(title, fontsize=18)
        ax.set_xlabel('Comp 1', fontsize=14)
        ax.set_ylabel('Comp 2', fontsize=14)
        cbar = plt.colorbar(sc, label='Generation')
        cbar.ax.tick_params(labelsize=12)

        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    def _save_combined_plot(self, best, avg, gens, title, dim, path):
        fig = plt.figure(figsize=FIG_SIZE_3D if dim == 3 else FIG_SIZE_2D, dpi=DPI)

        if dim == 3:
            ax = fig.add_subplot(111, projection='3d')
            # Best = Circles, Avg = Triangles
            sc1 = ax.scatter(best[:, 0], best[:, 1], best[:, 2], c=gens, cmap='viridis', marker='o', s=60, label='Best')
            sc2 = ax.scatter(avg[:, 0], avg[:, 1], avg[:, 2], c=gens, cmap='magma', marker='^', s=60, label='Avg')

            # Connect corresponding generations
            for i in range(len(best)):
                ax.plot([best[i, 0], avg[i, 0]], [best[i, 1], avg[i, 1]], [best[i, 2], avg[i, 2]],
                        color='black', alpha=0.1, linewidth=0.5)

            ax.set_zlabel('Comp 3')
        else:
            ax = fig.add_subplot(111)
            sc1 = ax.scatter(best[:, 0], best[:, 1], c=gens, cmap='viridis', marker='o', s=60, label='Best')
            sc2 = ax.scatter(avg[:, 0], avg[:, 1], c=gens, cmap='magma', marker='^', s=60, label='Avg')

            for i in range(len(best)):
                ax.plot([best[i, 0], avg[i, 0]], [best[i, 1], avg[i, 1]],
                        color='black', alpha=0.1, linewidth=0.5)

        ax.set_title(title, fontsize=18)
        ax.set_xlabel('Comp 1', fontsize=14)
        ax.set_ylabel('Comp 2', fontsize=14)
        ax.legend(fontsize=12)

        plt.tight_layout()
        plt.savefig(path)
        plt.close()


if __name__ == '__main__':
    set_seed(SEED)

    NUM_AGGREGATED_PRODUCTS = 20
    NUM_PRODUCTION_FACTORS = 20
    NUM_ASSIGNED_PRODUCTS = 10

    lp_data = {
        'production_matrix': np.random.uniform(0.1, 1, (NUM_AGGREGATED_PRODUCTS, NUM_PRODUCTION_FACTORS)),
        'b': np.random.uniform(2000, 10000, NUM_AGGREGATED_PRODUCTS),
        'c': np.random.uniform(1, 10, NUM_PRODUCTION_FACTORS),
        'y_assigned': np.random.uniform(5, 20, NUM_ASSIGNED_PRODUCTS),
        'directive_terms': np.sort(np.random.uniform(50, 150, NUM_ASSIGNED_PRODUCTS)),
        't0': np.zeros(NUM_ASSIGNED_PRODUCTS),
        'alpha': np.random.uniform(0.5, 1.5, NUM_ASSIGNED_PRODUCTS),
        'f_coeffs': np.random.uniform(0.1, 1.0, NUM_ASSIGNED_PRODUCTS)
    }

    algo = MonoEvolutionAlgorithm(
        lp_data=lp_data,
        pop_size=100,
        n_generations=60,
        beta1=0.8,
        beta2=0.2,
        device='cuda'
    )

    algo.run()
