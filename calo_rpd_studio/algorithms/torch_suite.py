"""Torch-native canonical implementations of the nineteen baseline optimizers.

The mathematical operators remain recognizable canonical baseline operators.  Scientific strength
comes from the common double-precision AC evaluator, exact mixed-variable decoder, Deb
feasibility-first comparison, equal evaluation accounting, reflection boundary handling, and
identical robust-scenario aggregation—not from silently turning standard baselines into new
algorithms.
"""
from __future__ import annotations

import math
import time

import numpy as np

from calo_rpd_studio.accelerated.device import reflect_unit_interval, resolve_device
from calo_rpd_studio.orpd.feasibility_rules import better

from .base_optimizer import BaseOptimizer


class TorchCanonicalOptimizer(BaseOptimizer):
    """Dispatch one canonical optimizer while retaining the common result contract."""

    def __init__(self, algorithm_name, problem, config=None, seed=0, progress_callback=None, cancel_callback=None):
        self.name = str(algorithm_name)
        super().__init__(problem, config, seed, progress_callback, cancel_callback)
        import torch

        requested = str(self.config.parameters.get("execution_device", "cpu"))
        self.device_context = resolve_device(requested)
        self.device = torch.device(self.device_context.resolved)
        self.dtype = torch.float64
        try:
            self.torch_generator = torch.Generator(device=self.device.type)
            self.torch_generator.manual_seed(int(self.seed))
        except Exception:
            self.torch_generator = None

    def _tensor(self, value):
        import torch

        return torch.as_tensor(value, dtype=self.dtype, device=self.device)

    def _rand(self, shape):
        import torch

        return torch.rand(shape, dtype=self.dtype, device=self.device, generator=self.torch_generator)

    def _normal(self, shape, scale=1.0):
        import torch

        return torch.randn(shape, dtype=self.dtype, device=self.device, generator=self.torch_generator) * float(scale)

    def _uniform(self, low, high, shape):
        return float(low) + (float(high) - float(low)) * self._rand(shape)

    def _population(self, n=None):
        return self._rand((n or self.config.population_size, self.problem.dimension))

    def _bounded(self, value):
        strategy = str(self.config.parameters.get("boundary_strategy", "reflection")).lower()
        if strategy == "clip":
            return value.clamp(0.0, 1.0)
        return reflect_unit_interval(value)

    @staticmethod
    def _tensor_where(condition, first, second):
        import torch

        return torch.where(condition, first, second)

    def _eval_pop(self, population):
        remaining = max(0, int(self.config.max_evaluations) - int(self.evaluations))
        if remaining <= 0 or self.cancelled():
            return []
        population = self._bounded(population[:remaining])
        if bool(getattr(self.problem, "device_resident_enabled", False)):
            evaluations = list(self.problem.evaluate_population(population))
            # The evaluator's single packed host transfer already carries the normalized vectors,
            # so no second population CUDA->CPU copy is needed for incumbent/provenance tracking.
            registered = []
            for evaluation in evaluations:
                vector = np.asarray(
                    evaluation.metadata.get("normalized_decision_vector", ()), dtype=float
                )
                if vector.size != self.problem.dimension:
                    raise RuntimeError("Device-resident evaluation omitted its normalized vector")
                registered.append(self._register_evaluation(vector, evaluation))
            return registered
        return super().evaluate_population(population.detach().to("cpu").numpy())

    def _eval_one(self, candidate):
        evaluations = self._eval_pop(candidate.unsqueeze(0))
        return evaluations[0] if evaluations else None

    def _best_index(self, evaluations):
        return int(self.order(evaluations)[0])

    def _metadata(self):
        return {
            "optimizer_kernel": "torch_canonical",
            "optimizer_device": str(self.device),
            "optimizer_device_name": self.device_context.name,
            "optimizer_dtype": "float64",
            "boundary_strategy": str(self.config.parameters.get("boundary_strategy", "reflection")),
            "scientific_formulation": "canonical operator + common feasibility-first mixed-variable ORPD wrapper",
        }

    def run(self):
        dispatch = {
            "TLBO": self._run_tlbo,
            "PSO": self._run_pso,
            "CLPSO": self._run_clpso,
            "MTLA-DE": self._run_mtla_de,
            "QODE": self._run_qode,
            "DA": self._run_dragonfly,
            "SA": self._run_sa,
            "SSA": self._run_ssa,
            "ACO": self._run_acor,
            "BA": self._run_bat,
            "CSA": self._run_crow,
            "FA": self._run_firefly,
            "FPA": self._run_fpa,
            "GOA": self._run_goa,
            "GWO": self._run_gwo,
            "MFO": self._run_mfo,
            "MVO": self._run_mvo,
            "WOA": self._run_woa,
            "ICA": self._run_ica,
        }
        try:
            return dispatch[self.name]()
        except KeyError as exc:
            raise KeyError(f"No torch canonical implementation for {self.name}") from exc

    def _run_tlbo(self):
        started = time.perf_counter()
        pop = self._population()
        evaluations = self._eval_pop(pop)
        pop = pop[: len(evaluations)]
        while self.iteration < self.config.max_iterations and self.can_evaluate() and len(pop) > 1:
            self.iteration += 1
            best = pop[self._best_index(evaluations)].clone()
            mean = pop.mean(dim=0)
            teaching_factor = int(self.rng.integers(1, 3))
            teacher = self._bounded(pop + self._rand(tuple(pop.shape)) * (best - teaching_factor * mean))
            teacher_evals = self._eval_pop(teacher)
            for i, ev in enumerate(teacher_evals):
                if better(ev, evaluations[i]):
                    pop[i] = teacher[i]
                    evaluations[i] = ev
            n = len(pop)
            partners = np.asarray([self.rng.choice([j for j in range(n) if j != i]) for i in range(n)])
            partner_t = self._tensor(partners).long()
            better_mask = self._tensor(
                [1.0 if better(evaluations[i], evaluations[int(partners[i])]) else 0.0 for i in range(n)]
            ).bool()
            raw_direction = pop - pop[partner_t]
            directions = self._tensor_where(better_mask[:, None], raw_direction, -raw_direction)
            learner = self._bounded(pop + self._rand(tuple(pop.shape)) * directions)
            learner_evals = self._eval_pop(learner)
            for i, ev in enumerate(learner_evals):
                if better(ev, evaluations[i]):
                    pop[i] = learner[i]
                    evaluations[i] = ev
            self.record({"kernel_device": str(self.device)})
        return self.finalize(pop.detach().cpu().numpy(), metadata=self._metadata(), started=started)

    def _run_pso(self):
        started = time.perf_counter()
        pos = self._population()
        vel = self._uniform(-0.1, 0.1, tuple(pos.shape))
        evaluations = self._eval_pop(pos)
        pos = pos[: len(evaluations)]
        vel = vel[: len(evaluations)]
        pbest = pos.clone()
        pbest_eval = list(evaluations)
        inertia = float(self.config.parameters.get("inertia", 0.7298))
        c1 = float(self.config.parameters.get("c1", 1.49618))
        c2 = float(self.config.parameters.get("c2", 1.49618))
        vmax = float(self.config.parameters.get("velocity_limit", 0.2))
        while self.iteration < self.config.max_iterations and self.can_evaluate():
            self.iteration += 1
            gbest = pbest[self._best_index(pbest_eval)]
            vel = inertia * vel + c1 * self._rand(tuple(pos.shape)) * (pbest - pos) + c2 * self._rand(tuple(pos.shape)) * (gbest - pos)
            vel = vel.clamp(-vmax, vmax)
            pos = self._bounded(pos + vel)
            evaluations = self._eval_pop(pos)
            for i, ev in enumerate(evaluations):
                if better(ev, pbest_eval[i]):
                    pbest[i] = pos[i]
                    pbest_eval[i] = ev
            self.record({"kernel_device": str(self.device)})
        return self.finalize(pos.detach().cpu().numpy(), metadata=self._metadata(), started=started)

    def _run_clpso(self):
        started = time.perf_counter()
        n = self.config.population_size
        d = self.problem.dimension
        pos = self._population(n)
        vel = self._uniform(-0.1, 0.1, tuple(pos.shape))
        evaluations = self._eval_pop(pos)
        n = len(evaluations)
        pos = pos[:n]
        vel = vel[:n]
        pbest = pos.clone()
        pbest_eval = list(evaluations)
        stale = np.zeros(n, dtype=int)
        refresh = int(self.config.parameters.get("refresh_gap", 7))
        c = float(self.config.parameters.get("c", 1.49445))
        while self.iteration < self.config.max_iterations and self.can_evaluate():
            self.iteration += 1
            exemplar = pbest.clone()
            for i in range(n):
                if stale[i] >= refresh:
                    for j in range(d):
                        a, b = self.rng.choice(n, 2, replace=False)
                        winner = int(a if better(pbest_eval[a], pbest_eval[b]) else b)
                        exemplar[i, j] = pbest[winner, j]
                    stale[i] = 0
            progress = min(self.iteration / max(self.config.max_iterations, 1), 1.0)
            inertia = 0.9 - 0.5 * progress
            vel = inertia * vel + c * self._rand(tuple(pos.shape)) * (exemplar - pos)
            pos = self._bounded(pos + vel)
            evaluations = self._eval_pop(pos)
            for i, ev in enumerate(evaluations):
                if better(ev, pbest_eval[i]):
                    pbest[i] = pos[i]
                    pbest_eval[i] = ev
                    stale[i] = 0
                else:
                    stale[i] += 1
            self.record({"kernel_device": str(self.device)})
        return self.finalize(pos.detach().cpu().numpy(), metadata=self._metadata(), started=started)

    def _de_trials(self, pop, f, cr):
        import torch

        n, dimension = pop.shape
        choices = np.empty((n, 3), dtype=np.int64)
        for i in range(n):
            pool = np.delete(np.arange(n, dtype=np.int64), i)
            choices[i] = self.rng.choice(pool, 3, replace=False)
        index = torch.as_tensor(choices, dtype=torch.long, device=self.device)
        mutant = pop[index[:, 0]] + float(f) * (pop[index[:, 1]] - pop[index[:, 2]])
        mask = self._rand((n, dimension)) < float(cr)
        forced = torch.as_tensor(
            self.rng.integers(0, dimension, size=n), dtype=torch.long, device=self.device
        )
        mask[torch.arange(n, device=self.device), forced] = True
        return self._bounded(torch.where(mask, mutant, pop))

    def _run_mtla_de(self):
        started = time.perf_counter()
        pop = self._population()
        evaluations = self._eval_pop(pop)
        pop = pop[: len(evaluations)]
        f = float(self.config.parameters.get("f", 0.5))
        cr = float(self.config.parameters.get("cr", 0.9))
        while self.iteration < self.config.max_iterations and self.can_evaluate() and len(pop) >= 4:
            self.iteration += 1
            best = pop[self._best_index(evaluations)]
            mean = pop.mean(dim=0)
            teacher = self._bounded(pop + self._normal(tuple(pop.shape)) * (best - mean))
            teacher_evals = self._eval_pop(teacher)
            for i, ev in enumerate(teacher_evals):
                if better(ev, evaluations[i]):
                    pop[i] = teacher[i]
                    evaluations[i] = ev
            trials = self._de_trials(pop, f, cr)
            trial_evals = self._eval_pop(trials)
            for i, ev in enumerate(trial_evals):
                if better(ev, evaluations[i]):
                    pop[i] = trials[i]
                    evaluations[i] = ev
            self.record({"kernel_device": str(self.device)})
        return self.finalize(pop.detach().cpu().numpy(), metadata=self._metadata(), started=started)

    def _run_qode(self):
        started = time.perf_counter()
        n = self.config.population_size
        pop = self._population(n)
        opposite = 1.0 - pop
        lower = self._tensor_where(opposite < 0.5, opposite, self._tensor(0.5))
        upper = self._tensor_where(opposite > 0.5, opposite, self._tensor(0.5))
        quasi = lower + self._rand(tuple(pop.shape)) * (upper - lower)
        pool = __import__("torch").cat((pop, quasi), dim=0)
        pool_evals = self._eval_pop(pool)
        order = self.order(pool_evals)[: min(n, len(pool_evals))]
        pop = pool[order]
        evaluations = [pool_evals[i] for i in order]
        f = float(self.config.parameters.get("f", 0.5))
        cr = float(self.config.parameters.get("cr", 0.9))
        while self.iteration < self.config.max_iterations and self.can_evaluate() and len(pop) >= 4:
            self.iteration += 1
            trials = self._de_trials(pop, f, cr)
            trial_evals = self._eval_pop(trials)
            for i, ev in enumerate(trial_evals):
                if better(ev, evaluations[i]):
                    pop[i] = trials[i]
                    evaluations[i] = ev
            self.record({"kernel_device": str(self.device)})
        return self.finalize(pop.detach().cpu().numpy(), metadata=self._metadata(), started=started)

    def _run_dragonfly(self):
        started = time.perf_counter()
        pop = self._population()
        step = __import__("torch").zeros_like(pop)
        evaluations = self._eval_pop(pop)
        pop = pop[: len(evaluations)]
        step = step[: len(evaluations)]
        while self.iteration < self.config.max_iterations and self.can_evaluate():
            self.iteration += 1
            order = self.order(evaluations)
            food = pop[order[0]]
            enemy = pop[order[-1]]
            mean = pop.mean(dim=0)
            progress = min(self.iteration / max(self.config.max_iterations, 1), 1.0)
            inertia = 0.9 - 0.5 * progress
            separation = 0.1 * (mean - pop)
            alignment = 0.1 * step.mean(dim=0)
            cohesion = 0.1 * (mean - pop)
            food_attraction = 0.5 * (food - pop)
            enemy_distraction = 0.1 * (pop - enemy)
            step = (inertia * step + separation + alignment + cohesion + food_attraction + enemy_distraction).clamp(-0.2, 0.2)
            pop = self._bounded(pop + step)
            evaluations = self._eval_pop(pop)
            self.record({"kernel_device": str(self.device)})
        return self.finalize(pop.detach().cpu().numpy(), metadata=self._metadata(), started=started)

    def _run_sa(self):
        started = time.perf_counter()
        x = self._population(1)[0]
        current = self._eval_one(x)
        temperature = float(self.config.parameters.get("temperature", 1.0))
        cooling = float(self.config.parameters.get("cooling", 0.995))
        scale = float(self.config.parameters.get("step_scale", 0.1))
        while self.iteration < self.config.max_iterations and self.can_evaluate():
            self.iteration += 1
            candidate = self._bounded(x + self._normal((self.problem.dimension,), scale))
            candidate_eval = self._eval_one(candidate)
            if candidate_eval is None:
                break
            delta = candidate_eval.value - current.value if candidate_eval.feasible == current.feasible else candidate_eval.violation - current.violation
            if better(candidate_eval, current) or self.rng.random() < math.exp(-max(float(delta), 0.0) / max(temperature, 1e-12)):
                x = candidate
                current = candidate_eval
            temperature *= cooling
            self.record({"kernel_device": str(self.device)})
        return self.finalize(np.asarray([x.detach().cpu().numpy()]), metadata=self._metadata(), started=started)

    def _run_ssa(self):
        started = time.perf_counter()
        pop = self._population()
        evaluations = self._eval_pop(pop)
        pop = pop[: len(evaluations)]
        while self.iteration < self.config.max_iterations and self.can_evaluate():
            self.iteration += 1
            food = pop[self._best_index(evaluations)]
            c1 = 2 * math.exp(-(4 * self.iteration / max(self.config.max_iterations, 1)) ** 2)
            new = pop.clone()
            sign = self._tensor_where(self._rand((self.problem.dimension,)) < 0.5, self._tensor(1.0), self._tensor(-1.0))
            new[0] = self._bounded(food + sign * c1 * self._rand((self.problem.dimension,)))
            for i in range(1, len(pop)):
                new[i] = 0.5 * (pop[i] + new[i - 1])
            pop = self._bounded(new)
            evaluations = self._eval_pop(pop)
            self.record({"kernel_device": str(self.device)})
        return self.finalize(pop.detach().cpu().numpy(), metadata=self._metadata(), started=started)

    def _run_acor(self):
        started = time.perf_counter()
        n = self.config.population_size
        archive = self._population(n)
        evaluations = self._eval_pop(archive)
        archive = archive[: len(evaluations)]
        n = len(archive)
        q = float(self.config.parameters.get("q", 0.5))
        xi = float(self.config.parameters.get("xi", 0.85))
        while self.iteration < self.config.max_iterations and self.can_evaluate() and n:
            self.iteration += 1
            order = self.order(evaluations)
            archive = archive[order]
            evaluations = [evaluations[i] for i in order]
            ranks = np.arange(n)
            weights = np.exp(-(ranks**2) / max(2 * (q * n) ** 2, 1e-12))
            weights /= weights.sum()
            samples = []
            for _ in range(min(n, self.config.max_evaluations - self.evaluations)):
                k = int(self.rng.choice(n, p=weights))
                sigma = xi * __import__("torch").mean(__import__("torch").abs(archive[k] - archive), dim=0)
                samples.append(self._bounded(archive[k] + self._normal((self.problem.dimension,)) * (sigma + 1e-12)))
            if not samples:
                break
            new = __import__("torch").stack(samples, dim=0)
            new_evals = self._eval_pop(new)
            combo = __import__("torch").cat((archive, new[: len(new_evals)]), dim=0)
            all_evals = evaluations + new_evals
            order = self.order(all_evals)[:n]
            archive = combo[order]
            evaluations = [all_evals[i] for i in order]
            self.record({"kernel_device": str(self.device)})
        return self.finalize(archive.detach().cpu().numpy(), metadata=self._metadata(), started=started)

    def _run_bat(self):
        started = time.perf_counter()
        pop = self._population()
        velocity = __import__("torch").zeros_like(pop)
        evaluations = self._eval_pop(pop)
        pop = pop[: len(evaluations)]
        velocity = velocity[: len(evaluations)]
        loudness = float(self.config.parameters.get("loudness", 0.9))
        pulse = float(self.config.parameters.get("pulse_rate", 0.5))
        while self.iteration < self.config.max_iterations and self.can_evaluate():
            self.iteration += 1
            best = pop[self._best_index(evaluations)]
            candidates = []
            for i in range(len(pop)):
                frequency = float(self.rng.uniform(0, 2))
                velocity[i] = velocity[i] + (pop[i] - best) * frequency
                candidate = self._bounded(pop[i] + velocity[i])
                if self.rng.random() > pulse:
                    candidate = self._bounded(best + self._normal((self.problem.dimension,), 0.01))
                candidates.append(candidate)
            cand = __import__("torch").stack(candidates, dim=0)
            cand_evals = self._eval_pop(cand)
            for i, ev in enumerate(cand_evals):
                if better(ev, evaluations[i]) and self.rng.random() < loudness:
                    pop[i] = cand[i]
                    evaluations[i] = ev
            self.record({"kernel_device": str(self.device)})
        return self.finalize(pop.detach().cpu().numpy(), metadata=self._metadata(), started=started)

    def _run_crow(self):
        started = time.perf_counter()
        pop = self._population()
        evaluations = self._eval_pop(pop)
        pop = pop[: len(evaluations)]
        memory = pop.clone()
        memory_evals = list(evaluations)
        awareness = float(self.config.parameters.get("awareness_probability", 0.1))
        flight = float(self.config.parameters.get("flight_length", 2.0))
        while self.iteration < self.config.max_iterations and self.can_evaluate():
            self.iteration += 1
            candidates = []
            for i in range(len(pop)):
                j = int(self.rng.integers(len(pop)))
                if self.rng.random() > awareness:
                    candidate = pop[i] + float(self.rng.random()) * flight * (memory[j] - pop[i])
                else:
                    candidate = self._population(1)[0]
                candidates.append(self._bounded(candidate))
            pop = __import__("torch").stack(candidates, dim=0)
            evaluations = self._eval_pop(pop)
            for i, ev in enumerate(evaluations):
                if better(ev, memory_evals[i]):
                    memory[i] = pop[i]
                    memory_evals[i] = ev
            self.record({"kernel_device": str(self.device)})
        return self.finalize(memory.detach().cpu().numpy(), metadata=self._metadata(), started=started)

    def _run_firefly(self):
        started = time.perf_counter()
        pop = self._population()
        evaluations = self._eval_pop(pop)
        pop = pop[: len(evaluations)]
        alpha = float(self.config.parameters.get("alpha", 0.2))
        beta0 = float(self.config.parameters.get("beta0", 1.0))
        gamma = float(self.config.parameters.get("gamma", 1.0))
        while self.iteration < self.config.max_iterations and self.can_evaluate():
            self.iteration += 1
            candidates = pop.clone()
            for i in range(len(pop)):
                attractors = [j for j in range(len(pop)) if better(evaluations[j], evaluations[i])]
                if attractors:
                    j = int(attractors[int(self.rng.integers(len(attractors)))])
                    r2 = __import__("torch").sum((pop[i] - pop[j]) ** 2)
                    beta = beta0 * __import__("torch").exp(-gamma * r2)
                    candidates[i] = self._bounded(pop[i] + beta * (pop[j] - pop[i]) + alpha * (self._rand((self.problem.dimension,)) - 0.5))
            candidate_evals = self._eval_pop(candidates)
            for i, ev in enumerate(candidate_evals):
                if better(ev, evaluations[i]):
                    pop[i] = candidates[i]
                    evaluations[i] = ev
            alpha *= 0.98
            self.record({"kernel_device": str(self.device)})
        return self.finalize(pop.detach().cpu().numpy(), metadata=self._metadata(), started=started)

    def _levy(self, shape, beta=1.5):
        sigma = (
            math.gamma(1 + beta)
            * math.sin(math.pi * beta / 2)
            / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
        ) ** (1 / beta)
        u = self.rng.normal(0, sigma, shape)
        v = self.rng.normal(0, 1, shape)
        return self._tensor(u / (np.abs(v) ** (1 / beta) + 1e-12))

    def _run_fpa(self):
        started = time.perf_counter()
        pop = self._population()
        evaluations = self._eval_pop(pop)
        pop = pop[: len(evaluations)]
        switch = float(self.config.parameters.get("switch_probability", 0.8))
        while self.iteration < self.config.max_iterations and self.can_evaluate():
            self.iteration += 1
            best = pop[self._best_index(evaluations)].clone()
            candidates = []
            for i in range(len(pop)):
                if self.rng.random() < switch:
                    candidate = pop[i] + self._levy((self.problem.dimension,)) * (best - pop[i])
                else:
                    a, b = self.rng.choice(len(pop), 2, replace=False)
                    candidate = pop[i] + float(self.rng.random()) * (pop[a] - pop[b])
                candidates.append(self._bounded(candidate))
            cand = __import__("torch").stack(candidates, dim=0)
            cand_evals = self._eval_pop(cand)
            for i, ev in enumerate(cand_evals):
                if better(ev, evaluations[i]):
                    pop[i] = cand[i]
                    evaluations[i] = ev
            self.record({"kernel_device": str(self.device)})
        return self.finalize(pop.detach().cpu().numpy(), metadata=self._metadata(), started=started)

    @staticmethod
    def _social(distance, f=0.5, length_scale=1.5):
        return f * math.exp(-distance / length_scale) - math.exp(-distance)

    def _run_goa(self):
        started = time.perf_counter()
        pop = self._population()
        evaluations = self._eval_pop(pop)
        pop = pop[: len(evaluations)]
        while self.iteration < self.config.max_iterations and self.can_evaluate():
            self.iteration += 1
            target = pop[self._best_index(evaluations)]
            c = 1.0 - 0.99999 * min(self.iteration / max(self.config.max_iterations, 1), 1.0)
            torch = __import__("torch")
            difference = pop[None, :, :] - pop[:, None, :]
            distance = torch.linalg.vector_norm(difference, dim=2).clamp_min(1e-12)
            eye = torch.eye(len(pop), dtype=torch.bool, device=self.device)
            social_strength = 0.5 * torch.exp(-(10.0 * distance) / 1.5) - torch.exp(-(10.0 * distance))
            social_strength = torch.where(eye, torch.zeros_like(social_strength), social_strength)
            social = torch.sum(social_strength[:, :, None] * difference / distance[:, :, None], dim=1)
            pop = self._bounded(target[None, :] + c * social / max(len(pop) - 1, 1))
            evaluations = self._eval_pop(pop)
            self.record({"kernel_device": str(self.device)})
        return self.finalize(pop.detach().cpu().numpy(), metadata=self._metadata(), started=started)

    def _run_gwo(self):
        started = time.perf_counter()
        pop = self._population()
        evaluations = self._eval_pop(pop)
        pop = pop[: len(evaluations)]
        while self.iteration < self.config.max_iterations and self.can_evaluate():
            self.iteration += 1
            order = self.order(evaluations)
            leaders = [pop[order[min(k, len(order) - 1)]] for k in range(3)]
            a = 2 * (1 - min(self.iteration / max(self.config.max_iterations, 1), 1.0))
            estimates = []
            for leader in leaders:
                A = 2 * a * self._rand(tuple(pop.shape)) - a
                C = 2 * self._rand(tuple(pop.shape))
                estimates.append(leader - A * abs(C * leader - pop))
            pop = self._bounded(sum(estimates) / 3.0)
            evaluations = self._eval_pop(pop)
            self.record({"kernel_device": str(self.device)})
        return self.finalize(pop.detach().cpu().numpy(), metadata=self._metadata(), started=started)

    def _run_mfo(self):
        started = time.perf_counter()
        pop = self._population()
        evaluations = self._eval_pop(pop)
        pop = pop[: len(evaluations)]
        flames = pop.clone()
        flame_evals = list(evaluations)
        while self.iteration < self.config.max_iterations and self.can_evaluate():
            self.iteration += 1
            combo = __import__("torch").cat((flames, pop), dim=0)
            all_evals = flame_evals + evaluations
            order = self.order(all_evals)[: len(pop)]
            flames = combo[order]
            flame_evals = [all_evals[i] for i in order]
            flame_count = max(1, int(round(len(pop) - (len(pop) - 1) * self.iteration / max(self.config.max_iterations, 1))))
            new = pop.clone()
            for i in range(len(pop)):
                flame = flames[min(i, flame_count - 1)]
                distance = abs(flame - pop[i])
                t = self._uniform(-1, 1, (self.problem.dimension,))
                new[i] = self._bounded(distance * t.exp() * (2 * math.pi * t).cos() + flame)
            pop = new
            evaluations = self._eval_pop(pop)
            self.record({"kernel_device": str(self.device)})
        return self.finalize(pop.detach().cpu().numpy(), metadata=self._metadata(), started=started)

    def _run_mvo(self):
        started = time.perf_counter()
        pop = self._population()
        evaluations = self._eval_pop(pop)
        pop = pop[: len(evaluations)]
        while self.iteration < self.config.max_iterations and self.can_evaluate():
            self.iteration += 1
            order = self.order(evaluations)
            best = pop[order[0]].clone()
            ranks = np.empty(len(pop))
            ranks[order] = np.arange(len(pop))
            normalized = (len(pop) - ranks) / max(len(pop), 1)
            donor_prob = normalized / normalized.sum()
            progress = self.iteration / max(self.config.max_iterations, 1)
            wep = 0.2 + 0.8 * progress
            tdr = 1 - progress ** (1 / 6)
            new = pop.clone()
            for i in range(len(pop)):
                for j in range(self.problem.dimension):
                    if self.rng.random() < normalized[i]:
                        donor = int(self.rng.choice(len(pop), p=donor_prob))
                        new[i, j] = pop[donor, j]
                    if self.rng.random() < wep:
                        sign = -1 if self.rng.random() < 0.5 else 1
                        new[i, j] = best[j] + sign * tdr * float(self.rng.random())
            pop = self._bounded(new)
            evaluations = self._eval_pop(pop)
            self.record({"kernel_device": str(self.device)})
        return self.finalize(pop.detach().cpu().numpy(), metadata=self._metadata(), started=started)

    def _run_woa(self):
        started = time.perf_counter()
        pop = self._population()
        evaluations = self._eval_pop(pop)
        pop = pop[: len(evaluations)]
        b = float(self.config.parameters.get("spiral_b", 1.0))
        while self.iteration < self.config.max_iterations and self.can_evaluate():
            self.iteration += 1
            best = pop[self._best_index(evaluations)].clone()
            a = 2 * (1 - min(self.iteration / max(self.config.max_iterations, 1), 1.0))
            new = pop.clone()
            for i, x in enumerate(pop):
                A = 2 * a * float(self.rng.random()) - a
                C = 2 * float(self.rng.random())
                if self.rng.random() < 0.5:
                    reference = best if abs(A) < 1 else pop[int(self.rng.integers(len(pop)))]
                    candidate = reference - A * abs(C * reference - x)
                else:
                    distance = abs(best - x)
                    l = float(self.rng.uniform(-1, 1))
                    candidate = distance * math.exp(b * l) * math.cos(2 * math.pi * l) + best
                new[i] = self._bounded(candidate)
            pop = new
            evaluations = self._eval_pop(pop)
            self.record({"kernel_device": str(self.device)})
        return self.finalize(pop.detach().cpu().numpy(), metadata=self._metadata(), started=started)

    def _run_ica(self):
        started = time.perf_counter()
        pop = self._population()
        evaluations = self._eval_pop(pop)
        pop = pop[: len(evaluations)]
        n_imperialists = max(1, min(int(self.config.parameters.get("imperialists", max(1, len(pop) // 5))), len(pop) - 1))
        while self.iteration < self.config.max_iterations and self.can_evaluate() and len(pop) > 1:
            self.iteration += 1
            order = self.order(evaluations)
            imperialists = order[:n_imperialists]
            colonies = order[n_imperialists:]
            candidates = pop.clone()
            for colony in colonies:
                imperialist = int(self.rng.choice(imperialists))
                candidates[colony] = self._bounded(
                    pop[colony]
                    + float(self.rng.uniform(0, 2)) * (pop[imperialist] - pop[colony])
                    + self._normal((self.problem.dimension,), 0.02)
                )
            candidate_evals = self._eval_pop(candidates)
            for i, ev in enumerate(candidate_evals):
                if better(ev, evaluations[i]):
                    pop[i] = candidates[i]
                    evaluations[i] = ev
            self.record({"kernel_device": str(self.device)})
        return self.finalize(pop.detach().cpu().numpy(), metadata=self._metadata(), started=started)
