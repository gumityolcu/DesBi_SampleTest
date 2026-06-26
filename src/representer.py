import os
import time

import torch

from dualxda import DualDA


class RepresenterPoints(DualDA):
    name = "RepresenterPointsExplainer"

    def __init__(
        self,
        model,
        dataset,
        classifier_layer,
        device,
        cache_dir,
        max_iter=3000,
        lmbd=0.003,
        initial_step=10.0,
        line_search_beta=0.5,
        min_step=1e-10,
        verbose=True,
    ):
        self.lmbd = lmbd
        self.initial_step = initial_step
        self.line_search_beta = line_search_beta
        self.min_step = min_step
        self.verbose = verbose
        super().__init__(
            model=model,
            dataset=dataset,
            classifier_layer=classifier_layer,
            device=device,
            cache_dir=cache_dir,
            C=None,
            max_iter=max_iter,
        )

    @staticmethod
    def _soft_cross_entropy(logits, targets):
        return -(targets * torch.log_softmax(logits, dim=1)).sum()

    def _objective(self, features, targets, weights):
        phi = self._soft_cross_entropy(features @ weights, targets)
        l2 = weights.square().sum()
        return self.lmbd * l2 + phi / features.shape[0], phi, l2

    def _backtracking_step(self, features, targets, weights, grad, loss):
        step = self.initial_step
        grad_norm_sq = grad.square().sum()

        while step >= self.min_step:
            candidate = weights - step * grad
            candidate_loss, _, _ = self._objective(features, targets, candidate)
            if candidate_loss - loss + step * grad_norm_sq / 2 < 0:
                return candidate.detach()
            step *= self.line_search_beta

        return weights.detach()

    def train(self):
        tstart = time.time()
        classifier = dict(self.model.named_modules()).get(self.classifier, None)
        if classifier is None:
            raise ValueError(f"Layer '{self.classifier}' not found in model.")
        if not hasattr(classifier, "weight"):
            raise ValueError(
                "Representer points require a classifier layer with a weight matrix."
            )

        features = self.samples.detach().to(self.device).float()
        with torch.no_grad():
            targets = torch.softmax(classifier(features), dim=1)

        weights = classifier.weight.detach().T.to(self.device).float().clone()
        best_weights = weights.clone()
        min_grad = None
        initial_grad = None

        if self.verbose:
            print("Training representer-point surrogate")

        for epoch in range(self.max_iter):
            weights = weights.detach().requires_grad_(True)
            loss, phi, _ = self._objective(features, targets, weights)
            grad = torch.autograd.grad(loss, weights)[0]
            grad_loss = torch.mean(torch.abs(grad)).detach()

            if min_grad is None or grad_loss < min_grad:
                if initial_grad is None:
                    initial_grad = grad_loss
                min_grad = grad_loss
                best_weights = weights.detach().clone()
                if min_grad < initial_grad / 200:
                    if self.verbose:
                        print(f"stopping criteria reached in epoch: {epoch}")
                    break

            if self.verbose and epoch % 100 == 0:
                print(
                    "Epoch:{:4d}\tloss:{}\tphi_loss:{}\tgrad:{}".format(
                        epoch,
                        loss.detach().cpu().item(),
                        (phi / features.shape[0]).detach().cpu().item(),
                        grad_loss.cpu().item(),
                    )
                )

            weights = self._backtracking_step(
                features=features,
                targets=targets,
                weights=weights.detach(),
                grad=grad.detach(),
                loss=loss.detach(),
            )

        with torch.no_grad():
            surrogate_probs = torch.softmax(features @ best_weights, dim=1)
            coefficients = (surrogate_probs - targets) / (
                -2.0 * self.lmbd * features.shape[0]
            )

        self.learned_weights = best_weights.T.contiguous()
        self.coefficients = coefficients.contiguous()
        self._active_indices = torch.ones(
            features.shape[0], dtype=torch.bool, device=self.device
        )

        os.makedirs(os.path.join(self.cache_dir, self.name), exist_ok=True)

        torch.save(
            self.learned_weights.cpu(),
            os.path.join(self.cache_dir, self.name, "weights"),
        )
        torch.save(
            self.coefficients.cpu(),
            os.path.join(self.cache_dir, self.name, "coefficients"),
        )
        torch.save(
            self._active_indices.cpu(),
            os.path.join(self.cache_dir, self.name, "active_indices"),
        )
        torch.save(
            self.samples.cpu(),
            os.path.join(self.cache_dir, self.name, "samples"),
        )
        torch.save(
            self.labels.cpu(),
            os.path.join(self.cache_dir, self.name, "labels"),
        )

        self.train_time = torch.tensor(time.time() - tstart, device=self.device)
        torch.save(
            self.train_time.cpu(),
            os.path.join(self.cache_dir, self.name, "train_time"),
        )
        return self.train_time
