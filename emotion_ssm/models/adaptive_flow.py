"""State-dependent flow in the observer's fixed affect coordinates.

The low-rank skew operator mixes affect coordinates without changing their
definition. Directed, bounded messages continuously couple the two roles;
fresh events/actions remain separate inputs in state_core. No future evidence
or stochastic layers are used by this autonomous vector field.
"""
from __future__ import annotations

from dataclasses import replace
import math

import torch
from torch import nn


class AdaptiveDyadicFlow(nn.Module):
    REVISION = "adaptive_dyadic_v1"

    def __init__(self, dimension, relation_dim, hidden_dim, rank=8,
                 max_cross_rate=.05, max_feedback=2.):
        super().__init__()
        if rank < 1 or min(max_cross_rate, max_feedback) <= 0:
            raise ValueError("Adaptive flow rank and bounds must be positive")
        self.dimension, self.rank = int(dimension), int(rank)
        self.max_cross_rate, self.max_feedback = float(max_cross_rate), float(max_feedback)
        self.left = nn.Parameter(torch.randn(2*dimension, rank) / math.sqrt(2*dimension*rank))
        self.right = nn.Parameter(torch.randn(2*dimension, rank) / math.sqrt(2*dimension*rank))
        # Shared parameters, role-relative inputs: swapping A/B swaps outputs.
        self.conditioner = nn.Sequential(
            nn.Linear(4*dimension+2*relation_dim, hidden_dim, bias=False), nn.Tanh(),
            nn.Linear(hidden_dim, 2*dimension+rank+3, bias=False))
        self.message = nn.Linear(dimension, dimension, bias=False)
        with torch.no_grad():
            self.message.weight.mul_(.1).add_(torch.eye(dimension)*.5)
        self.relation_target = nn.Sequential(
            nn.Linear(4*dimension+2*relation_dim, hidden_dim, bias=False), nn.Tanh(),
            nn.Linear(hidden_dim, relation_dim, bias=False), nn.Tanh())

    def coefficients(self, state, rates, omega, enable_partner=True, diagnostics=None):
        """Return diagonal damping and non-diagonal/feedback terms in dx/dt.

        Damping remains positive with distinct fast/slow ranges. Skew mixing
        has bounded operator norm; partner forcing and relation targets are
        bounded by tanh. Joint energy may increase through feedback, so this
        is an ultimately bounded driven system, not a monotone-decay claim.
        """
        d = self.dimension
        x = torch.cat((state.fast, state.slow), -1)
        other = x.flip(1) if enable_partner else torch.zeros_like(x)
        relation = state.relation if enable_partner else torch.zeros_like(state.relation)
        inputs = torch.cat((x, other, relation, relation.flip(1)), -1)
        dtype = self.conditioner[0].weight.dtype
        control = self.conditioner(inputs.to(dtype)).to(x.dtype)
        fast, slow, rel_rate = rates
        fast = (fast * (.5*control[..., :d].tanh()).exp()).clamp(1/16., 4.)
        slow = (slow * (.5*control[..., d:2*d].tanh()).exp()).clamp(1/1800., 1/30.)
        gates = (control[..., 2*d:2*d+self.rank] + .2).tanh()
        # Frobenius caps also bound the spectral norm, without an SVD per step.
        left = self.left / self.left.norm().clamp_min(1.)
        right = self.right / self.right.norm().clamp_min(1.)
        left, right = left.to(x.dtype), right.to(x.dtype)
        skew = ((x @ left)*gates) @ right.T - ((x @ right)*gates) @ left.T
        skew = .5*self.max_cross_rate*skew
        rotation = omega*(1 + .5*control[..., -3:-2].tanh())
        fast_force = skew[..., :d] + rotation*state.slow
        slow_force = skew[..., d:] - rotation*state.fast
        relation_force = torch.zeros_like(state.relation)
        partner_fast, partner_slow = torch.zeros_like(state.fast), torch.zeros_like(state.slow)
        if enable_partner:
            message = self.message((state.fast+state.slow).flip(1).to(dtype)).tanh().to(x.dtype)
            strength = self.max_feedback*(control[..., -2:]-2.).sigmoid()
            partner_fast = fast*strength[..., :1]*message
            partner_slow = slow*strength[..., 1:]*message
            fast_force = fast_force + partner_fast
            slow_force = slow_force + partner_slow
            relation_force = rel_rate*self.relation_target(inputs.to(dtype)).to(x.dtype)
        if diagnostics is not None:
            diagnostics.update(cross_coordinate_drive_norm=skew.norm(dim=-1),
                partner_feedback_norm=torch.cat((partner_fast, partner_slow), -1).norm(dim=-1),
                fast_rate_ratio=(fast/rates[0]).mean(-1), slow_rate_ratio=(slow/rates[1]).mean(-1),
                relation_drive_norm=relation_force.norm(dim=-1))
        return (fast, slow, rel_rate), (fast_force, slow_force, relation_force)

    def derivative(self, state, rates, omega, enable_partner=True):
        decay, force = self.coefficients(state, rates, omega, enable_partner)
        return tuple(-rate*value+drive for rate, value, drive in
                     zip(decay, (state.fast, state.slow, state.relation), force))

    @staticmethod
    def _exponential_step(state, dt, rates, forces):
        h = dt[:, None, None]
        values = []
        for value, rate, force in zip((state.fast, state.slow, state.relation), rates, forces):
            amount = -torch.expm1(-rate*h)
            values.append(value*(1-amount) + (amount/rate)*force)
        return replace(state, fast=values[0], slow=values[1], relation=values[2],
                       elapsed=state.elapsed+dt)

    def propagate(self, state, dt, rates, omega, max_step, enable_partner=True,
                  force=None, relation_force=None):
        """Exponential midpoint, with bounded physical-second substeps.

        Coefficients are recomputed at every midpoint. Exponential damping
        handles short fast time constants without an explicit Euler instability.
        Nonlinear flow has numerical, not exact algebraic, composition accuracy.
        """
        count = math.ceil(float(dt.max().detach())/max_step)
        current, remaining = state, dt

        def coefficients(value):
            decay, drive = self.coefficients(value, rates, omega, enable_partner)
            fast, slow, relation = drive
            if force is not None:
                fast, slow = fast+force[..., 0], slow+force[..., 1]
            if relation_force is not None:
                relation = relation+rates[2]*relation_force
            return decay, (fast, slow, relation)

        for _ in range(count):
            step = remaining.clamp(0, max_step)
            damping, forcing = coefficients(current)
            middle = self._exponential_step(current, step*.5, damping, forcing)
            damping, forcing = coefficients(middle)
            current = self._exponential_step(current, step, damping, forcing)
            remaining = (remaining-step).clamp_min(0)
        # Avoid accumulated floating-point drift in the declared timestamp.
        return replace(current, elapsed=state.elapsed+dt)
