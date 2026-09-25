"""MD task queue + ES compute FIFO; sticky association until task done."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

import numpy as np


STAGE_LOCAL = "local"
STAGE_TX = "tx"
STAGE_EDGE = "edge"
STAGE_DONE = "done"


def draw_task_qos(agent, time_slot=0, user_idx=0, arrival_idx=0):
    """One task's delay limit, energy limit, and weights.

    Each arrival draws its own requirement. Draws are keyed by
    ``(seed, slot, user_idx, arrival_idx)`` so sweep / algo comparisons share
    the same exogenous QoS sequence (see ``omnis.exog``).
    """
    from omnis.exog import draw_qos
    return draw_qos(agent, time_slot, user_idx, arrival_idx)


def task_arrivals(agent, n_by_user, time_slot=0):
    """Poisson counts in, one QoS draw per arrival out."""
    user_index = {u: i for i, u in enumerate(getattr(agent, "users", n_by_user))}
    task_dic = {}
    t = int(time_slot)
    for user, n_arrivals in n_by_user.items():
        n_arrivals = int(n_arrivals)
        ui = int(user_index.get(user, 0))
        draws = [
            draw_task_qos(agent, t, ui, k) for k in range(n_arrivals)
        ]
        # HOL placeholder when idle this slot: still keyed (arrival_idx=-1).
        hol = draws[0] if draws else draw_task_qos(agent, t, ui, -1)
        task_dic[user] = {
            **hol,
            "n_arrivals": n_arrivals,
            "qos_draws": draws,
        }
    return task_dic


@dataclass
class Task:
    user: str
    delay_constraint: float
    energy_constraint: float
    delay_weight: float
    energy_weight: float
    t_arrive: float
    model_name: Optional[str] = None
    cell_id: Optional[int] = None
    payload_bits: float = 0.0
    stage: str = STAGE_LOCAL
    residual: float = 0.0  # seconds (local/edge) or bits (tx)
    local_s: float = 0.0
    local_e: float = 0.0
    tx_s: float = 0.0
    tx_e: float = 0.0
    edge_s: float = 0.0
    edge_e: float = 0.0
    md_wait_s: float = 0.0
    edge_wait_s: float = 0.0
    t_admit: Optional[float] = None
    t_tx_done: Optional[float] = None
    t_edge_start: Optional[float] = None
    t_done: Optional[float] = None
    last_mcs: Optional[int] = None  # MCS in last uplink slot (κ^f)
    # edge_delay_unit = psi / (cores * flops_per_cycle); tau_e = unit / f
    edge_delay_unit: float = 0.0
    edge_power_coeff: float = 0.0
    meta: dict = field(default_factory=dict)

    @property
    def service_delay(self) -> float:
        return float(self.local_s + self.tx_s + self.edge_s)

    @property
    def e2e_delay(self) -> float:
        return float(self.md_wait_s + self.service_delay + self.edge_wait_s)

    @property
    def total_energy(self) -> float:
        return float(self.local_e + self.tx_e + self.edge_e)


class TaskPipeline:
    """Per-MD pending/active tasks and per-cell edge FIFO queues."""

    def __init__(self, users, num_cells: int):
        self.users = list(users)
        self.num_cells = int(num_cells)
        self.pending: Dict[str, Deque[Task]] = {u: deque() for u in self.users}
        self.active: Dict[str, Optional[Task]] = {u: None for u in self.users}
        self.edge_q: Dict[int, Deque[Task]] = {
            c: deque() for c in range(self.num_cells)
        }
        self.energy_queue: Dict[str, float] = {u: 0.0 for u in self.users}
        self.time_s: float = 0.0
        self.completed: List[Task] = []
        self._just_done: List[Task] = []

    def pending_len(self, user: str) -> int:
        return len(self.pending[user])

    def tx_queue_len(self, user: str) -> int:
        """Q^u: pending + in-flight local/TX (not edge)."""
        n = len(self.pending[user])
        at = self.active[user]
        if at is not None and at.stage in (STAGE_LOCAL, STAGE_TX):
            n += 1
        return n

    def compute_queue_len(self, cell_id: int) -> int:
        return len(self.edge_q[int(cell_id)])

    def composite_backlog(self, user: str, cell_id: Optional[int] = None) -> float:
        """Aligned task backlog: Q^{u}_m + Q^{e}_c [tasks].

        ``cell_id`` defaults to the locked serving cell; if none, only Q^{u}.
        """
        q_tx = float(self.tx_queue_len(user))
        if cell_id is None:
            cell_id = self.locked_cell(user)
        if cell_id is None:
            return q_tx
        return q_tx + float(self.compute_queue_len(cell_id))

    def locked_cell(self, user: str) -> Optional[int]:
        task = self.active[user]
        if task is None or task.cell_id is None:
            return None
        return int(task.cell_id)

    def locked_model(self, user: str) -> Optional[str]:
        task = self.active[user]
        return None if task is None else task.model_name

    def is_idle(self, user: str) -> bool:
        return self.active[user] is None

    def needs_radio(self, user: str) -> bool:
        """True iff MD has residual uplink bits (radio-active M^u)."""
        task = self.active[user]
        return (
            task is not None
            and task.stage == STAGE_TX
            and float(task.residual) > 1e-9
        )

    def needs_gpu(self, user: str) -> bool:
        task = self.active[user]
        return task is not None and task.stage == STAGE_EDGE

    def uplink_residual_bits(self, user: str) -> float:
        task = self.active[user]
        if task is None or task.stage != STAGE_TX:
            return 0.0
        return float(max(task.residual, 0.0))

    def local_residual_s(self, user: str) -> float:
        task = self.active[user]
        if task is None or task.stage != STAGE_LOCAL:
            return 0.0
        return float(max(task.residual, 0.0))

    def jobs_ahead(self, user: str, cell_id: Optional[int] = None) -> float:
        """Q^j: edge jobs ahead of the tagged task (paper proxy)."""
        if cell_id is None:
            cell_id = self.locked_cell(user)
        if cell_id is None:
            return 0.0
        q = self.edge_q[int(cell_id)]
        at = self.active.get(user)
        if at is not None and at.stage == STAGE_EDGE and at in q:
            return float(max(0, list(q).index(at)))
        return float(len(q))

    def enqueue_arrivals(self, user: str, n: int, qos: dict):
        for _ in range(int(max(0, n))):
            self.pending[user].append(
                Task(
                    user=user,
                    delay_constraint=float(qos["delay_constraint"]),
                    energy_constraint=float(qos["energy_constraint"]),
                    delay_weight=float(qos["delay_weight"]),
                    energy_weight=float(qos["energy_weight"]),
                    t_arrive=self.time_s,
                )
            )

    def admit(
        self,
        user: str,
        model_name: str,
        cell_id: int,
        payload_bits: float,
        local_delay_s: float,
        local_energy: float,
        edge_delay_unit: float,
        edge_power_coeff: float,
    ) -> Optional[Task]:
        """Admit HOL pending task if MD idle. Locks model/cell until done."""
        if self.active[user] is not None or not self.pending[user]:
            return None
        task = self.pending[user].popleft()
        task.model_name = model_name
        task.cell_id = int(cell_id)
        task.payload_bits = float(payload_bits)
        task.local_s = float(local_delay_s)
        task.local_e = float(local_energy)
        task.edge_delay_unit = float(edge_delay_unit)
        task.edge_power_coeff = float(edge_power_coeff)
        task.md_wait_s = max(0.0, self.time_s - task.t_arrive)
        task.t_admit = self.time_s
        if float(local_delay_s) <= 1e-15:
            task.stage = STAGE_TX
            task.residual = float(payload_bits)
        else:
            task.stage = STAGE_LOCAL
            task.residual = float(local_delay_s)
        self.active[user] = task
        return task

    def _complete(self, task: Task, energy_budget: float):
        if task.t_done is None:
            task.t_done = self.time_s
        task.stage = STAGE_DONE
        user = task.user
        if self.active[user] is task:
            self.active[user] = None
        cell = int(task.cell_id)
        if self.edge_q[cell] and self.edge_q[cell][0] is task:
            self.edge_q[cell].popleft()
        elif task in self.edge_q[cell]:
            self.edge_q[cell].remove(task)
        self.energy_queue[user] = max(
            0.0,
            self.energy_queue[user] + task.total_energy - float(energy_budget),
        )
        self._just_done.append(task)
        self.completed.append(task)

    def advance(
        self,
        dt: float,
        bandwidth: Dict[str, float],
        gpu_hz: Dict[str, float],
        goodput_se: Dict[str, float],
        tx_power: Dict[str, float],
        energy_budget: Dict[str, float],
        default_gpu_hz: float = 0.0,
    ):
        """Advance by ``dt`` seconds. Edge queues are FIFO (one head per cell).

        Within-slot causality: edge service for a task cannot start before its
        uplink completion time in this slot. A mid-slot HOL with gpu_hz=0 is
        granted ``default_gpu_hz`` (full pool) for the remaining time.
        """
        if dt <= 0:
            return
        self._just_done.clear()
        slot_start = float(self.time_s)
        slot_end = slot_start + float(dt)
        f_pool = float(default_gpu_hz) if default_gpu_hz > 0 else max(
            (float(v) for v in gpu_hz.values()), default=0.0)

        # --- MD local + TX ---
        for user in self.users:
            task = self.active[user]
            if task is None:
                continue
            rem = float(dt)
            t_cursor = slot_start
            if task.stage == STAGE_LOCAL and rem > 0:
                step = min(rem, task.residual)
                task.residual -= step
                rem -= step
                t_cursor += step
                if task.residual <= 1e-12:
                    task.residual = 0.0
                    task.stage = STAGE_TX
                    task.residual = float(task.payload_bits)
            if task.stage == STAGE_TX and rem > 0:
                rate = max(
                    float(bandwidth.get(user, 0.0))
                    * float(goodput_se.get(user, 0.0)),
                    1e-12,
                )
                bits = rate * rem
                served = min(bits, task.residual)
                air = served / rate
                task.residual -= served
                task.tx_s += air
                task.tx_e = float(tx_power[user]) * task.tx_s
                t_cursor += air
                if task.residual <= 1e-9:
                    task.residual = 0.0
                    task.t_tx_done = t_cursor
                    task.stage = STAGE_EDGE
                    self.edge_q[int(task.cell_id)].append(task)

        # --- ES compute FIFO with within-slot time cursor ---
        for cell in range(self.num_cells):
            t_cursor = slot_start
            while t_cursor < slot_end - 1e-12 and self.edge_q[cell]:
                head = self.edge_q[cell][0]
                if head.stage != STAGE_EDGE:
                    break
                earliest = slot_start
                if head.t_tx_done is not None:
                    earliest = max(earliest, float(head.t_tx_done))
                if t_cursor < earliest:
                    t_cursor = earliest
                if t_cursor >= slot_end - 1e-12:
                    break

                f = float(gpu_hz.get(head.user, 0.0))
                if f <= 1e-12:
                    f = max(f_pool, 1e-12)
                tau_e = head.edge_delay_unit / f

                if head.t_edge_start is None:
                    head.t_edge_start = t_cursor
                    head.edge_s = tau_e
                    head.residual = tau_e
                    head.edge_e = (
                        head.edge_power_coeff * (f ** 2) * head.edge_delay_unit
                    )
                    if head.t_tx_done is not None:
                        head.edge_wait_s = max(
                            0.0, head.t_edge_start - float(head.t_tx_done)
                        )
                else:
                    if head.edge_s > 1e-12:
                        head.residual *= tau_e / head.edge_s
                    head.edge_s = tau_e
                    head.edge_e = (
                        head.edge_power_coeff * (f ** 2) * head.edge_delay_unit
                    )

                rem_slot = slot_end - t_cursor
                step = min(rem_slot, head.residual)
                head.residual -= step
                t_cursor += step
                if head.residual <= 1e-12:
                    head.t_done = t_cursor
                    self._complete(head, energy_budget[head.user])
                else:
                    break

        self.time_s = slot_end

    def pop_slot_completed(self) -> List[Task]:
        out = list(self._just_done)
        self._just_done.clear()
        return out
