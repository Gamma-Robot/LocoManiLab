"""locomanilab/runtime/sim_client.py: the observation.state vector per backend.

Needs g1_bridge + sim2real (the sim conda env); skipped elsewhere. Uses a fake
composite so no simulator is started.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("g1_bridge")
pytest.importorskip("sim2real")

from locomanilab.runtime import sim_client as client  # noqa: E402


class _Backend:
    def torso_command_state(self):
        return np.array([0.01, 0.02, 0.03]), 0.75


class _Session:
    backend = _Backend()


class _Composite:
    session = _Session()

    def read_joint_state(self):
        return np.arange(29, dtype=np.float64) * 0.1, np.zeros(29)

    def read_hand_state(self):
        return {"left": np.full(7, 10.0), "right": np.full(7, 20.0)}


OBS = {"robot0_base_quat": np.array([0.0, 0.0, 0.0, 1.0])}   # xyzw identity


def test_sonic_keeps_the_43_vector_and_default_slices():
    q, slices, extra = client.state_observation(_Composite(), OBS, "sonic")
    assert q.shape == (43,) and slices is None
    assert np.allclose(q[22:29], 10.0) and np.allclose(q[36:43], 20.0)   # fingers interleaved
    assert {"rpy", "height", "root_orientation"} <= set(extra)


@pytest.mark.parametrize("backend", ["heft", "scalebfm", "almi"])
def test_tracking_backends_declare_the_same_43_layout(backend):
    from sim2real.vla.observation import G1_43DOF_STATE_SLICES
    q, slices, extra = client.state_observation(_Composite(), OBS, backend)
    assert slices == dict(G1_43DOF_STATE_SLICES) and q.shape == (43,)
    assert "rpy" in extra and "root_orientation" in extra


def test_amo_state_is_the_32_vector_in_layout_order():
    q, slices, extra = client.state_observation(_Composite(), OBS, "amo")
    assert list(slices) == ["left_hand", "right_hand", "left_arm", "right_arm", "rpy", "height"]
    assert q.shape == (32,)
    body = np.arange(29) * 0.1
    assert np.allclose(q[0:7], 10.0) and np.allclose(q[7:14], 20.0)
    assert np.allclose(q[14:21], body[15:22]) and np.allclose(q[21:28], body[22:29])
    assert np.allclose(q[28:31], [0.01, 0.02, 0.03]) and q[31] == pytest.approx(0.75)
    assert "rpy" not in extra and "height" not in extra and "root_orientation" in extra


def test_build_observation_round_trip_for_amo():
    from sim2real.vla import build_observation
    q, slices, extra = client.state_observation(_Composite(), OBS, "amo")
    obs = build_observation(joint_pos=q, state_slices=slices, images={"ego_view": np.zeros((4, 4, 3), np.uint8)},
                            prompt="t", base_quat_wxyz=np.array([1.0, 0, 0, 0]), extra_state=extra)
    st = obs["state"]
    assert st["left_arm"].shape == (1, 1, 7) and st["height"].shape == (1, 1, 1)
    assert np.allclose(st["rpy"].reshape(-1), [0.01, 0.02, 0.03])
    assert "left_leg" not in st                     # nothing the amo layout does not declare
