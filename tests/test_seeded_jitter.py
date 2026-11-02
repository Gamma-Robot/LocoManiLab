"""Seeded reset streams must survive different earlier episode RNG consumption."""
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from locomanilab.runtime import sim_client as client


def test_twenty_episode_seeds_are_independent_and_repeatable(monkeypatch):
    draws = []
    def jitter(base, *, rng, **kwargs):
        draws.append(rng.uniform(size=4))
        return True
    module = types.ModuleType('robocasa.utils.scene_initial_jitter')
    module.apply_initial_pose_jitter = jitter
    monkeypatch.setitem(sys.modules, module.__name__, module)
    args = SimpleNamespace(reset_jitter=.05, reset_object_jitter=.02, jitter_seed=20260912)
    base = SimpleNamespace(_get_observations=lambda: {'fresh': True})
    def reset(ep):
        draws.clear()
        assert client._apply_reset_jitter(None, base, args, {}, episode=ep) == {'fresh': True}
        return np.concatenate(draws)
    first = [reset(ep) for ep in range(20)]
    reset(999)  # unrelated episode must not move the streams
    for ep in reversed(range(20)):
        np.testing.assert_array_equal(reset(ep), first[ep])
    assert len({row.tobytes() for row in first}) == 20


def test_legacy_stream_advances_and_seeded_calls_do_not_consume_it(monkeypatch):
    draws = []
    module = types.ModuleType('robocasa.utils.scene_initial_jitter')
    module.apply_initial_pose_jitter = lambda base, rng, **kw: draws.append(rng.random()) or True
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(client, '_JITTER_RNG', np.random.default_rng(12))
    args = SimpleNamespace(reset_jitter=.05, reset_object_jitter=0, jitter_seed=None)
    client._apply_reset_jitter(None, object(), args, {})
    args.jitter_seed = 0
    client._apply_reset_jitter(None, object(), args, {})
    args.jitter_seed = None
    client._apply_reset_jitter(None, object(), args, {})
    np.testing.assert_array_equal([draws[0], draws[2]], np.random.default_rng(12).random(2))


def test_seed_parse_and_negative_rejection():
    assert client.parse_args([]).jitter_seed is None
    assert client.parse_args(['--jitter-seed', '0']).jitter_seed == 0
    with pytest.raises(SystemExit):
        client.parse_args(['--jitter-seed', '-1'])


def test_physics_digest_detects_changed_initial_pose():
    data = SimpleNamespace(**{k: np.zeros(2) for k in ('qpos', 'qvel', 'act', 'ctrl', 'mocap_pos', 'mocap_quat')})
    base = SimpleNamespace(sim=SimpleNamespace(data=data))
    before = client._initial_state_digest(base)
    assert client._initial_state_digest(base) == before
    data.qpos[0] = .001
    assert client._initial_state_digest(base) != before
