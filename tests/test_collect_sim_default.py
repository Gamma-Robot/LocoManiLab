"""The simulator collection default is the asset-free teleoperation scene."""


def test_sim_collection_defaults_to_empty_env(plugin, tmp_path):
    result = plugin("gr00t").collect(
        task_prompt="test",
        out_dir=tmp_path / "empty",
        sim=True,
        backend="sonic",
        pico_source="mock",
    )

    assert result.ok
    assert "--env EmptyEnv" in result.command
