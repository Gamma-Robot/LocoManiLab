"""sim_task_overrides.yaml and the class patching sim_client applies before building an env."""
import xml.etree.ElementTree as ET

import pytest
import yaml

from locomanilab.runtime import script_path
from locomanilab.runtime import sim_client as client


def test_override_table_is_shipped_and_well_formed():
    path = script_path("sim_task_overrides.yaml")
    assert path == client.TASK_OVERRIDES
    table = yaml.safe_load(path.read_text())
    assert table
    for env_name, scenes in table.items():
        assert env_name.isidentifier()
        for scene_id, entry in scenes.items():
            assert isinstance(scene_id, int)
            assert entry and set(entry) <= client._OVERRIDE_KEYS
            for fixture in entry.get("scene_fixtures") or ():
                assert fixture["file"].endswith(".xml")


def test_overrides_need_a_pinned_scene(tmp_path):
    path = tmp_path / "overrides.yaml"
    path.write_text("Task:\n  7:\n    fixed_attrs: {X: 1}\n")
    assert client.load_task_overrides("Task", None, path) is None
    assert client.load_task_overrides("Task", 8, path) is None
    assert client.load_task_overrides("Other", 7, path) is None
    assert client.load_task_overrides("Task", 7, path) == {"fixed_attrs": {"X": 1}}


def _classes():
    class Base:
        RADIUS = 0.1
        SCENE_SPECS = {"FloorPlan7": {"a": 1, "b": 2}, "FloorPlan8": {"a": 3}}

        def constrain(self):
            return "base"

    class Task(Base):
        def constrain(self):
            return "task"

        def finalize(self):
            self.RADIUS = 0.0

    return Base, Task


def test_override_task_class_patches_only_the_task():
    Base, Task = _classes()
    entry = {"inherit_methods": ["constrain"], "fixed_attrs": {"RADIUS": 0.1},
             "scene_spec": {"b": 5}}
    applied = client.override_task_class(Task, 7, entry)
    assert len(applied) == 3
    task = Task()
    assert task.constrain() == "base"
    task.finalize()  # the task's own write is ignored
    assert task.RADIUS == 0.1
    assert Task.SCENE_SPECS["FloorPlan7"] == {"a": 1, "b": 5}
    assert Base.SCENE_SPECS["FloorPlan7"] == {"a": 1, "b": 2}  # parent table untouched
    assert client.override_task_class(Task, 7, entry) == []  # idempotent


def test_override_task_class_rejects_unknown_or_unbacked_settings():
    _, Task = _classes()
    with pytest.raises(ValueError, match="unknown"):
        client.override_task_class(Task, 7, {"typo": 1})
    with pytest.raises(ValueError, match="does not define"):
        client.override_task_class(Task, 7, {"inherit_methods": ["missing"]})
    with pytest.raises(ValueError, match="no entry for scene"):
        client.override_task_class(Task, 9, {"scene_spec": {"a": 0}})


def test_insert_children_keeps_source_body_order():
    parent = ET.fromstring('<worldbody><body name="a"/><body name="vase"/><body name="c"/></worldbody>')
    source = ET.fromstring('<worldbody><body name="box"/><body name="lid"/></worldbody>')
    client.insert_children(parent, source, "vase")
    assert [el.get("name") for el in parent] == ["a", "box", "lid", "vase", "c"]
    contact = ET.fromstring('<contact><exclude name="x"/></contact>')
    client.insert_children(contact, ET.fromstring('<contact><exclude name="p0"/></contact>'), None, at_start=True)
    assert [el.get("name") for el in contact] == ["p0", "x"]
    with pytest.raises(ValueError):
        client.insert_children(parent, source, "missing")


def test_parse_args_keeps_overrides_on_by_default():
    assert client.parse_args([]).no_task_overrides is False
    assert client.parse_args(["--no-task-overrides"]).no_task_overrides is True
