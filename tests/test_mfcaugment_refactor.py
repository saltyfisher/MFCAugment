from types import SimpleNamespace
from pathlib import Path
import sys
import types

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core import MFCAugment as mfc


def test_build_search_bounds_without_probability_dimension():
    lb, ub, n_dims, var_dim = mfc.build_search_bounds(
        total_op_num=16,
        num_ops=2,
        mag_bin=31,
        prob_bin=10,
        use_prob=False,
    )

    assert n_dims == 2
    assert var_dim == 4
    assert lb.tolist() == [0, 0, 0, 0]
    assert ub.tolist() == [15, 15, 30, 30]


def test_build_search_bounds_with_probability_dimension():
    lb, ub, n_dims, var_dim = mfc.build_search_bounds(
        total_op_num=16,
        num_ops=2,
        mag_bin=31,
        prob_bin=10,
        use_prob=True,
    )

    assert n_dims == 3
    assert var_dim == 6
    assert lb.tolist() == [0, 0, 0, 0, 0, 0]
    assert ub.tolist() == [15, 15, 30, 30, 9, 9]


def test_build_search_tasks_selects_eval_function_by_mode():
    lb = np.array([0, 0, 0, 0])
    ub = np.array([1, 1, 1, 1])
    groups = [np.array([0, 1]), np.array([2, 3])]

    normal_tasks = mfc.build_search_tasks(groups, num_ops=2, n_dims=2, lb=lb, ub=ub, use_bayes=False)
    bayes_tasks = mfc.build_search_tasks(groups, num_ops=2, n_dims=2, lb=lb, ub=ub, use_bayes=True)

    assert [task.dims for task in normal_tasks] == [4, 4]
    assert all(task.evalfnc is mfc.evalFunc for task in normal_tasks)
    assert all(task.evalfnc is mfc.evalFuncBayes for task in bayes_tasks)


def test_combine_feature_batches_matches_gpu_and_cpu_paths():
    feature_batches = [torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, 4.0]])]
    class_batches = [torch.tensor([[0.8, 0.2]]), torch.tensor([[0.1, 0.9]])]

    cpu_features, cpu_classes = mfc.combine_feature_batches(feature_batches, class_batches, use_gpu=False)
    gpu_features, gpu_classes = mfc.combine_feature_batches(feature_batches, class_batches, use_gpu=True)

    assert isinstance(cpu_features, np.ndarray)
    assert cpu_features.tolist() == [[1.0, 2.0], [3.0, 4.0]]
    np.testing.assert_allclose(cpu_classes, [[0.8, 0.2], [0.1, 0.9]])
    assert torch.equal(gpu_features, torch.cat(feature_batches, dim=0))
    assert torch.equal(gpu_classes, torch.cat(class_batches, dim=0))


def test_get_pca_component_count_matches_dataset_rule():
    assert mfc.get_pca_component_count("breakhis", feature_dim=200) == 10
    assert mfc.get_pca_component_count("chestct", feature_dim=200) == 2


def test_build_mfc_params_keeps_expected_contract():
    args = SimpleNamespace(batch_size=8)
    lb = np.array([0, 0])
    ub = np.array([1, 1])
    groups = [np.array([0, 1])]
    params = mfc.build_mfc_params(
        model="model",
        data_list=["a", "b"],
        feat_list=np.array([[1.0], [2.0]]),
        groups=groups,
        centers=[],
        pca="pca",
        lb=lb,
        ub=ub,
        args=args,
        resize_size=(224, 224),
        num_ops=1,
        mag_bin=31,
        prob_bin=10,
    )

    assert params["feat_extractor"] == "model"
    assert params["data_list"] == ["a", "b"]
    assert params["groups"] == groups
    assert params["Lb"] is lb
    assert params["Ub"] is ub
    assert params["n_op"] == 1
    assert params["batch_size"] == 8


def test_representative_group_indices_keep_small_groups_complete():
    feat_list = np.arange(12, dtype=float).reshape(6, 2)
    groups = [np.array([0, 2, 4])]

    sampled_groups = mfc.build_representative_groups(
        feat_list,
        groups,
        sample_ratio=0.99,
    )

    assert len(sampled_groups) == 1
    assert sampled_groups[0].tolist() == [0, 2, 4]


def test_representative_group_indices_mix_center_middle_and_boundary():
    feat_list = np.arange(10, dtype=float).reshape(10, 1)
    groups = [np.arange(10)]

    sampled_groups = mfc.build_representative_groups(
        feat_list,
        groups,
        sample_ratio=0.5,
    )

    assert len(sampled_groups[0]) == 5
    assert set(sampled_groups[0].tolist()) == {2, 4, 5, 7, 9}


def test_build_mfc_params_includes_representative_groups_when_requested():
    args = SimpleNamespace(batch_size=8, mfc_eval_sample_ratio=0.5)
    groups = [np.arange(10)]
    params = mfc.build_mfc_params(
        model="model",
        data_list=list(range(10)),
        feat_list=np.arange(10, dtype=float).reshape(10, 1),
        groups=groups,
        centers=[],
        pca="pca",
        lb=np.array([0]),
        ub=np.array([1]),
        args=args,
        resize_size=(224, 224),
        num_ops=1,
        mag_bin=31,
        prob_bin=10,
    )

    assert params["eval_groups"][0].tolist() == [4, 5, 2, 7, 9]
    assert params["full_groups"] == groups


def test_process_policy_loads_augmentation_and_loss_dependencies(monkeypatch):
    class FakeAugment:
        def __init__(self, policy, num_ops):
            self.policy = policy
            self.num_ops = num_ops

        def __call__(self, data):
            return data

    class FakePCA:
        def transform(self, value):
            return value

    augmentations = types.ModuleType("core.augmentations")
    augmentations.MyAugment = FakeAugment
    utils = types.ModuleType("core.utils")
    utils.KL_loss = lambda p, q: float(np.asarray(q).sum())
    utils.kl_divergence_multivariate_torch = lambda p, q: torch.as_tensor(q).sum()

    monkeypatch.setitem(sys.modules, "core.augmentations", augmentations)
    monkeypatch.setitem(sys.modules, "core.utils", utils)
    monkeypatch.setattr(
        mfc,
        "getdatafeat",
        lambda args, resize_size, data_list, model: ([torch.tensor([[1.0, 2.0]])], None),
    )

    args = SimpleNamespace(gpu=False, device="cpu", resize=True)
    loss = mfc.process_policy(
        (
            {"op_index": np.array([[0]])},
            [torch.tensor([1.0])],
            args,
            1,
            (224, 224),
            "model",
            FakePCA(),
            np.array([[0.0, 0.0], [3.0, 4.0]]),
            [np.array([0])],
            0,
        )
    )

    assert loss == 0.0


def test_eval_func_bayes_uses_representative_eval_groups(monkeypatch):
    class FakeAugment:
        def __init__(self, policy, num_ops):
            self.policy = policy
            self.num_ops = num_ops

        def __call__(self, data):
            return data

    class FakePCA:
        def transform(self, value):
            return value

    augmentations = types.ModuleType("core.augmentations")
    augmentations.MyAugment = FakeAugment
    utils = types.ModuleType("core.utils")
    utils.KL_loss = lambda p, q: 0.0
    utils.kl_divergence_multivariate_torch = lambda p, q: 0.0
    seen_values = []

    def fake_getdatafeat(args, resize_size, data_list, model):
        seen_values.extend(float(item.item()) for item in data_list)
        return [torch.ones(len(data_list), 2)], None

    monkeypatch.setitem(sys.modules, "core.augmentations", augmentations)
    monkeypatch.setitem(sys.modules, "core.utils", utils)
    monkeypatch.setattr(mfc, "getdatafeat", fake_getdatafeat)

    args = SimpleNamespace(gpu=False, device="cpu", resize=True, l=1)
    params = {
        "feat_extractor": "model",
        "data_list": [torch.tensor(value) for value in [0.0, 0.25, 0.5, 0.75]],
        "feat_list": np.ones((4, 2)),
        "batch_size": 2,
        "groups": [np.array([0, 1, 2, 3])],
        "eval_groups": [np.array([1, 3])],
        "pca": FakePCA(),
        "w": 0,
        "task_id": 0,
        "Lb": np.array([0]),
        "Ub": np.array([1]),
        "args": args,
        "model": "model",
        "resize_size": (224, 224),
        "n_op": 1,
    }

    loss = mfc.evalFuncBayes({"op_index": np.array([[0]])}, params)

    assert loss == 0.0
    np.testing.assert_allclose(seen_values, [63 / 255, 191 / 255])


def test_build_task_params_copies_base_params_without_mutating_source():
    base_params = {
        "groups": [np.array([0, 1])],
        "eval_groups": [np.array([1])],
    }

    task_params = mfc.build_task_params(base_params, task_id=2)

    assert task_params is not base_params
    assert task_params["task_id"] == 2
    assert "task_id" not in base_params
    assert task_params["groups"] is base_params["groups"]


def test_reevaluate_top_policies_uses_full_groups_and_restores_eval_groups(monkeypatch):
    calls = []
    policy_a = {"op_index": np.array([[0]]), "magnitude_index": np.array([[0]]), "prob_index": []}
    policy_b = {"op_index": np.array([[1]]), "magnitude_index": np.array([[1]]), "prob_index": []}
    params = {
        "task_id": 0,
        "groups": [np.array([0, 1, 2, 3])],
        "eval_groups": [np.array([1, 3])],
    }

    def fake_evaluate(policy, eval_params):
        calls.append((policy, eval_params["eval_groups"][0].copy()))
        return 1.0 if policy is policy_a else 0.25

    monkeypatch.setattr(mfc, "evalFuncBayes", fake_evaluate)

    selected = mfc.reevaluate_top_policies_with_full_groups(
        [{"policy": policy_a, "loss": 0.1}, {"policy": policy_b, "loss": 0.2}],
        params,
        topk=2,
    )

    assert selected == [{"policy": policy_b, "loss": 0.25}, {"policy": policy_a, "loss": 1.0}]
    assert params["eval_groups"][0].tolist() == [1, 3]
    assert [group.tolist() for _, group in calls] == [[0, 1, 2, 3], [0, 1, 2, 3]]


def test_bayesian_parallel_passes_independent_params_per_task(monkeypatch):
    seen = []

    def fake_single_task(task_idx, args, task, params, rep, topk, max_evals):
        seen.append((task_idx, id(params), params["task_id"]))
        params["local_mutation"] = task_idx
        return {"task": task_idx}, task_idx, 0.0, float(task_idx)

    monkeypatch.setattr(mfc, "bayesian_optimization_single_task", fake_single_task)

    args = SimpleNamespace()
    params = {"groups": [np.array([0]), np.array([1])]}
    results = mfc.bayesian_optimization_tasks_parallel(
        tasks=["task0", "task1"],
        args=args,
        params=params,
        rep=1,
        topk=1,
        max_evals=1,
    )

    assert results == [{"task": 0}, {"task": 1}]
    assert [entry[0] for entry in seen] == [0, 1]
    assert [entry[2] for entry in seen] == [0, 1]
    assert len({entry[1] for entry in seen}) == 2
    assert "task_id" not in params
    assert "local_mutation" not in params


def test_merge_trial_policies_keeps_two_dimensional_rows_for_topk_one():
    trial_history = [{
        "policy": {
            "op_index": np.array([[1, 2]]),
            "magnitude_index": np.array([[3, 4]]),
            "prob_index": [],
        },
        "loss": 0.5,
    }]

    merged = mfc.merge_trial_policies(trial_history, use_prob=False)

    assert merged["op_index"].shape == (1, 2)
    assert len(merged["magnitude_index"]) == 1
    assert merged["magnitude_index"][0].shape == (1, 2)
    assert merged["magnitude_index"][0].tolist() == [[3, 4]]


def test_close_policy_writers_closes_every_writer():
    class FakeWriter:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    writers = [FakeWriter(), FakeWriter()]

    mfc.close_policy_writers(writers)

    assert all(writer.closed for writer in writers)


def test_sample_weight_centers_uses_one_center_unless_diff_c_enabled(monkeypatch):
    weights = np.array([0.1, 0.2, 0.3, 0.4])

    monkeypatch.setattr(np.random, "choice", lambda size, count, p: np.arange(count))

    shared = mfc.sample_weight_centers(weights, center_count=3, diff_c=False)
    distinct = mfc.sample_weight_centers(weights, center_count=3, diff_c=True)

    assert shared.tolist() == [0.1, 0.1, 0.1]
    assert distinct.tolist() == [0.1, 0.2, 0.3]
