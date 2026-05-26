import pytest

from nvision.cache.data_store import CategoryDataStore
from nvision.cache.locator_repository import LocatorResultsRepository
from nvision.cache.repeats_repository import RepeatsRepository


@pytest.fixture
def store(tmp_path):
    db_path = tmp_path / "test.db"
    return CategoryDataStore(db_path)


@pytest.fixture
def repo(store):
    return LocatorResultsRepository(store)


def test_repeats_repository_roundtrip(store):
    repeats_repo = RepeatsRepository(store)
    combo_key = "test_combo"
    entries = [{"p": "plot1.png"}]
    main_result_row = {"abs_err_x": 0.1}

    repeats_repo.save_repeat(combo_key, 0, entries, main_result_row)

    # Check count
    assert repeats_repo.count_saved(combo_key) == 1

    # Load
    loaded = repeats_repo.load_repeat(combo_key, 0)
    assert loaded[0] == entries
    assert loaded[1] == main_result_row

    # Load multiple
    all_loaded = repeats_repo.load_repeats(combo_key, 1)
    assert len(all_loaded) == 1
    assert all_loaded[0] == (entries, main_result_row)


def test_locator_repository_standardized_pointer(repo):
    # Standardized: all repeats should use the streaming pointer format to prevent duplication
    results = [([{"p": f"p{i}"}], {"idx": i}) for i in range(3)]
    repo.save_cached_combination(
        generator="gen", noise="noise", strategy="strat", repeats=3, seed=1, max_steps=10, timeout_s=10, results=results
    )

    # Verify it has a streaming pointer
    from nvision.cache.hashing import stable_config_hash
    from nvision.cache.locator_keys import combination_base_cache_config

    ptr_cfg = combination_base_cache_config(
        generator="gen", noise="noise", strategy="strat", seed=1, max_steps=10, timeout_s=10
    )
    assert repo._store.load_df(stable_config_hash(ptr_cfg)) is not None

    # But it should load normally
    loaded = repo.get_cached_combination(
        generator="gen", noise="noise", strategy="strat", repeats=3, seed=1, max_steps=10, timeout_s=10
    )
    assert len(loaded) == 3


def test_locator_repository_threshold_streaming(repo):
    # Threshold is 5. 10 repeats should be streaming.
    count = 10
    results = [([{"p": f"p{i}"}], {"idx": i}) for i in range(count)]
    repo.save_cached_combination(
        generator="gen",
        noise="noise",
        strategy="strat",
        repeats=count,
        seed=1,
        max_steps=10,
        timeout_s=10,
        results=results,
    )

    # Verify streaming pointer exists
    from nvision.cache.hashing import stable_config_hash
    from nvision.cache.locator_keys import combination_base_cache_config

    ptr_cfg = combination_base_cache_config(
        generator="gen", noise="noise", strategy="strat", seed=1, max_steps=10, timeout_s=10
    )
    ptr_key = stable_config_hash(ptr_cfg)
    ptr_df = repo._store.load_df(ptr_key)
    assert ptr_df is not None
    assert int(ptr_df.get_column("achieved_repeats")[0]) == count

    # Load partial
    partial, n = repo.get_cached_combination_partial(
        generator="gen", noise="noise", strategy="strat", repeats=count, seed=1, max_steps=10, timeout_s=10
    )
    assert n == count
    assert len(partial) == count


def test_append_repeats(repo):
    count1 = 3
    results1 = [([{"p": f"p{i}"}], {"idx": i}) for i in range(count1)]

    # Manually trigger streaming start or just use append
    repo.append_cached_repeats(
        generator="gen",
        noise="noise",
        strategy="strat",
        seed=1,
        max_steps=10,
        timeout_s=10,
        new_results=results1,
        start_idx=0,
    )

    # Append more
    count2 = 2
    results2 = [([{"p": f"p{i}"}], {"idx": i}) for i in range(count1, count1 + count2)]
    repo.append_cached_repeats(
        generator="gen",
        noise="noise",
        strategy="strat",
        seed=1,
        max_steps=10,
        timeout_s=10,
        new_results=results2,
        start_idx=count1,
    )

    # Load all
    partial, n = repo.get_cached_combination_partial(
        generator="gen", noise="noise", strategy="strat", repeats=10, seed=1, max_steps=10, timeout_s=10
    )
    assert n == count1 + count2
    assert partial[4][1]["idx"] == 4


def test_locator_repository_updated_date(repo):
    results = [([{"p": "p0"}], {"idx": 0})]
    repo.save_cached_combination(
        generator="gen", noise="noise", strategy="strat", repeats=1, seed=1, max_steps=10, timeout_s=10, results=results
    )

    from nvision.cache.hashing import stable_config_hash
    from nvision.cache.locator_keys import combination_base_cache_config

    ptr_cfg = combination_base_cache_config(
        generator="gen", noise="noise", strategy="strat", seed=1, max_steps=10, timeout_s=10
    )
    ptr_key = stable_config_hash(ptr_cfg)
    payload = repo._store._backend.get(ptr_key)
    assert payload is not None
    assert "updated_at" in payload
    assert isinstance(payload["updated_at"], str)
    assert len(payload["updated_at"]) > 0


def test_locator_repository_repeat_offset(repo):
    # Sub-tasks all share a single streaming pointer (repeat_offset always 0 in ptr_key).
    # save_cached_combination with repeat_offset=N and start_idx=N lands data at global index N.

    # Simulate 10 repeats already in cache (0..9)
    results_initial = [([{"p": f"p{i}"}], {"idx": i}) for i in range(10)]
    repo.save_cached_combination(
        generator="gen",
        noise="noise",
        strategy="strat",
        repeats=10,
        seed=1,
        max_steps=10,
        timeout_s=10,
        repeat_offset=0,
        results=results_initial,
        start_idx=0,
    )

    # Pointer should now be at 10
    from nvision.cache.hashing import stable_config_hash
    from nvision.cache.locator_keys import combination_base_cache_config

    ptr_cfg = combination_base_cache_config(
        generator="gen", noise="noise", strategy="strat", seed=1, max_steps=10, timeout_s=10
    )
    ptr_key = stable_config_hash(ptr_cfg)
    ptr_df = repo._store.load_df(ptr_key)
    assert int(ptr_df.get_column("achieved_repeats")[0]) == 10

    # Simulate a sub-task that owns slice 10..14 (repeat_offset=10, chunk_size=5)
    results_subtask = [([{"p": f"p{10 + i}"}], {"idx": 10 + i}) for i in range(5)]
    repo.save_cached_combination(
        generator="gen",
        noise="noise",
        strategy="strat",
        repeats=15,
        seed=1,
        max_steps=10,
        timeout_s=10,
        repeat_offset=0,
        results=results_subtask,
        start_idx=10,  # global start index for new results
    )

    # Pointer should now be at 15
    ptr_df = repo._store.load_df(ptr_key)
    assert int(ptr_df.get_column("achieved_repeats")[0]) == 15

    # Sub-task partial load with chunk_size=5 should return exactly 5 items from offset 10
    partial, n = repo.get_cached_combination_partial(
        generator="gen",
        noise="noise",
        strategy="strat",
        repeats=15,
        seed=1,
        max_steps=10,
        timeout_s=10,
        repeat_offset=10,
        chunk_size=5,
    )
    assert n == 5
    assert len(partial) == 5
    assert partial[0][1]["idx"] == 10
    assert partial[4][1]["idx"] == 14

    # Full load of all 15 repeats should work
    loaded = repo.get_cached_combination(
        generator="gen",
        noise="noise",
        strategy="strat",
        repeats=15,
        seed=1,
        max_steps=10,
        timeout_s=10,
        repeat_offset=0,
    )
    assert loaded is not None
    assert len(loaded) == 15
