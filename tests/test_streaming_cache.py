import json
from pathlib import Path
import polars as pl
import pytest
from nvision.cache.data_store import CategoryDataStore
from nvision.cache.locator_repository import LocatorResultsRepository, STREAMING_REPEAT_THRESHOLD
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

def test_locator_repository_threshold_inline(repo):
    # Threshold is 5. 3 repeats should be inline.
    results = [([{"p": f"p{i}"}], {"idx": i}) for i in range(3)]
    repo.save_cached_combination(
        generator="gen", noise="noise", strategy="strat",
        repeats=3, seed=1, max_steps=10, timeout_s=10,
        results=results
    )
    
    # Verify it's inline (no streaming pointer)
    # We can check by seeing if load_df on the base config key returns None
    from nvision.cache.locator_keys import combination_base_cache_config
    from nvision.cache.hashing import stable_config_hash
    
    ptr_cfg = combination_base_cache_config(
        generator="gen", noise="noise", strategy="strat",
        seed=1, max_steps=10, timeout_s=10
    )
    assert repo._store.load_df(stable_config_hash(ptr_cfg)) is None
    
    # But it should load normally
    loaded = repo.get_cached_combination(
        generator="gen", noise="noise", strategy="strat",
        repeats=3, seed=1, max_steps=10, timeout_s=10
    )
    assert len(loaded) == 3

def test_locator_repository_threshold_streaming(repo):
    # Threshold is 5. 10 repeats should be streaming.
    count = 10
    results = [([{"p": f"p{i}"}], {"idx": i}) for i in range(count)]
    repo.save_cached_combination(
        generator="gen", noise="noise", strategy="strat",
        repeats=count, seed=1, max_steps=10, timeout_s=10,
        results=results
    )
    
    # Verify streaming pointer exists
    from nvision.cache.locator_keys import combination_base_cache_config
    from nvision.cache.hashing import stable_config_hash
    
    ptr_cfg = combination_base_cache_config(
        generator="gen", noise="noise", strategy="strat",
        seed=1, max_steps=10, timeout_s=10
    )
    ptr_key = stable_config_hash(ptr_cfg)
    ptr_df = repo._store.load_df(ptr_key)
    assert ptr_df is not None
    assert int(ptr_df.get_column("achieved_repeats")[0]) == count
    
    # Load partial
    partial, n = repo.get_cached_combination_partial(
        generator="gen", noise="noise", strategy="strat",
        repeats=count, seed=1, max_steps=10, timeout_s=10
    )
    assert n == count
    assert len(partial) == count

def test_append_repeats(repo):
    count1 = 3
    results1 = [([{"p": f"p{i}"}], {"idx": i}) for i in range(count1)]
    
    # Manually trigger streaming start or just use append
    repo.append_cached_repeats(
        generator="gen", noise="noise", strategy="strat",
        seed=1, max_steps=10, timeout_s=10,
        new_results=results1, start_idx=0
    )
    
    # Append more
    count2 = 2
    results2 = [([{"p": f"p{i}"}], {"idx": i}) for i in range(count1, count1+count2)]
    repo.append_cached_repeats(
        generator="gen", noise="noise", strategy="strat",
        seed=1, max_steps=10, timeout_s=10,
        new_results=results2, start_idx=count1
    )
    
    # Load all
    partial, n = repo.get_cached_combination_partial(
        generator="gen", noise="noise", strategy="strat",
        repeats=10, seed=1, max_steps=10, timeout_s=10
    )
    assert n == count1 + count2
    assert partial[4][1]["idx"] == 4
