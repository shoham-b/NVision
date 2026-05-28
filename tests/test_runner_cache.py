import logging
from pathlib import Path

from nvision.runner.cache import _decompress_content, embed_graph_content


def test_embed_graph_content_happy_path(tmp_path: Path):
    """Test that existing files are read and compressed into the content key."""
    # Setup
    test_file = tmp_path / "test_graph.txt"
    test_content = "This is some test graph content."
    test_file.write_text(test_content, encoding="utf-8")

    entries = [{"path": "test_graph.txt", "other": "data"}]

    # Execute
    result = embed_graph_content(entries, tmp_path)

    # Verify
    assert len(result) == 1
    assert "content" in result[0]
    assert result[0]["path"] == "test_graph.txt"
    assert result[0]["other"] == "data"

    # Decompress to verify content
    decompressed = _decompress_content(result[0])
    assert decompressed == test_content

    # Original should be untouched
    assert "content" not in entries[0]


def test_embed_graph_content_missing_file(tmp_path: Path):
    """Test that missing files are gracefully skipped without error."""
    # Setup
    entries = [{"path": "missing_file.txt"}]

    # Execute
    result = embed_graph_content(entries, tmp_path)

    # Verify
    assert len(result) == 1
    assert "content" not in result[0]
    assert result[0]["path"] == "missing_file.txt"


def test_embed_graph_content_no_path_key(tmp_path: Path):
    """Test that entries without a path key are skipped."""
    # Setup
    entries = [{"no_path": "here"}]

    # Execute
    result = embed_graph_content(entries, tmp_path)

    # Verify
    assert len(result) == 1
    assert "content" not in result[0]
    assert result[0]["no_path"] == "here"


def test_embed_graph_content_read_exception(tmp_path: Path, caplog, monkeypatch):
    """Test that file read exceptions are caught and logged as warnings."""
    # Setup
    test_file = tmp_path / "error_file.txt"
    test_file.write_text("content", encoding="utf-8")

    entries = [{"path": "error_file.txt"}]

    # Force read_text to raise an exception
    def mock_read_text(*args, **kwargs):
        raise PermissionError("Access denied")

    monkeypatch.setattr(Path, "read_text", mock_read_text)

    # Execute
    with caplog.at_level(logging.WARNING):
        result = embed_graph_content(entries, tmp_path)

    # Verify
    assert len(result) == 1
    assert "content" not in result[0]

    # Check logs
    assert "Failed to read graph content for caching" in caplog.text
    assert "Access denied" in caplog.text


def test_harvest_partial_results_from_cache(tmp_path: Path):
    import logging

    from nvision.cache import CacheBridge
    from nvision.cli.run import _harvest_partial_results_from_cache
    from nvision.models.task import LocatorTask
    from nvision.noises.over_frequency.gaussian_noise import OverFrequencyGaussianNoise
    from nvision.sim.combinations import Combination
    from nvision.sim.locs.bayesian.sobol_bayesian_locator import SimpleSobolBayesianLocator
    from nvision.spectra.lorentzian import LorentzianModel

    sig = LorentzianModel()
    noise = OverFrequencyGaussianNoise(0.01)
    combo = Combination(
        generator=sig,
        noise=noise,
        strategy=SimpleSobolBayesianLocator,
        generator_name="NVCenter-lorentzian",
        noise_name="Gauss(0.01)",
        strategy_name="Bayesian-SBED",
    )

    task = LocatorTask(
        combination=combo,
        repeats=1,
        seed=1,
        slug="test_slug",
        out_dir=tmp_path / "out",
        scans_dir=tmp_path / "scans",
        bayes_dir=tmp_path / "bayes",
        loc_max_steps=10,
        sweep_max_steps=None,
        loc_timeout_s=10,
        use_cache=True,
        cache_dir=tmp_path / "cache",
        log_queue=None,
        log_level=logging.INFO,
        ignore_cache_strategy=None,
        repeat_offset=0,
    )

    # 2. Write a result directly to the cache
    bridge = CacheBridge(task.cache_dir)
    try:
        repo = bridge.get_cache_for_category("NVCenter")
        results = [
            (
                [{"path": "p0.png"}],
                {
                    "generator": "NVCenter-lorentzian",
                    "noise": "Gauss(0.01)",
                    "strategy": "Bayesian-SBED",
                    "seed": 1,
                    "attempt": 1,
                },
            )
        ]
        repo.save_cached_combination(
            generator="NVCenter-lorentzian",
            noise="Gauss(0.01)",
            strategy="Bayesian-SBED",
            repeats=1,
            seed=1,
            max_steps=10,
            timeout_s=10,
            repeat_offset=0,
            results=results,
        )
    finally:
        bridge.close()

    # 3. Call _harvest_partial_results_from_cache with cache_bridge=None
    df_rows = []
    plot_manifest = []
    logger = logging.getLogger("test_harvest")

    _harvest_partial_results_from_cache(
        tasks=[task], cache_bridge=None, df_rows=df_rows, plot_manifest=plot_manifest, log=logger
    )

    # 4. Verify results were successfully harvested!
    assert len(df_rows) == 1
    assert df_rows[0]["attempt"] == 1
    assert len(plot_manifest) == 1
    assert plot_manifest[0]["path"] == "p0.png"


def test_task_runner_resume_and_override(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("nvision.runner.executor.generate_attempt_plots", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        "nvision.runner.executor._TaskRunner._run_sobol_baseline",
        lambda *args, **kwargs: {
            "sobol_baseline_steps": 1,
            "sobol_freq_steps": 1,
            "sobol_freq_uncert_at_conv": 0.0,
            "sobol_freq_err_at_conv": 0.0,
            "sobol_xs": [],
            "sobol_ys": [],
            "sobol_mode_estimates": {},
        },
    )

    import logging

    from nvision import SimpleSweepLocator
    from nvision.cache import CacheBridge
    from nvision.models.noise import CompositeNoise, CompositeOverFrequencyNoise
    from nvision.models.task import LocatorTask
    from nvision.noises import OverFrequencyGaussianNoise
    from nvision.runner.executor import _TaskRunner
    from nvision.sim.combinations import Combination
    from nvision.sim.gen.nv_center_generator import NVCenterCoreGenerator

    sig = NVCenterCoreGenerator(x_min=2.6e9, x_max=3.1e9, variant="lorentzian")
    noise = CompositeNoise(over_frequency_noise=CompositeOverFrequencyNoise([OverFrequencyGaussianNoise(0.01)]))
    combo = Combination(
        generator=sig,
        noise=noise,
        strategy=SimpleSweepLocator,
        generator_name="NVCenter-lorentzian",
        noise_name="Gauss(0.01)",
        strategy_name="SimpleSweep",
    )

    # 1. First run: run 3 repeats
    task1 = LocatorTask(
        combination=combo,
        repeats=3,
        seed=1,
        slug="test_slug",
        out_dir=tmp_path / "out",
        scans_dir=tmp_path / "out/scans",
        bayes_dir=tmp_path / "out/bayes",
        loc_max_steps=10,
        sweep_max_steps=10,
        loc_timeout_s=10,
        use_cache=True,
        cache_dir=tmp_path / "cache",
        log_queue=None,
        log_level=logging.INFO,
        ignore_cache_strategy=None,
        repeat_offset=0,
    )

    runner1 = _TaskRunner(task1)
    results1 = runner1.run()
    assert len(results1) == 3

    # 2. Second run: resume and request 5 repeats.
    # It should load the 3 cached repeats and execute only 2 more repeats!
    task2 = LocatorTask(
        combination=combo,
        repeats=5,
        seed=1,
        slug="test_slug",
        out_dir=tmp_path / "out",
        scans_dir=tmp_path / "out/scans",
        bayes_dir=tmp_path / "out/bayes",
        loc_max_steps=10,
        sweep_max_steps=10,
        loc_timeout_s=10,
        use_cache=True,
        cache_dir=tmp_path / "cache",
        log_queue=None,
        log_level=logging.INFO,
        ignore_cache_strategy=None,
        repeat_offset=0,
    )

    # Print database tables to debug
    db_path = tmp_path / "cache" / "nv_center.db"
    import sqlite3

    if db_path.exists():
        conn = sqlite3.connect(db_path)
        print("nv_center.db cache table:")
        for r in conn.execute("SELECT key, value FROM cache"):
            print(r)
        conn.close()
    else:
        print("nv_center.db DOES NOT EXIST")
    index_path = tmp_path / "cache" / "nv_center_index.db"
    if index_path.exists():
        conn = sqlite3.connect(index_path)
        print("nv_center_index.db cache_index table:")
        for r in conn.execute("SELECT key, shard_id FROM cache_index"):
            print(r)
        conn.close()
    else:
        print("nv_center_index.db DOES NOT EXIST")

    # List all files in the cache dir
    cache_dir_path = tmp_path / "cache"
    if cache_dir_path.exists():
        print("Cache directory files:", list(cache_dir_path.glob("*")))

    runner2 = _TaskRunner(task2)
    cached, n_cached = runner2._restore_cached_results()
    assert n_cached == 3
    assert len(cached) == 3

    results2 = runner2.run()
    assert len(results2) == 5

    # 3. Third run: `--no-cache` (use_cache=False) to override previous results.
    # It should purge the cache and run from scratch!
    task3 = LocatorTask(
        combination=combo,
        repeats=2,
        seed=1,
        slug="test_slug",
        out_dir=tmp_path / "out",
        scans_dir=tmp_path / "out/scans",
        bayes_dir=tmp_path / "out/bayes",
        loc_max_steps=10,
        sweep_max_steps=10,
        loc_timeout_s=10,
        use_cache=False,  # no-cache!
        cache_dir=tmp_path / "cache",
        log_queue=None,
        log_level=logging.INFO,
        ignore_cache_strategy=None,
        repeat_offset=0,
    )

    runner3 = _TaskRunner(task3)
    cached3, n_cached3 = runner3._restore_cached_results()
    assert n_cached3 == 0  # skip cache should not restore anything!

    results3 = runner3.run()
    assert len(results3) == 2

    # After running with override, check that the cache now contains exactly the 2 repeats from runner3 (not 5).
    bridge = CacheBridge(task3.cache_dir)
    try:
        repo = bridge.get_cache_for_category("NVCenter")
        final_loaded = repo.get_cached_combination(
            generator="NVCenter-lorentzian",
            noise="Gauss(0.01)",
            strategy="SimpleSweep",
            repeats=2,
            seed=1,
            max_steps=10,
            timeout_s=10,
            repeat_offset=0,
        )
        assert len(final_loaded) == 2

        # Requesting 5 should now result in partial hit of only 2 repeats (since the old ones were purged)
        partial_loaded, n_partial = repo.get_cached_combination_partial(
            generator="NVCenter-lorentzian",
            noise="Gauss(0.01)",
            strategy="SimpleSweep",
            repeats=5,
            seed=1,
            max_steps=10,
            timeout_s=10,
            repeat_offset=0,
        )
        assert n_partial == 2
        assert len(partial_loaded) == 2
    finally:
        bridge.close()


def test_harvest_partial_results_with_gaps_and_attempt_keys(tmp_path: Path):
    import logging

    from nvision.cache import CacheBridge
    from nvision.cli.run import _harvest_partial_results_from_cache
    from nvision.models.task import LocatorTask
    from nvision.noises.over_frequency.gaussian_noise import OverFrequencyGaussianNoise
    from nvision.sim.combinations import Combination
    from nvision.sim.locs.bayesian.sobol_bayesian_locator import SimpleSobolBayesianLocator
    from nvision.spectra.lorentzian import LorentzianModel

    sig = LorentzianModel()
    noise = OverFrequencyGaussianNoise(0.01)
    combo = Combination(
        generator=sig,
        noise=noise,
        strategy=SimpleSobolBayesianLocator,
        generator_name="NVCenter-lorentzian",
        noise_name="Gauss(0.01)",
        strategy_name="Bayesian-SBED",
    )

    task = LocatorTask(
        combination=combo,
        repeats=10,
        seed=1,
        slug="test_slug",
        out_dir=tmp_path / "out",
        scans_dir=tmp_path / "scans",
        bayes_dir=tmp_path / "bayes",
        loc_max_steps=10,
        sweep_max_steps=10,
        loc_timeout_s=10,
        use_cache=True,
        cache_dir=tmp_path / "cache",
        log_queue=None,
        log_level=logging.INFO,
        ignore_cache_strategy=None,
        repeat_offset=0,
    )

    # Write non-contiguous repeats (gaps) directly into the cache.
    # E.g., repeat 0, 2, 4 are saved. 1 and 3 are missing.
    bridge = CacheBridge(task.cache_dir)
    try:
        repo = bridge.get_cache_for_category("NVCenter")

        # Save repeat 0
        repo.save_repeat(
            generator="NVCenter-lorentzian",
            noise="Gauss(0.01)",
            strategy="Bayesian-SBED",
            seed=1,
            max_steps=10,
            timeout_s=10,
            repeat_offset=0,
            repeat_idx=0,
            entries=[{"path": "p0.png"}],
            main_result_row={
                "generator": "NVCenter-lorentzian",
                "noise": "Gauss(0.01)",
                "strategy": "Bayesian-SBED",
                "seed": 1,
                "attempt": 1,
            },
        )

        # Save repeat 2
        repo.save_repeat(
            generator="NVCenter-lorentzian",
            noise="Gauss(0.01)",
            strategy="Bayesian-SBED",
            seed=1,
            max_steps=10,
            timeout_s=10,
            repeat_offset=0,
            repeat_idx=2,
            entries=[{"path": "p2.png"}],
            main_result_row={
                "generator": "NVCenter-lorentzian",
                "noise": "Gauss(0.01)",
                "strategy": "Bayesian-SBED",
                "seed": 1,
                "attempt": 3,
            },
        )

        # Update pointer row to achieved=5 manually to simulate a hole from an older version
        from nvision.cache.hashing import stable_config_hash
        from nvision.cache.locator_keys import combination_base_cache_config

        config = combination_base_cache_config(
            generator="NVCenter-lorentzian",
            noise="Gauss(0.01)",
            strategy="Bayesian-SBED",
            seed=1,
            max_steps=10,
            timeout_s=10,
            repeat_offset=0,
        )
        ptr_key = stable_config_hash(config)
        import datetime

        import polars as pl

        ptr_df = pl.DataFrame({"achieved_repeats": [5], "streaming": [True]})
        repo._store.save_df(
            ptr_df, ptr_key, metadata={"config": config, "updated_at": datetime.datetime.now().isoformat()}
        )
    finally:
        bridge.close()

    df_rows = []
    plot_manifest = []
    logger = logging.getLogger("test_harvest_gaps")

    # Run harvest with allow_gaps=True (which is our new default in harvesting)
    _harvest_partial_results_from_cache(
        tasks=[task], cache_bridge=None, df_rows=df_rows, plot_manifest=plot_manifest, log=logger
    )

    # Both non-contiguous repeats (0 and 2) should be successfully harvested!
    assert len(df_rows) == 2
    assert df_rows[0]["attempt"] == 1
    assert df_rows[1]["attempt"] == 3
    assert len(plot_manifest) == 2
    assert plot_manifest[0]["path"] == "p0.png"
    assert plot_manifest[1]["path"] == "p2.png"


def test_cache_miss_explanations(tmp_path: Path, caplog):
    import logging

    from nvision import SimpleSweepLocator
    from nvision.models.noise import CompositeNoise, CompositeOverFrequencyNoise
    from nvision.models.task import LocatorTask
    from nvision.noises import OverFrequencyGaussianNoise
    from nvision.runner.executor import _TaskRunner
    from nvision.sim.combinations import Combination
    from nvision.sim.gen.nv_center_generator import NVCenterCoreGenerator

    sig = NVCenterCoreGenerator(x_min=2.6e9, x_max=3.1e9, variant="lorentzian")
    noise = CompositeNoise(over_frequency_noise=CompositeOverFrequencyNoise([OverFrequencyGaussianNoise(0.01)]))
    combo = Combination(
        generator=sig,
        noise=noise,
        strategy=SimpleSweepLocator,
        generator_name="NVCenter-lorentzian",
        noise_name="Gauss(0.01)",
        strategy_name="SimpleSweep",
    )

    # Task with cache disabled
    task_no_cache = LocatorTask(
        combination=combo,
        repeats=1,
        seed=1,
        slug="test_slug",
        out_dir=tmp_path / "out",
        scans_dir=tmp_path / "out/scans",
        bayes_dir=tmp_path / "out/bayes",
        loc_max_steps=10,
        sweep_max_steps=10,
        loc_timeout_s=10,
        use_cache=False,  # Skip cache!
        cache_dir=tmp_path / "cache",
        log_queue=None,
        log_level=logging.INFO,
        ignore_cache_strategy=None,
        repeat_offset=0,
    )

    runner_no_cache = _TaskRunner(task_no_cache)
    with caplog.at_level(logging.INFO):
        runner_no_cache._explain_cache_miss_total(0, 1)
    assert "Caching was bypassed/disabled via task config" in caplog.text

    caplog.clear()

    # Task with cache enabled, but database is completely empty
    task_empty_cache = LocatorTask(
        combination=combo,
        repeats=1,
        seed=1,
        slug="test_slug",
        out_dir=tmp_path / "out",
        scans_dir=tmp_path / "out/scans",
        bayes_dir=tmp_path / "out/bayes",
        loc_max_steps=10,
        sweep_max_steps=10,
        loc_timeout_s=10,
        use_cache=True,
        cache_dir=tmp_path / "cache_empty",
        log_queue=None,
        log_level=logging.INFO,
        ignore_cache_strategy=None,
        repeat_offset=0,
    )

    runner_empty_cache = _TaskRunner(task_empty_cache)
    with caplog.at_level(logging.INFO):
        runner_empty_cache._explain_cache_miss_total(0, 1)
    assert "No prior cache entries exist for combination" in caplog.text

    caplog.clear()

    # Now let's save a runner result for a DIFFERENT seed (e.g. seed=2)
    task_diff_seed = LocatorTask(
        combination=combo,
        repeats=1,
        seed=2,  # DIFFERENT seed!
        slug="test_slug",
        out_dir=tmp_path / "out",
        scans_dir=tmp_path / "out/scans",
        bayes_dir=tmp_path / "out/bayes",
        loc_max_steps=10,
        sweep_max_steps=10,
        loc_timeout_s=10,
        use_cache=True,
        cache_dir=tmp_path / "cache_shared",
        log_queue=None,
        log_level=logging.INFO,
        ignore_cache_strategy=None,
        repeat_offset=0,
    )
    runner_diff_seed = _TaskRunner(task_diff_seed)
    # Save seed=2 to the cache database
    from nvision.cache import CacheBridge

    bridge = CacheBridge(task_diff_seed.cache_dir)
    try:
        repo = bridge.get_cache_for_category("NVCenter")
        # Save the repeat first so count_saved finds it
        repo.save_repeat(
            generator="NVCenter-lorentzian",
            noise="Gauss(0.01)",
            strategy="SimpleSweep",
            seed=2,
            max_steps=10,
            timeout_s=10,
            repeat_offset=0,
            repeat_idx=0,
            entries=[],
            main_result_row={"test": 1},
        )
        repo.append_cached_repeats(
            generator="NVCenter-lorentzian",
            noise="Gauss(0.01)",
            strategy="SimpleSweep",
            seed=2,
            max_steps=10,
            timeout_s=10,
            repeat_offset=0,
            new_results=[],
            start_idx=1,
        )
    finally:
        bridge.close()

    # Now let's try to load/explain for seed=1
    task_target = LocatorTask(
        combination=combo,
        repeats=1,
        seed=1,  # target seed is 1
        slug="test_slug",
        out_dir=tmp_path / "out",
        scans_dir=tmp_path / "out/scans",
        bayes_dir=tmp_path / "out/bayes",
        loc_max_steps=10,
        sweep_max_steps=10,
        loc_timeout_s=10,
        use_cache=True,
        cache_dir=tmp_path / "cache_shared",
        log_queue=None,
        log_level=logging.INFO,
        ignore_cache_strategy=None,
        repeat_offset=0,
    )
    runner_target = _TaskRunner(task_target)
    with caplog.at_level(logging.INFO):
        runner_target._explain_cache_miss_total(0, 1)
    assert "Prior runs found" in caplog.text
    assert "seed mismatch" in caplog.text


def test_schema_version_8_fallback(tmp_path: Path, caplog):
    import logging

    from nvision import SimpleSweepLocator
    from nvision.models.noise import CompositeNoise, CompositeOverFrequencyNoise
    from nvision.models.task import LocatorTask
    from nvision.noises import OverFrequencyGaussianNoise
    from nvision.runner.executor import _TaskRunner
    from nvision.sim.combinations import Combination
    from nvision.sim.gen.nv_center_generator import NVCenterCoreGenerator

    sig = NVCenterCoreGenerator(x_min=2.6e9, x_max=3.1e9, variant="lorentzian")
    noise = CompositeNoise(over_frequency_noise=CompositeOverFrequencyNoise([OverFrequencyGaussianNoise(0.01)]))
    combo = Combination(
        generator=sig,
        noise=noise,
        strategy=SimpleSweepLocator,
        generator_name="NVCenter-lorentzian",
        noise_name="Gauss(0.01)",
        strategy_name="SimpleSweep",
    )

    task = LocatorTask(
        combination=combo,
        repeats=1,
        seed=1,
        slug="test_slug",
        out_dir=tmp_path / "out",
        scans_dir=tmp_path / "out/scans",
        bayes_dir=tmp_path / "out/bayes",
        loc_max_steps=10,
        sweep_max_steps=10,
        loc_timeout_s=10,
        use_cache=True,
        cache_dir=tmp_path / "cache_v8",
        log_queue=None,
        log_level=logging.INFO,
        ignore_cache_strategy=None,
        repeat_offset=0,
    )

    # Save a schema version 8 pointer and repeat to the database
    from nvision.cache import CacheBridge

    bridge = CacheBridge(task.cache_dir)
    try:
        repo = bridge.get_cache_for_category("NVCenter")

        # Pointer for version 8
        from nvision.cache.hashing import stable_config_hash
        from nvision.cache.locator_keys import combination_base_cache_config

        ptr_config = combination_base_cache_config(
            generator="NVCenter-lorentzian",
            noise="Gauss(0.01)",
            strategy="SimpleSweep",
            seed=1,
            max_steps=10,
            timeout_s=10,
            repeat_offset=0,
        )
        ptr_config_v8 = dict(ptr_config)
        ptr_config_v8["schema_version"] = 8
        ptr_key_v8 = stable_config_hash(ptr_config_v8)

        # Save pointer
        import polars as pl

        ptr_df = pl.DataFrame({"achieved_repeats": [1], "streaming": [True]})
        repo._store.save_df(ptr_df, ptr_key_v8, metadata={"config": ptr_config_v8})

        # Save repeat 0
        repo._repeats.save_repeat(
            ptr_key_v8,
            0,
            [{"path": "p0.png"}],
            {
                "generator": "NVCenter-lorentzian",
                "noise": "Gauss(0.01)",
                "strategy": "SimpleSweep",
                "seed": 1,
                "attempt": 1,
            },
        )
    finally:
        bridge.close()

    # Now load and run the TaskRunner with current version 9
    runner = _TaskRunner(task)
    cached, n_cached = runner._restore_cached_results()

    # It should successfully load the schema version 8 repeat!
    assert n_cached == 1
    assert len(cached) == 1
    assert cached[0][0][0]["path"] == "p0.png"


def test_task_runner_dry_run(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("nvision.runner.executor.generate_attempt_plots", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        "nvision.runner.executor._TaskRunner._run_sobol_baseline",
        lambda *args, **kwargs: {
            "sobol_baseline_steps": 1,
            "sobol_freq_steps": 1,
            "sobol_freq_uncert_at_conv": 0.0,
            "sobol_freq_err_at_conv": 0.0,
            "sobol_xs": [],
            "sobol_ys": [],
            "sobol_mode_estimates": {},
        },
    )

    import logging

    from nvision import SimpleSweepLocator
    from nvision.cache import CacheBridge
    from nvision.models.noise import CompositeNoise, CompositeOverFrequencyNoise
    from nvision.models.task import LocatorTask
    from nvision.noises import OverFrequencyGaussianNoise
    from nvision.runner.executor import _TaskRunner
    from nvision.sim.combinations import Combination
    from nvision.sim.gen.nv_center_generator import NVCenterCoreGenerator

    sig = NVCenterCoreGenerator(x_min=2.6e9, x_max=3.1e9, variant="lorentzian")
    noise = CompositeNoise(over_frequency_noise=CompositeOverFrequencyNoise([OverFrequencyGaussianNoise(0.01)]))
    combo = Combination(
        generator=sig,
        noise=noise,
        strategy=SimpleSweepLocator,
        generator_name="NVCenter-lorentzian",
        noise_name="Gauss(0.01)",
        strategy_name="SimpleSweep",
    )

    # Run with dry_run = True
    task = LocatorTask(
        combination=combo,
        repeats=1,
        seed=1,
        slug="test_slug",
        out_dir=tmp_path / "out",
        scans_dir=tmp_path / "out/scans",
        bayes_dir=tmp_path / "out/bayes",
        loc_max_steps=10,
        sweep_max_steps=10,
        loc_timeout_s=10,
        use_cache=True,
        dry_run=True,  # Dry Run!
        cache_dir=tmp_path / "cache_dry",
        log_queue=None,
        log_level=logging.INFO,
        ignore_cache_strategy=None,
        repeat_offset=0,
    )

    runner = _TaskRunner(task)
    results = runner.run()
    assert len(results) == 1

    # Verify that the cache DB does not have any saved combination
    bridge = CacheBridge(task.cache_dir)
    try:
        repo = bridge.get_cache_for_category("NVCenter")
        loaded = repo.get_cached_combination(
            generator="NVCenter-lorentzian",
            noise="Gauss(0.01)",
            strategy="SimpleSweep",
            repeats=1,
            seed=1,
            max_steps=10,
            timeout_s=10,
            repeat_offset=0,
        )
        assert loaded is None
    finally:
        bridge.close()
