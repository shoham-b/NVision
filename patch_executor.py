with open("nvision/runner/executor.py", "r") as f:
    content = f.read()

# Add imports
if "from nvision.models.locator import LocatorConfig, ConvergenceConfig" not in content:
    content = content.replace("from nvision.models.locator import Locator", "from nvision.models.locator import Locator, LocatorConfig, ConvergenceConfig")

# Replace first create
s1 = """    if needs_belief and ("belief" not in locator_config or "signal_model" not in locator_config):
        locator_config.setdefault("belief", _create_sweep_belief(experiment))
        locator_config.setdefault("signal_model", experiment.true_signal.model)

    locator = locator_class.create(**locator_config)"""
r1 = """    if needs_belief and ("belief" not in locator_config or "signal_model" not in locator_config):
        locator_config.setdefault("belief", _create_sweep_belief(experiment))
        locator_config.setdefault("signal_model", experiment.true_signal.model)

    conv_cfg = ConvergenceConfig(
        threshold=locator_config.pop("convergence_threshold", 0.01),
        patience_steps=locator_config.pop("convergence_patience_steps", 8),
        params=locator_config.pop("convergence_params", None)
    )
    loc_cfg = LocatorConfig(
        max_steps=locator_config.pop("max_steps", 150),
        noise_std=locator_config.pop("noise_std", None),
        initial_sweep_steps=locator_config.pop("initial_sweep_steps", None),
        convergence=conv_cfg
    )

    locator = locator_class.create(config=loc_cfg, **locator_config)"""

content = content.replace(s1, r1)


# Replace third create inside _run_single_repeat
s3 = """        last_loc = observer.last_locator
        if last_loc is not None:
            locator_instance = last_loc
        else:
            locator_instance = locator_class.create(**cfg)"""

r3 = """        last_loc = observer.last_locator
        if last_loc is not None:
            locator_instance = last_loc
        else:
            conv_cfg = ConvergenceConfig(
                threshold=cfg.pop("convergence_threshold", 0.01),
                patience_steps=cfg.pop("convergence_patience_steps", 8),
                params=cfg.pop("convergence_params", None)
            )
            loc_cfg = LocatorConfig(
                max_steps=cfg.pop("max_steps", 150),
                noise_std=cfg.pop("noise_std", None),
                initial_sweep_steps=cfg.pop("initial_sweep_steps", None),
                convergence=conv_cfg
            )
            locator_instance = locator_class.create(config=loc_cfg, **cfg)"""

content = content.replace(s3, r3)

with open("nvision/runner/executor.py", "w") as f:
    f.write(content)
