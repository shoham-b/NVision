try:
    print("Importing ProjectBayesianLocator...")
    from nvision.sim.locs.nv_center.project_bayesian_locator import ProjectBayesianLocator

    print("Success!")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Error: {e}")

print("-" * 20)

try:
    print("Importing from nvision.sim...")
    from nvision.sim import ProjectBayesianLocator

    print("Success!")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Error: {e}")
