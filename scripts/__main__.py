from importlib import import_module


def run_module(module_path: str) -> None:
    """
    Run one of the workflow modules in sequence.

    Parameters
    ----
    module_path (str):
        Dotted path for the script module (e.g. ``"scripts.model_training"``).

    Notes
    ----
    ChatGPT helped me create this orchestration as I never seen it before.
    I was part curious but also wanted to make sure the package can be run
    out of the box.
    """
    print(f"=== Running {module_path} ===")
    import_module(module_path)
    print(f"=== Finished {module_path} ===")


def main() -> None:
    """
    Orchestrate the full workflow:

    1. Train and save models.
    2. Evaluate saved models.
    3. Generate visualisations.
    """
    modules = [
        "scripts.model_training",
        "scripts.evaluation",
        "scripts.visualisation",
    ]
    for module_path in modules:
        run_module(module_path)


if __name__ == "__main__":
    main()
