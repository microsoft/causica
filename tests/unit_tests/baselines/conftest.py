import glob

DIRS = ["evaluation_pipeline/configs", "open_source/configs"]


def pytest_generate_tests(metafunc):
    if "config_path" in metafunc.fixturenames:
        matching_configs = [path for dir in DIRS for path in glob.glob(f"{dir}/**/*.json")]
        metafunc.parametrize("config_path", matching_configs)
