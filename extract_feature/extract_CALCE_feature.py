from _bootstrap import import_feature_lib


lib = import_feature_lib()


if __name__ == "__main__":
    lib.run_single_dataset_cli("CALCE")
