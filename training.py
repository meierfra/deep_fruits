import load_data

lables = {}
X_train, Y_train, inv_lables_train = load_data.load_data_cached(load_data.DATA_PATH_TRAIN, lables)
X_val, Y_val, inv_lables_val = load_data.load_data_cached(load_data.DATA_PATH_VAL, lables)


