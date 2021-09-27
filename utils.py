import pickle

def load(datadir):
    with open(datadir) as f:
        data = f.read().splitlines()
    return data

def write(data, savedir, mode='w'):
    f = open(savedir, mode)
    for text in data:
        f.write(text+'\n')
    f.close()


def load_pickle(datadir):
  file = open(datadir, 'rb')
  data = pickle.load(file)
  return data

def write_pickle(data, savedir):
  file = open(savedir, 'wb')
  pickle.dump(data, file)
  file.close()

def scale(X, min, max, curr_min=None, curr_max=None):
    if curr_min is None:
        curr_min = X.min()
    if curr_max is None:
        curr_max = X.max()

    if curr_min == curr_max:
        return np.ones_like(X)

    X_std = (X - curr_min) / (curr_max - curr_min) + 1e-5
    X_scaled = X_std * (max - min) + min
    return X_scaled
