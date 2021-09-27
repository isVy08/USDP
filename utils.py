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