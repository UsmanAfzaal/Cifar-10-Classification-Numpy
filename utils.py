import matplotlib.pyplot as plt
import numpy as np

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def data_visualizer(file):  # TODO make it work for any number of samples
    result = unpickle(file)
    image_data = result[b"data"]

    #for number_of_images in range(samples):
    number_of_rows = image_data.shape[0]
    random_indices = np.random.choice(number_of_rows, size=1, replace=False)
    image_to_view = image_data[random_indices][0]

    R = image_to_view[:1024].reshape((32, 32))
    G = image_to_view[1024:2048].reshape((32, 32))
    B = image_to_view[2048:3072].reshape((32, 32))

    image = np.dstack((R, G, B))
    plt.imshow(image)


def load_train(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        dict_decoded = {}
        for k, v in dict.items():
            dict_decoded[k.decode('utf8')] = v
        dict = dict_decoded

    data = dict['data']
    data = data.reshape(data.shape[0], 3, 32, 32)
    return data

def normalize(image):
    return np.array(image) / 255.0
