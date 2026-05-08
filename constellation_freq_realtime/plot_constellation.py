import numpy as np
import matplotlib.pyplot as plt

def load_binary_data(filename):
    data = np.fromfile(filename, dtype=np.complex64)
    return data

def plot_constellation_binary(filename, title_suffix):
    data = load_binary_data(filename)
    real = data.real
    imag = data.imag

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    axes[0].scatter(real, imag, s=1, alpha=0.1)
    axes[0].axis('equal')
    axes[0].grid(True)
    axes[0].set_xlabel('Real')
    axes[0].set_ylabel('Imaginary')
    axes[0].set_title(f'Constellation Diagram - {title_suffix}')

    subcarrier_start = 890
    subcarrier_end = 1910
    frame_size = 3780
    num_frames = len(data) // frame_size

    sub_data = []
    for i in range(num_frames):
        start = i * frame_size + subcarrier_start
        end = i * frame_size + subcarrier_end
        sub_data.append(data[start:end])
    sub_data = np.concatenate(sub_data)

    axes[1].scatter(sub_data.real, sub_data.imag, s=1, alpha=0.1)
    axes[1].axis('equal')
    axes[1].grid(True)
    axes[1].set_xlabel('Real')
    axes[1].set_ylabel('Imaginary')
    axes[1].set_title(f'Constellation Diagram (Subcarriers {subcarrier_start}-{subcarrier_end}) - {title_suffix}')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    plot_constellation_binary('constellation_batch_003.bin', 'Frequency Domain Channel Estimation')
