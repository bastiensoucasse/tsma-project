import numpy as np


def time_warp(mel_spec, time_warping_para):
    '''
    Applies time warping to the input melspectro data.
    '''

    # Get the number of rows and columns in the melspectro data
    num_rows, num_cols = mel_spec.shape[1], mel_spec.shape[2]

    # Sample the number of time warpings to apply from a Poisson distribution
    num_time_warpings = np.random.poisson(time_warping_para)
    for _ in range(num_time_warpings):

        # Choose two random time steps
        t1, t2 = np.random.randint(0, num_cols, 2)

        # Swap the time steps
        mel_spec[:, :, t1], mel_spec[:, :, t2] = mel_spec[:, :, t2], mel_spec[:, :, t1]

    return mel_spec


def frequency_mask(mel_spec, frequency_masking_para):
    '''
    Applies frequency masking to the input melspectro data.
    '''

    # Sample the number of frequency masks to apply from a Poisson distribution
    num_masks = np.random.poisson(frequency_masking_para)

    for _ in range(num_masks):
        # Choose a random frequency range
        f1, f2 = np.random.uniform(0, 1, 2)
        f1, f2 = int(f1 * mel_spec.shape[1]), int(f2 * mel_spec.shape[1])

        # Zero out the chosen frequency range
        mel_spec[:, f1:f2, :] = 0

    return mel_spec


def time_mask(mel_spec, time_masking_para):
    '''
    Applies time masking to the input melspectro data.
    '''

    # Sample the number of time masks to apply from a Poisson distribution
    num_masks = np.random.poisson(time_masking_para)

    for _ in range(num_masks):
        # Choose a random time range
        t1, t2 = np.random.uniform(0, 1, 2)
        t1, t2 = int(t1 * mel_spec.shape[2]), int(t2 * mel_spec.shape[2])

        # Zero out the chosen time range
        mel_spec[:, :, t1:t2] = 0

    return mel_spec


def spec_augment(mel_spec):
    '''
    Applies the SpecAugment transformations (time warping, frequency masking, and time masking) to the input melspectro data.
    '''

    # Define the SpecAugment parameters
    time_warping_para = 5
    frequency_masking_para = 5
    time_masking_para = 10

    # Apply time warping to the melspectro data
    mel_spec = time_warp(mel_spec, time_warping_para)

    # Apply frequency masking
    mel_spec = frequency_mask(mel_spec, frequency_masking_para)

    # Apply time masking to the melspectro data
    mel_spec = time_mask(mel_spec, time_masking_para)

    return mel_spec
