
import numpy as np

def generate_spec_batches(data_train, batch_size, noise_flag):
    if noise_flag == 0:
        data_ = data_train['input_nf']
    else:
        data_ = data_train['input']
    label_ = data_train['target_spec']
    data_len = len(label_)

    # shuffle data
    shuffle_seq = np.random.permutation(range(data_len))
    data = [data_[idx] for idx in shuffle_seq]
    label = [label_[idx] for idx in shuffle_seq]

    # generate batches
    num_batch = int(data_len / batch_size)
    data_batches = []
    label_batches = []
    for batch_idx in range(num_batch):
        batch_start = batch_idx * batch_size
        batch_end = np.min([(batch_idx + 1) * batch_size, data_len])
        data_batch = data[batch_start : batch_end]
        label_batch = label[batch_start: batch_end]
        data_batches.append(data_batch)
        label_batches.append(label_batch)

    return data_batches, label_batches

def convert_real_to_complex(real_vector):
    vector_len = len(real_vector)

    vector_real = real_vector[:int((vector_len + 1) / 2)]
    vector_imag = real_vector[int((vector_len + 1) / 2):]
    complex_vector = vector_real + 1j * vector_imag

    return complex_vector

def generate_array_cov_vector(M, N, d, wavelength, DOA, SNR):
    K = len(DOA)
    add_noise = np.random.randn(M, N) + 1j * np.random.randn(M, N)
    array_signal = 0

    for ki in range(K):
        signal_i = 10 ** (SNR[ki] / 20) * (np.random.randn(1, N) + 1j * np.random.randn(1, N))
        phase_shift_unit = 2 * np.pi * d / wavelength * np.sin(DOA[ki] / 180 * np.pi)
        a_i_ = np.cos(np.array(range(M)) * phase_shift_unit) + 1j * np.sin(np.array(range(M)) * phase_shift_unit)
        a_i = np.expand_dims(a_i_, axis=-1)
        array_signal_i = np.matmul(a_i, signal_i)
        array_signal += array_signal_i

    array_output = array_signal + add_noise

    array_covariance = 1 / N * (np.matmul(array_output, np.matrix.getH(array_output)))
    cov_vector_ = []
    for row_idx in range(M):
        cov_vector_.extend(array_covariance[row_idx, (row_idx + 1):])
    cov_vector_ = np.asarray(cov_vector_)
    cov_vector_ext = np.concatenate([cov_vector_.real, cov_vector_.imag])
    cov_vector = 1 / np.linalg.norm(cov_vector_ext) * cov_vector_ext

    return cov_vector

def spatial_scan(vector_complex, vector_dictionary):
    Spatial_Spectrum = []
    NUM_GRID = len(vector_dictionary)
    for grid_idx in range(NUM_GRID):
        basis_real = vector_dictionary[grid_idx]
        basis_complex = convert_real_to_complex(basis_real)
        corr_i_ = np.matmul(np.matrix.getH(basis_complex), vector_complex)
        corr_i = np.abs(corr_i_)
        Spatial_Spectrum.extend(corr_i)

    return Spatial_Spectrum

def generate_target_spectrum(DOA, doa_min, grid, NUM_GRID):
    K = len(DOA)
    target_vector = 0
    for ki in range(K):
        doa_i = DOA[ki]
        target_vector_i_ = []
        grid_idx = 0
        while grid_idx < NUM_GRID:
            grid_pre = doa_min + grid * grid_idx
            grid_post = doa_min + grid * (grid_idx + 1)
            if grid_pre <= doa_i and grid_post > doa_i:
                expand_vec = np.array([grid_post - doa_i, doa_i - grid_pre]) / grid
                grid_idx += 2
            else:
                expand_vec = np.array([0.0])
                grid_idx += 1
            target_vector_i_.extend(expand_vec)
        if len(target_vector_i_) >= NUM_GRID:
            target_vector_i = target_vector_i_[:NUM_GRID]
        else:
            expand_vec = np.zeros(NUM_GRID - len(target_vector_i_))
            target_vector_i = target_vector_i_
            target_vector_i.extend(expand_vec)
        target_vector += np.asarray(target_vector_i)
    # target_vector /= K

    return target_vector

def generate_target_svm(DOA, doa_min, grid, NUM_GRID):
    K = len(DOA)
    target_svm = np.zeros([NUM_GRID, 1])
    target_svr = np.zeros([NUM_GRID, 1])
    for ki in range(K):
        doa_i = DOA[ki]
        grid_idx = 0
        while grid_idx < NUM_GRID:
            grid_pre = doa_min + grid * grid_idx
            grid_post = doa_min + grid * (grid_idx + 1)
            if grid_pre <= doa_i and grid_post > doa_i:
                target_svm[grid_idx] = 1
                target_svr[grid_idx] = doa_i
            grid_idx += 1

    return target_svm, target_svr

def generate_training_data_sf(M, N, d, wavelength, SNR, doa_min, NUM_REPEAT_SF, grid, GRID_NUM, output_size, SF_NUM, SF_SCOPE):
    data_train_sf = {}
    data_train_sf['input_nf'] = []
    data_train_sf['input'] = []
    data_train_sf['target_spec'] = []
    for doa_idx in range(GRID_NUM):
        DOA = doa_min + grid * doa_idx

        for rep_idx in range(NUM_REPEAT_SF):
            add_noise = np.random.randn(M, N) + 1j * np.random.randn(M, N)
            array_signal = 0

            signal_i = 10 ** (SNR / 20) * (np.random.randn(1, N) + 1j * np.random.randn(1, N))
            phase_shift_unit = 2 * np.pi * d / wavelength * np.sin(DOA / 180 * np.pi)
            a_i_ = np.cos(np.array(range(M)) * phase_shift_unit) + 1j * np.sin(np.array(range(M)) * phase_shift_unit)
            a_i = np.expand_dims(a_i_, axis=-1)
            array_signal_i = np.matmul(a_i, signal_i)
            array_signal += array_signal_i

            array_output_nf = array_signal + 0 * add_noise  # noise-free output
            array_output = array_signal + 1 * add_noise

            array_covariance_nf = 1 / N * (np.matmul(array_output_nf, np.matrix.getH(array_output_nf)))
            array_covariance = 1 / N * (np.matmul(array_output, np.matrix.getH(array_output)))
            cov_vector_nf_ = []
            cov_vector_ = []
            for row_idx in range(M):
                cov_vector_nf_.extend(array_covariance_nf[row_idx, (row_idx + 1):])
                cov_vector_.extend(array_covariance[row_idx, (row_idx + 1):])
            cov_vector_nf_ = np.asarray(cov_vector_nf_)
            cov_vector_nf_ext = np.concatenate([cov_vector_nf_.real, cov_vector_nf_.imag])
            cov_vector_nf = 1 / np.linalg.norm(cov_vector_nf_ext) * cov_vector_nf_ext
            data_train_sf['input_nf'].append(cov_vector_nf)
            cov_vector_ = np.asarray(cov_vector_)
            cov_vector_ext = np.concatenate([cov_vector_.real, cov_vector_.imag])
            cov_vector = 1 / np.linalg.norm(cov_vector_ext) * cov_vector_ext
            data_train_sf['input'].append(cov_vector)
            # construct multi-task autoencoder target
            scope_label = int((DOA - doa_min) / SF_SCOPE)
            target_curr_pre = np.zeros([output_size * scope_label, 1])
            target_curr_post = np.zeros([output_size * (SF_NUM - scope_label - 1), 1])
            target_curr = np.expand_dims(cov_vector, axis=-1)
            target = np.concatenate([target_curr_pre, target_curr, target_curr_post], axis=0)
            data_train_sf['target_spec'].append(np.squeeze(target))

    return data_train_sf


def generate_training_data_ss(M, N, K, d, wavelength, SNR_ss, SNR_DIFF, doa_min, doa_max, step, doa_delta, NUM_REPEAT_SS, grid_ss, NUM_GRID_SS):
    data_train_ss = {}
    data_train_ss['input_nf'] = []
    data_train_ss['input'] = []
    data_train_ss['target_spec'] = []
    SNR0 = SNR_ss[0]
    for snr_diff in SNR_DIFF:
        SNR = [SNR0 - k * snr_diff for k in range(K)]
        for delta_idx in range(len(doa_delta)):
            delta_curr = doa_delta[delta_idx]  # inter-signal direction differences
            delta_cum_seq_ = [delta_curr]  # doa differences w.r.t first signal
            delta_cum_seq = np.concatenate([[0], delta_cum_seq_])  # the first signal included
            delta_sum = np.sum(delta_curr)  # direction difference between first and last signals
            NUM_STEP = int((doa_max - doa_min - delta_sum) / step)  # number of scanning steps

            for step_idx in range(NUM_STEP):
                doa_first = doa_min + step * step_idx
                DOA = delta_cum_seq + doa_first

                for rep_idx in range(NUM_REPEAT_SS):
                    add_noise = np.random.randn(M, N) + 1j * np.random.randn(M, N)
                    array_signal = 0
                    for ki in range(K):
                        signal_i = 10 ** (SNR[ki] / 20) * (np.random.randn(1, N) + 1j * np.random.randn(1, N))
                        phase_shift_unit = 2 * np.pi * d / wavelength * np.sin(DOA[ki] / 180 * np.pi)
                        a_i_ = np.cos(np.array(range(M)) * phase_shift_unit) + 1j * np.sin(
                            np.array(range(M)) * phase_shift_unit)
                        a_i = np.expand_dims(a_i_, axis=-1)
                        array_signal_i = np.matmul(a_i, signal_i)
                        array_signal += array_signal_i

                    array_output_nf = array_signal + 0 * add_noise  # noise-free output
                    array_output = array_signal + 1 * add_noise

                    array_covariance_nf = 1 / N * (np.matmul(array_output_nf, np.matrix.getH(array_output_nf)))
                    array_covariance = 1 / N * (np.matmul(array_output, np.matrix.getH(array_output)))
                    cov_vector_nf_ = []
                    cov_vector_ = []
                    for row_idx in range(M):
                        cov_vector_nf_.extend(array_covariance_nf[row_idx, (row_idx + 1):])
                        cov_vector_.extend(array_covariance[row_idx, (row_idx + 1):])
                    cov_vector_nf_ = np.asarray(cov_vector_nf_)
                    cov_vector_nf_ext = np.concatenate([cov_vector_nf_.real, cov_vector_nf_.imag])
                    cov_vector_nf = 1 / np.linalg.norm(cov_vector_nf_ext) * cov_vector_nf_ext
                    data_train_ss['input_nf'].append(cov_vector_nf)
                    cov_vector_ = np.asarray(cov_vector_)
                    cov_vector_ext = np.concatenate([cov_vector_.real, cov_vector_.imag])
                    cov_vector = 1 / np.linalg.norm(cov_vector_ext) * cov_vector_ext
                    data_train_ss['input'].append(cov_vector)
                    # construct spatial spectrum target
                    target_spectrum = generate_target_spectrum(DOA, doa_min, grid_ss, NUM_GRID_SS)
                    data_train_ss['target_spec'].append(target_spectrum)

    return data_train_ss

def smooth_spectrum(ss_orig, smooth_width):
    ss_len = len(ss_orig)
    half_width = int(smooth_width / 2)
    ss_smooth = []
    for idx in range(ss_len):
        start_loc = np.max([0, idx - half_width])
        end_loc = np.min([ss_len, idx + half_width])
        ss_ave_curr = np.sum(ss_orig[start_loc:end_loc]) / smooth_width
        ss_smooth.append(ss_ave_curr)

    return ss_smooth

def get_DOA_estimate(spec, DOA, doa_min, grid):
    K = len(DOA)

    # extract peaks from spectrum
    peaks = []
    peak_flag = False
    peak_start = 0
    peak_end = 0
    idx = 0
    while idx < len(spec):
        if spec[idx][0] > 0:
            if peak_flag == False:
                peak_start = idx
                peak_end = idx
            else:
                peak_end += 1
            peak_flag = True
        else:
            if peak_flag == True:
                peak_curr = np.array([peak_start, peak_end])
                peaks.append(peak_curr)
            peak_flag = False
        idx += 1

    # estimate directions
    K_est = len(peaks)
    peak_doa_list = []
    peak_amp_list = []
    for ki in range(K_est):
        curr_start = peaks[ki][0]
        curr_end = peaks[ki][1]
        curr_spec = [spec[ii][0] for ii in range(curr_start, curr_end + 1)]
        curr_grid = doa_min + grid * np.arange(curr_start, curr_end + 1)
        curr_amp = np.sum(curr_spec)          # sort peaks with total energy
        curr_doa = np.sum(curr_spec * curr_grid) / np.sum(curr_spec)
        peak_doa_list.append(curr_doa)
        peak_amp_list.append(curr_amp)

    # output doa estimates
    doa_est = []
    if K_est == 0:
        for ki in range(K):
            doa_est.append(DOA[0])
    elif K_est <= K:
        for ki in range(K):
            doa_i = DOA[ki]
            est_error = [np.abs(peak_doa - doa_i) for peak_doa in peak_doa_list]
            est_idx = np.argmin(est_error)
            doa_est_i = peak_doa_list[est_idx]
            doa_est.append(doa_est_i)
    else:
        doa_est_ = []
        for ki in range(K):
            est_idx = np.argmax(peak_amp_list)
            doa_est_i = peak_doa_list[est_idx]
            doa_est_.append(doa_est_i)
            peak_amp_list[est_idx] = -1
        for ki in range(K):
            doa_i = DOA[ki]
            est_error = [np.abs(peak_doa - doa_i) for peak_doa in doa_est_]
            est_idx = np.argmin(est_error)
            doa_est_i = doa_est_[est_idx]
            doa_est.append(doa_est_i)

    return doa_est


def generate_training_data_sf_AI(M, N, d, wavelength, SNR, doa_min, NUM_REPEAT_SF, grid, GRID_NUM, output_size, SF_NUM, SF_SCOPE, MC_mtx, AP_mtx, pos_para):
    data_train_sf = {}
    data_train_sf['input_nf'] = []
    data_train_sf['input'] = []
    data_train_sf['target_spec'] = []
    for doa_idx in range(GRID_NUM):
        DOA = doa_min + grid * doa_idx

        for rep_idx in range(NUM_REPEAT_SF):
            add_noise = np.random.randn(M, N) + 1j * np.random.randn(M, N)
            array_signal = 0

            signal_i = 10 ** (SNR / 20) * (np.random.randn(1, N) + 1j * np.random.randn(1, N))
            # phase_shift_unit = 2 * np.pi * d / wavelength * np.sin(DOA / 180 * np.pi)
            array_geom = np.expand_dims(np.array(np.arange(M)), axis=-1) * d + pos_para
            phase_shift_array = 2 * np.pi * array_geom / wavelength * np.sin(DOA / 180 * np.pi)
            a_i = np.cos(phase_shift_array) + 1j * np.sin(phase_shift_array)
            a_i = np.matmul(AP_mtx, a_i)
            a_i = np.matmul(MC_mtx, a_i)
            array_signal_i = np.matmul(a_i, signal_i)
            array_signal += array_signal_i

            array_output_nf = array_signal + 0 * add_noise  # noise-free output
            array_output = array_signal + 1 * add_noise

            array_covariance_nf = 1 / N * (np.matmul(array_output_nf, np.matrix.getH(array_output_nf)))
            array_covariance = 1 / N * (np.matmul(array_output, np.matrix.getH(array_output)))
            cov_vector_nf_ = []
            cov_vector_ = []
            for row_idx in range(M):
                cov_vector_nf_.extend(array_covariance_nf[row_idx, (row_idx + 1):])
                cov_vector_.extend(array_covariance[row_idx, (row_idx + 1):])
            cov_vector_nf_ = np.asarray(cov_vector_nf_)
            cov_vector_nf_ext = np.concatenate([cov_vector_nf_.real, cov_vector_nf_.imag])
            cov_vector_nf = 1 / np.linalg.norm(cov_vector_nf_ext) * cov_vector_nf_ext
            data_train_sf['input_nf'].append(cov_vector_nf)
            cov_vector_ = np.asarray(cov_vector_)
            cov_vector_ext = np.concatenate([cov_vector_.real, cov_vector_.imag])
            cov_vector = 1 / np.linalg.norm(cov_vector_ext) * cov_vector_ext
            data_train_sf['input'].append(cov_vector)
            # construct multi-task autoencoder target
            scope_label = int((DOA - doa_min) / SF_SCOPE)
            target_curr_pre = np.zeros([output_size * scope_label, 1])
            target_curr_post = np.zeros([output_size * (SF_NUM - scope_label - 1), 1])
            target_curr = np.expand_dims(cov_vector, axis=-1)
            target = np.concatenate([target_curr_pre, target_curr, target_curr_post], axis=0)
            data_train_sf['target_spec'].append(np.squeeze(target))

    return data_train_sf


def generate_training_data_ss_AI(M, N, K, d, wavelength, SNR, doa_min, doa_max, step, doa_delta, NUM_REPEAT_SS, grid_ss, NUM_GRID_SS, MC_mtx, AP_mtx, pos_para):
    data_train_ss = {}
    data_train_ss['input_nf'] = []
    data_train_ss['input'] = []
    data_train_ss['target_spec'] = []
    for delta_idx in range(len(doa_delta)):
        delta_curr = doa_delta[delta_idx]  # inter-signal direction differences
        delta_cum_seq_ = [delta_curr]  # doa differences w.r.t first signal
        delta_cum_seq = np.concatenate([[0], delta_cum_seq_])  # the first signal included
        delta_sum = np.sum(delta_curr)  # direction difference between first and last signals
        NUM_STEP = int((doa_max - doa_min - delta_sum) / step)  # number of scanning steps

        for step_idx in range(NUM_STEP):
            doa_first = doa_min + step * step_idx
            DOA = delta_cum_seq + doa_first

            for rep_idx in range(NUM_REPEAT_SS):
                add_noise = np.random.randn(M, N) + 1j * np.random.randn(M, N)
                array_signal = 0
                for ki in range(K):
                    signal_i = 10 ** (SNR[ki] / 20) * (np.random.randn(1, N) + 1j * np.random.randn(1, N))
                    # phase_shift_unit = 2 * np.pi * d / wavelength * np.sin(DOA / 180 * np.pi)
                    array_geom = np.expand_dims(np.array(np.arange(M)), axis=-1) * d + pos_para
                    phase_shift_array = 2 * np.pi * array_geom / wavelength * np.sin(DOA[ki] / 180 * np.pi)
                    a_i = np.cos(phase_shift_array) + 1j * np.sin(phase_shift_array)
                    a_i = np.matmul(AP_mtx, a_i)
                    a_i = np.matmul(MC_mtx, a_i)
                    array_signal_i = np.matmul(a_i, signal_i)
                    array_signal += array_signal_i

                array_output_nf = array_signal + 0 * add_noise  # noise-free output
                array_output = array_signal + 1 * add_noise

                array_covariance_nf = 1 / N * (np.matmul(array_output_nf, np.matrix.getH(array_output_nf)))
                array_covariance = 1 / N * (np.matmul(array_output, np.matrix.getH(array_output)))
                cov_vector_nf_ = []
                cov_vector_ = []
                for row_idx in range(M):
                    cov_vector_nf_.extend(array_covariance_nf[row_idx, (row_idx + 1):])
                    cov_vector_.extend(array_covariance[row_idx, (row_idx + 1):])
                cov_vector_nf_ = np.asarray(cov_vector_nf_)
                cov_vector_nf_ext = np.concatenate([cov_vector_nf_.real, cov_vector_nf_.imag])
                cov_vector_nf = 1 / np.linalg.norm(cov_vector_nf_ext) * cov_vector_nf_ext
                data_train_ss['input_nf'].append(cov_vector_nf)
                cov_vector_ = np.asarray(cov_vector_)
                cov_vector_ext = np.concatenate([cov_vector_.real, cov_vector_.imag])
                cov_vector = 1 / np.linalg.norm(cov_vector_ext) * cov_vector_ext
                data_train_ss['input'].append(cov_vector)
                # construct spatial spectrum target
                target_spectrum = generate_target_spectrum(DOA, doa_min, grid_ss, NUM_GRID_SS)
                data_train_ss['target_spec'].append(target_spectrum)

    return data_train_ss


def generate_array_cov_vector_AI(M, N, d, wavelength, DOA, SNR, MC_mtx, AP_mtx, pos_para):
    K = len(DOA)
    add_noise = np.random.randn(M, N) + 1j * np.random.randn(M, N)
    array_signal = 0

    for ki in range(K):
        signal_i = 10 ** (SNR[ki] / 20) * (np.random.randn(1, N) + 1j * np.random.randn(1, N))
        # phase_shift_unit = 2 * np.pi * d / wavelength * np.sin(DOA / 180 * np.pi)
        array_geom = np.expand_dims(np.array(np.arange(M)), axis=-1) * d + pos_para
        phase_shift_array = 2 * np.pi * array_geom / wavelength * np.sin(DOA[ki] / 180 * np.pi)
        a_i = np.cos(phase_shift_array) + 1j * np.sin(phase_shift_array)
        a_i = np.matmul(AP_mtx, a_i)
        a_i = np.matmul(MC_mtx, a_i)
        array_signal_i = np.matmul(a_i, signal_i)
        array_signal += array_signal_i

    array_output = array_signal + add_noise

    array_covariance = 1 / N * (np.matmul(array_output, np.matrix.getH(array_output)))
    cov_vector_ = []
    for row_idx in range(M):
        cov_vector_.extend(array_covariance[row_idx, (row_idx + 1):])
    cov_vector_ = np.asarray(cov_vector_)
    cov_vector_ext = np.concatenate([cov_vector_.real, cov_vector_.imag])
    cov_vector = 1 / np.linalg.norm(cov_vector_ext) * cov_vector_ext

    return cov_vector