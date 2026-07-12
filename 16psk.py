import os
import sys
import time
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Tuple
import threading
import traceback
import uhd

# Configuration Settings for 16-PSK
M = 16
BITS_PER_SYM = int(np.log2(M))    
rolloff = 0.8
span = 6
REP_COUNT = 10
stat_store = {}
os.add_dll_directory(r"C:\Program Files\UHD\bin") 

def err_track(args):
    print(f"\n❌ CRITICAL: Exception in thread {args.thread.name}:", file=sys.stderr)
    traceback.print_exception(args.exc_type, args.exc_value, args.exc_traceback, file=sys.stderr)

threading.excepthook = err_track


def gray_to_bin_scalar(g: int) -> int:
    """Algorithmic Gray Code to Binary Integer Converter"""
    b = 0
    while g > 0:
        b ^= g
        g >>= 1
    return b

def make_rrc(rolloff, num_span, sps) -> np.ndarray:
    g_delay = int(num_span * sps / 2)
    t_vec = np.arange(-g_delay, g_delay + 1) / sps
    pulse_h = np.zeros(len(t_vec))
    for idx, t_val in enumerate(t_vec):
        if t_val == 0.0:
            pulse_h[idx] = 1.0 - rolloff + (4 * rolloff / np.pi)
        elif np.isclose(np.abs(t_val), 1.0 / (4 * rolloff), atol=1e-8):
            pulse_h[idx] = (rolloff / np.sqrt(2)) * (
                ((1 + 2 / np.pi) * np.sin(np.pi / (4 * rolloff)))
                + ((1 - 2 / np.pi) * np.cos(np.pi / (4 * rolloff)))
            )
        else:
            n_val = np.sin(np.pi * t_val * (1 - rolloff)) + 4 * rolloff * t_val * np.cos(
                np.pi * t_val * (1 + rolloff)
            )
            d_val = np.pi * t_val * (1 - (4 * rolloff * t_val) ** 2)
            pulse_h[idx] = n_val / d_val
    return pulse_h / np.sqrt(np.sum(pulse_h**2))

def map_bits(bit_arr):
    """Universal Dynamic M-PSK Gray Bit-to-Symbol Mapper"""
    bit_arr = np.array(bit_arr).astype(int)
    rem = len(bit_arr) % BITS_PER_SYM
    if rem != 0:
        bit_arr = np.concatenate((bit_arr, np.zeros(BITS_PER_SYM - rem, dtype=int)))
        
    reshape_bits = bit_arr.reshape(-1, BITS_PER_SYM)
    
    # Pack grouped bits into integers
    ints = np.zeros(len(reshape_bits), dtype=int)
    for b_idx in range(BITS_PER_SYM):
        ints += reshape_bits[:, b_idx] * (2 ** (BITS_PER_SYM - 1 - b_idx))
        
    # Bitwise shift conversion to generate Gray Code Indexing
    gray_idxs = ints ^ (ints >> 1)
    phases = (2 * np.pi * gray_idxs / M) + (np.pi/M)
    return np.exp(1j * phases)

def build_frame(info_bits, base_pat, sync_bits, sps, tx_filter, scale_amp):
    f_cfo_bits = np.tile(base_pat, REP_COUNT)
    head_bits = np.concatenate((f_cfo_bits, sync_bits))
    head_syms = map_bits(head_bits)
    head_up = np.repeat(head_syms, sps)
    info_syms = map_bits(info_bits)
    info_up = np.zeros(len(info_syms) * sps, dtype=np.complex128)
    info_up[0::sps] = info_syms
    raw_payload = np.convolve(info_up, tx_filter, 'full')
    scaled_payload = (raw_payload / np.max(np.abs(raw_payload))) * scale_amp
    np.random.seed(123)
    dummy_bits = np.random.randint(0, 2, 40 * BITS_PER_SYM)
    dummy_syms = map_bits(dummy_bits)
    dummy_up = np.zeros(len(dummy_syms) * sps, dtype=np.complex128)
    dummy_up[0::sps] = dummy_syms
    raw_dummy = np.convolve(dummy_up, tx_filter, 'full')
    scaled_dummy = (raw_dummy / np.max(np.abs(raw_dummy))) * scale_amp
    quiet_gap = np.zeros(int(200e3 * 0.05), dtype=np.complex128)
    tail_gap = np.zeros(1000, dtype=np.complex128)
    head_up = (head_up / np.max(np.abs(head_up))) * scale_amp
    out_packet = np.concatenate((quiet_gap, scaled_dummy, head_up, scaled_payload, tail_gap))
    mix_syms = np.concatenate((head_syms, info_syms))
    return out_packet, head_bits, mix_syms, scaled_payload

def init_hw(tx_id, rx_id, fs_rate, carrier_f, gain_tx, gain_rx):
    print(f"Connecting to TX usrp {tx_id}...")
    dev_tx = uhd.usrp.MultiUSRP(f"serial={tx_id}")
    print(f"Connecting to RX usrp {rx_id}...")
    dev_rx = uhd.usrp.MultiUSRP(f"serial={rx_id}")
    try:
        dev_tx.set_master_clock_rate(40e6)
        dev_rx.set_master_clock_rate(40e6)
        dev_tx.set_tx_rate(fs_rate)
        dev_rx.set_rx_rate(fs_rate)
    except Exception as hardware_err:
        print(f"UHD Configuration Error: {hardware_err}")
    finally:
        print('Configured the clock speeds')
    dev_tx.set_tx_freq(uhd.types.TuneRequest(carrier_f), 0)
    dev_rx.set_rx_freq(uhd.types.TuneRequest(carrier_f), 0)
    dev_tx.set_tx_gain(gain_tx, 0)
    dev_rx.set_rx_gain(gain_rx, 0)
    dev_tx.set_tx_antenna("TX/RX", 0)
    dev_rx.set_rx_antenna("RX2", 0)
    print("Waiting for Local Oscillators to lock...")
    max_wait = time.time() + 2.0
    while not dev_tx.get_tx_sensor("lo_locked", 0).to_bool() and time.time() < max_wait:
        time.sleep(0.01)
    while not dev_rx.get_rx_sensor("lo_locked", 0).to_bool() and time.time() < max_wait:
        time.sleep(0.01)
    print("Hardware PLLs Locked.")
    args_tx = uhd.usrp.StreamArgs("fc32", "sc12")
    args_tx.channels = [0]
    stream_tx = dev_tx.get_tx_stream(args_tx)
    args_rx = uhd.usrp.StreamArgs("fc32", "sc12")
    args_rx.channels = [0]
    stream_rx = dev_rx.get_rx_stream(args_rx)
    return stream_tx, stream_rx

def run_radio(stream_tx, stream_rx, data_tx, total_rx_size, delay_tx=2.0):
    captured_data = np.zeros(total_rx_size, dtype=np.complex64)
    def rx_proc():
        cmd_start = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
        cmd_start.stream_now = True
        stream_rx.issue_stream_cmd(cmd_start)
        rx_info = uhd.types.RXMetadata()
        tmp_2d = np.zeros((1, stream_rx.get_max_num_samps()), dtype=np.complex64)
        got_samps = 0
        while got_samps < total_rx_size:
            n_samps = stream_rx.recv(tmp_2d, rx_info, 5.0)
            if n_samps == 0:
                time.sleep(0.001)
                continue
            pos_end = min(got_samps + n_samps, total_rx_size)
            captured_data[got_samps:pos_end] = tmp_2d[0, : pos_end - got_samps]
            got_samps += n_samps
        cmd_stop = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
        stream_rx.issue_stream_cmd(cmd_stop)
    def tx_proc():
        time.sleep(delay_tx)
        tx_info = uhd.types.TXMetadata()
        tx_info.has_time_spec = False
        tx_info.start_of_burst = True
        chunk_max = stream_tx.get_max_num_samps()
        sent_samps = 0
        total_tx_size = len(data_tx)
        tx_info.end_of_burst = False
        while sent_samps < total_tx_size:
            n_samps = min(total_tx_size - sent_samps, chunk_max)
            if sent_samps + n_samps == total_tx_size:
                tx_info.end_of_burst = True
            block_tx = data_tx[sent_samps : sent_samps + n_samps][np.newaxis, :]
            stream_tx.send(block_tx, tx_info)
            tx_info.start_of_burst = False
            sent_samps += n_samps
    th_rx = threading.Thread(target=rx_proc)
    th_tx = threading.Thread(target=tx_proc)
    th_rx.start()
    th_tx.start()
    th_rx.join()
    th_tx.join()
    return captured_data

def proc_pipe(raw_signal, base_pat, head_bits, head_syms, mix_syms, rx_filter, sampling_f, sps, num_span, runs):
    global stat_store
    def sync_find(vec_in):
        ref_wave = np.repeat(head_syms, sps)
        corr_out = signal.correlate(vec_in, ref_wave, mode='full')
        corr_lags = signal.correlation_lags(len(vec_in), len(ref_wave), mode='full')
        peak_pos = max(0, int(corr_lags[np.argmax(np.abs(corr_out))]))
        return peak_pos, corr_out, corr_lags

    def freq_est(vec_in, base_offset):
        block_len = len(base_pat) * sps
        cross_prod = np.conj(vec_in[:-block_len]) * vec_in[block_len:]
        box_win = np.ones(block_len)
        smooth_r = np.convolve(cross_prod, box_win, mode='valid')
        samp_pos = base_offset + block_len
        if samp_pos < len(smooth_r):
            ph_delta = np.angle(smooth_r[samp_pos])
        else:
            ph_delta = 0.0
        cfo_step = ph_delta / block_len
        freq_hz = cfo_step * sampling_f / (2 * np.pi)
        metric_m = (np.abs(smooth_r) ** 2)
        return cfo_step, freq_hz, metric_m, samp_pos

    def chan_eval(vec_in, base_offset):
        start_p = base_offset + (sps // 2)
        count_p = len(head_syms)
        extracted_p = np.zeros(count_p, dtype=np.complex128)
        for idx in range(count_p):
            t_pos = start_p + (idx * sps)
            if t_pos < len(vec_in):
                extracted_p[idx] = vec_in[t_pos]
        h_val = np.mean(extracted_p * np.conj(head_syms))
        return h_val if np.abs(h_val) > 0.05 else 1.0 + 0j

    # def phase_track(sig_raw, sig_filtered, base_offset, h_val):
    #     total_syms = len(mix_syms)
    #     out_syms = np.zeros(total_syms, dtype=np.complex128)
    #     out_bits = np.zeros(total_syms * BITS_PER_SYM, dtype=int)
    #     hist_p = np.zeros(total_syms)
    #     p_val, f_val = 0.0, 0.0
        
    #     # Tightened tracking loops for smaller 8-PSK decision margins
    #     zeta = np.sqrt(2.0) / 2.0
    #     bw_w = 0.05
    #     bw_n = 0.01
    #     d_w = 1.0 + 2.0 * zeta * bw_w + bw_w**2
    #     kp_w, ki_w = (4.0 * zeta * bw_w) / d_w, (4.0 * bw_w**2) / d_w
    #     d_n = 1.0 + 2.0 * zeta * bw_n + bw_n**2
    #     kp_n, ki_n = ((4.0 * zeta * bw_n) / d_n, (4.0 * bw_n**2) / d_n)
    #     idx_p = base_offset + (sps // 2)
    #     idx_d = base_offset + (len(head_syms) * sps) + (num_span * sps)
        
    #     for idx in range(total_syms):
    #         kp, ki = (kp_w, ki_w) if idx < len(head_syms) else (kp_n, ki_n)
    #         if idx < len(head_syms):
    #             t_pos = idx_p + (idx * sps)
    #             target_sig = sig_raw / (h_val if np.abs(h_val) > 1e-12 else 1)
    #         else:
    #             t_pos = idx_d + ((idx - len(head_syms)) * sps)
    #             target_sig = sig_filtered
                
    #         if t_pos < len(target_sig):
    #             rot_sym = target_sig[t_pos] * np.exp(-1j * p_val)
    #             out_syms[idx] = rot_sym
                
    #             # Polymorphic M-PSK Decision Slicer
    #             ang = np.angle(rot_sym)
    #             if ang < 0:
    #                 ang += 2 * np.pi
    #             p_idx = int(np.round(ang / (2 * np.pi / M))) % M
    #             sliced_sym = np.exp(1j * 2 * np.pi * p_idx / M)
                
    #             # Universal Decision-Directed Loop Discriminator
    #             err_metric = np.imag(rot_sym * np.conj(sliced_sym))
                
    #             f_val += ki * err_metric
    #             p_val += kp * err_metric + f_val
    #             p_val = (p_val + np.pi) % (2 * np.pi) - np.pi
    #             hist_p[idx] = p_val
                
    #             # Unpack gray phase slices back to bit elements
    #             val_int = gray_to_bin_scalar(p_idx)
    #             for b_idx in range(BITS_PER_SYM):
    #                 out_bits[idx*BITS_PER_SYM + b_idx] = (val_int >> (BITS_PER_SYM - 1 - b_idx)) & 1

    #     # M-Fold Phase Ambiguity Compensator
    #     best_rot = 0
    #     min_errors = len(head_bits) + 1
    #     for k_rot in range(M):
    #         test_bits = np.zeros(len(head_bits), dtype=int)
    #         for h_idx in range(len(head_syms)):
    #             ang = np.angle(out_syms[h_idx] * np.exp(-1j * 2 * np.pi * k_rot / M))
    #             if ang < 0:
    #                 ang += 2 * np.pi
    #             p_idx = int(np.round(ang / (2 * np.pi / M))) % M
    #             val_int = gray_to_bin_scalar(p_idx)
    #             for b_idx in range(BITS_PER_SYM):
    #                 test_bits[h_idx*BITS_PER_SYM + b_idx] = (val_int >> (BITS_PER_SYM - 1 - b_idx)) & 1
            
    #         errors = np.sum(test_bits != head_bits)
    #         if errors < min_errors:
    #             min_errors = errors
    #             best_rot = k_rot
                
    #     if best_rot != 0:
    #         out_syms = out_syms * np.exp(-1j * 2 * np.pi * best_rot / M)
    #         hist_p = (hist_p + (2 * np.pi * best_rot / M) + np.pi) % (2 * np.pi) - np.pi
    #         for idx in range(total_syms):
    #             ang = np.angle(out_syms[idx])
    #             if ang < 0:
    #                 ang += 2 * np.pi
    #             p_idx = int(np.round(ang / (2 * np.pi / M))) % M
    #             val_int = gray_to_bin_scalar(p_idx)
    #             for b_idx in range(BITS_PER_SYM):
    #                 out_bits[idx*BITS_PER_SYM + b_idx] = (val_int >> (BITS_PER_SYM - 1 - b_idx)) & 1
                
    #     return out_bits, out_syms, hist_p

    def phase_track(sig_raw, sig_filtered, base_offset, h_val):
        total_syms = len(mix_syms)
        out_syms = np.zeros(total_syms, dtype=np.complex128)
        out_bits = np.zeros(total_syms * BITS_PER_SYM, dtype=int)
        hist_p = np.zeros(total_syms)
        p_val, f_val = 0.0, 0.0
        
        zeta = np.sqrt(2.0) / 2.0
        bw_w = 0.08
        bw_n = 0.015
        d_w = 1.0 + 2.0 * zeta * bw_w + bw_w**2
        kp_w, ki_w = (4.0 * zeta * bw_w) / d_w, (4.0 * bw_w**2) / d_w
        d_n = 1.0 + 2.0 * zeta * bw_n + bw_n**2
        kp_n, ki_n = ((4.0 * zeta * bw_n) / d_n, (4.0 * bw_n**2) / d_n)
        idx_p = base_offset + (sps // 2)
        idx_d = base_offset + (len(head_syms) * sps) + (num_span * sps)
        
        for idx in range(total_syms):
            kp, ki = (kp_w, ki_w) if idx < len(head_syms) else (kp_n, ki_n)
            if idx < len(head_syms):
                t_pos = idx_p + (idx * sps)
                target_sig = sig_raw / (h_val if np.abs(h_val) > 1e-12 else 1)
            else:
                t_pos = idx_d + ((idx - len(head_syms)) * sps)
                target_sig = sig_filtered
                
            if t_pos < len(target_sig):
                rot_sym = target_sig[t_pos] * np.exp(-1j * p_val)
                out_syms[idx] = rot_sym
                
                # UPDATED: Slicer handles the 45-degree offset cleanly
                ang = np.angle(rot_sym) - (np.pi / M)
                if ang < 0:
                    ang += 2 * np.pi
                p_idx = int(np.round(ang / (2 * np.pi / M))) % M
                sliced_sym = np.exp(1j * ((2 * np.pi * p_idx / M) + (np.pi / M)))
                
                # UPDATED: Standard robust Costas loop discriminator
                err_metric = np.imag(rot_sym * np.conj(sliced_sym))
                
                f_val += ki * err_metric
                p_val += kp * err_metric + f_val
                p_val = (p_val + np.pi) % (2 * np.pi) - np.pi
                hist_p[idx] = p_val
                
                val_int = gray_to_bin_scalar(p_idx)
                for b_idx in range(BITS_PER_SYM):
                    out_bits[idx*BITS_PER_SYM + b_idx] = (val_int >> (BITS_PER_SYM - 1 - b_idx)) & 1

        # UPDATED: Phase Ambiguity Resolver tracking the shifted coordinates
        best_rot = 0
        min_errors = len(head_bits) + 1
        for k_rot in range(M):
            test_bits = np.zeros(len(head_bits), dtype=int)
            for h_idx in range(len(head_syms)):
                ang = np.angle(out_syms[h_idx] * np.exp(-1j * 2 * np.pi * k_rot / M)) - (np.pi / M)
                if ang < 0:
                    ang += 2 * np.pi
                p_idx = int(np.round(ang / (2 * np.pi / M))) % M
                val_int = gray_to_bin_scalar(p_idx)
                for b_idx in range(BITS_PER_SYM):
                    test_bits[h_idx*BITS_PER_SYM + b_idx] = (val_int >> (BITS_PER_SYM - 1 - b_idx)) & 1
            
            errors = np.sum(test_bits != head_bits)
            if errors < min_errors:
                min_errors = errors
                best_rot = k_rot
                
        if best_rot != 0:
            out_syms = out_syms * np.exp(-1j * 2 * np.pi * best_rot / M)
            hist_p = (hist_p + (2 * np.pi * best_rot / M) + np.pi) % (2 * np.pi) - np.pi
            for idx in range(total_syms):
                ang = np.angle(out_syms[idx]) - (np.pi / M)
                if ang < 0:
                    ang += 2 * np.pi
                p_idx = int(np.round(ang / (2 * np.pi / M))) % M
                val_int = gray_to_bin_scalar(p_idx)
                for b_idx in range(BITS_PER_SYM):
                    out_bits[idx*BITS_PER_SYM + b_idx] = (val_int >> (BITS_PER_SYM - 1 - b_idx)) & 1
                
        return out_bits, out_syms, hist_p

    def step_run(vec_in):
        sorted_mags = np.sort(np.abs(vec_in))
        peak_thresh = sorted_mags[int(0.999 * len(sorted_mags))] if len(sorted_mags) > 0 else 1.0
        if peak_thresh == 0:
            peak_thresh = 1.0
        norm_sig = vec_in / peak_thresh
        base_offset, corr_m, corr_l = sync_find(norm_sig)
        cfo_step, freq_hz, metric_m, sc_peak = freq_est(norm_sig, base_offset)
        grid_t = np.arange(len(norm_sig))
        coarse_adj = norm_sig * np.exp(-1j * cfo_step * grid_t)
        filt_adj = np.convolve(coarse_adj, rx_filter, 'full')
        h_val = chan_eval(coarse_adj, base_offset)
        filt_eq = filt_adj / h_val if np.abs(h_val) > 0.05 else filt_adj
        idx_d = base_offset + (len(head_syms) * sps) + (num_span * sps)
        check_syms = []
        for idx in range(min(40, len(mix_syms) - len(head_syms))):
            t_pos = idx_d + (idx * sps)
            if t_pos < len(filt_eq):
                check_syms.append(np.abs(filt_eq[t_pos]))
        if len(check_syms) > 0:
            avg_gain = np.mean(check_syms)
            filt_norm = filt_eq / avg_gain
        else:
            filt_norm = filt_eq

        unfilt_points = []
        for idx in range(len(mix_syms) - len(head_syms)):
            t_pos = idx_d + (idx * sps)
            if t_pos < len(coarse_adj):
                unfilt_points.append(coarse_adj[t_pos] / (h_val if np.abs(h_val) > 0.05 else 1.0))

        bits_out, loop_syms, hist_p = phase_track(coarse_adj, filt_norm, base_offset, h_val)
        cont_p = np.repeat(hist_p, sps)
        full_p_corr = np.zeros(len(filt_norm))
        idx_p = base_offset + (sps // 2)
        pos_end = min(len(filt_norm), idx_p + len(cont_p))
        span_len = pos_end - idx_p
        full_p_corr[idx_p:pos_end] = cont_p[:span_len]
        filt_norm = filt_norm * np.exp(-1j * full_p_corr)
        tail_pad = num_span * sps * 2
        bound_end = idx_d + ((len(mix_syms) - len(head_syms)) * sps) + tail_pad
        seg_raw = norm_sig[base_offset:bound_end]
        seg_filt = filt_norm[idx_d - (num_span * sps):bound_end]

        stat_store['corr_lags'] = corr_l
        stat_store['corr_mag'] = corr_m
        stat_store['corr_anchor'] = base_offset
        stat_store['metric_m'] = metric_m
        stat_store['sc_peak'] = sc_peak
        stat_store['loop_phase'] = hist_p
        stat_store['seg_raw'] = seg_raw
        stat_store['seg_filt'] = seg_filt
        stat_store['loop_syms'] = loop_syms
        stat_store['unfilt_points'] = np.array(unfilt_points)
        stat_store['ml_signal_stage'] = filt_norm[base_offset : bound_end]
        return coarse_adj, freq_hz, h_val, bits_out

    active_sig = np.copy(raw_signal)
    bits_result = np.zeros(len(mix_syms) * BITS_PER_SYM)
    total_cfo, active_h = 0.0, 1.0 + 0j
    for r_idx in range(runs):
        active_sig, step_cfo, active_h, bits_result = step_run(active_sig)
        total_cfo += step_cfo
    info_extracted_bits = bits_result[len(head_bits):]
    return info_extracted_bits, total_cfo, active_h

def start_app():
    dataset_dir = "rf_dataset"
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
        print(f"Created central dataset directory: {dataset_dir}")
        
    base_pat = np.array([1, 1, 0, 0] * BITS_PER_SYM)
    sync_bits = np.array([0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0] * BITS_PER_SYM)
    
    # FIXED BALANCING FORMULA: 250 target windows * 1024 size * 3 bits/sym = 768,000 bits
    info_bits = np.random.randint(0, 2, 600000)
    
    fs_rate = 20e6
    carrier_f = 1.5e9
    gain_tx = 60
    gain_rx = 30
    global sps 
    sps = 4
    runs = 1
    tx_delay = 0.01

    rx_duration = 0.08
    print(f"Tx sends after {tx_delay} s")
    print(f"Rx listens for {rx_duration} s")
    matched_f = make_rrc(rolloff, span, sps)
    data_tx, head_bits, mix_syms, scaled_payload = build_frame(
        info_bits, base_pat, sync_bits, sps, matched_f, 0.8
    )
    stream_tx, stream_rx = init_hw(
        tx_id="344C4FE", rx_id="34B1945", fs_rate=20e6, carrier_f=carrier_f, gain_tx=gain_tx, gain_rx=gain_rx
    )
    print(f"\nExecuting Asynchronous Continuous OTA Tx/Rx at {carrier_f/1e9} GHz using {M}-PSK...")
    total_samples = int(rx_duration * fs_rate)
    raw_buffer = run_radio(stream_tx, stream_rx, data_tx.astype(np.complex64), total_samples, tx_delay)
    print("Capture complete. Processing Signal...")
    skip_samples = int(fs_rate * 0.15)
    if len(raw_buffer) > skip_samples:
        clean_rx = raw_buffer[skip_samples:]
        centered_buffer = raw_buffer - np.mean(clean_rx)
        centered_buffer[:skip_samples] = 0
    else:
        centered_buffer = raw_buffer - np.mean(raw_buffer)
        
    info_extracted_bits = info_bits
    info_extracted_bits, total_cfo, active_h = proc_pipe(
        raw_signal=centered_buffer, base_pat=base_pat, head_bits=head_bits,
        head_syms=map_bits(head_bits), mix_syms=mix_syms,
        rx_filter=matched_f, sampling_f=fs_rate, sps=sps, num_span=span, runs=runs)
    bit_errors = np.sum(info_bits != info_extracted_bits[:len(info_bits)])
    print(f" CFO Est: {total_cfo:.2f} Hz")
    print(f" Phase :      {np.angle(active_h):.4f} rad")
    print(f"BER: {bit_errors} / {len(info_bits)}")

    # ml_complex_data = stat_store['ml_signal_stage']
    
    # # DYNAMIC TRUNCATION/PADDING SYSTEM - Enforces uniform CNN matrix outputs
    # TARGET_WINDOWS = 250  
    # WINDOW_SIZE = 1024
    # REQUIRED_LENGTH = TARGET_WINDOWS * WINDOW_SIZE # Exactly 256,000 matrix columns
    
    # current_length = len(ml_complex_data)
    # if current_length >= REQUIRED_LENGTH:
    #     ml_complex_data = ml_complex_data[:REQUIRED_LENGTH]
    # else:
    #     ml_complex_data = np.concatenate((ml_complex_data, np.zeros(REQUIRED_LENGTH - current_length, dtype=ml_complex_data.dtype)))
    
    # # Split the complex float values into explicit In-Phase and Quadrature rows
    # ml_ready_matrix = np.vstack((np.real(ml_complex_data), np.imag(ml_complex_data)))
    
    # # Save formatted array out to disk
    # file_path = os.path.join("rf_dataset", "ml_16psk_samples.npy")
    # np.save(file_path, ml_ready_matrix)
    # print(f">>> ML Feature Capture Saved! Shape: {ml_ready_matrix.shape} to '{file_path}'")

    all_tracked_symbols = stat_store['loop_syms']
    
    # 2. Skip the header symbols to isolate just your actual payload data
    header_length_symbols = len(head_bits) // BITS_PER_SYM
    ml_complex_data = all_tracked_symbols[header_length_symbols:]
    
    # 3. Ensure it perfectly matches your expected 256,000 payload size
    REQUIRED_LENGTH = 150000
    current_length = len(ml_complex_data)
    
    if current_length >= REQUIRED_LENGTH:
        ml_complex_data = ml_complex_data[:REQUIRED_LENGTH]
    else:
        padding_needed = REQUIRED_LENGTH - current_length
        ml_complex_data = np.concatenate((ml_complex_data, np.zeros(padding_needed, dtype=ml_complex_data.dtype)))
    
    # 4. Split into explicit In-Phase and Quadrature rows
    ml_ready_matrix = np.vstack((np.real(ml_complex_data), np.imag(ml_complex_data)))
    
    # 5. Save perfectly formatted array out to disk
    file_path = os.path.join("data_equal", "ml_16psk_samples.npy")
    np.save(file_path, ml_ready_matrix)
    print(f">>> Full Synchronized IQ Payload Saved! Shape: {ml_ready_matrix.shape} to '{file_path}'")
    

    time_lags = stat_store['corr_lags'] / fs_rate
    corr_mags = np.abs(stat_store['corr_mag'])
    unfilt_pts = stat_store['unfilt_points']
    loop_syms = stat_store['loop_syms'][len(head_bits)//BITS_PER_SYM:]

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.2])

    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(time_lags, corr_mags, color='black')
    ax0.axvline(stat_store['corr_anchor'] / fs_rate, color='red', linestyle='--')
    ax0.set_title("Cross-Correlation Magnitude over Time Axis")
    ax0.set_xlabel("Time (seconds)")
    ax0.set_ylabel("Correlation Amplitude")
    ax0.grid(True)

    ax1 = fig.add_subplot(gs[1, 0])
    ax1.scatter(np.real(unfilt_pts), np.imag(unfilt_pts), color='red', alpha=0.6, edgecolors='k')
    ax1.axhline(0, color='black', linestyle=':')
    ax1.axvline(0, color='black', linestyle=':')
    ax1.set_title(f"Received {M}-PSK Points Before Equalization & Tracking")
    ax1.set_xlabel("In-Phase (I)")
    ax1.set_ylabel("Quadrature (Q)")
    ax1.set_xlim([-2.5, 2.5])
    ax1.set_ylim([-2.5, 2.5])
    ax1.grid(True)

    ax2 = fig.add_subplot(gs[1, 1])
    ax2.scatter(np.real(loop_syms), np.imag(loop_syms), color='green', alpha=0.6, edgecolors='k')
    ax2.axhline(0, color='black', linestyle=':')
    ax2.axvline(0, color='black', linestyle=':')
    ax2.set_title(f"Processed {M}-PSK Constellation Points (Tracked Loop)")
    ax2.set_xlabel("In-Phase (I)")
    ax2.set_ylabel("Quadrature (Q)")
    ax2.set_xlim([-2.5, 2.5])
    ax2.set_ylim([-2.5, 2.5])
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    start_app()