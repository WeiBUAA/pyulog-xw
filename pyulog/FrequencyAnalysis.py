#! /usr/bin/env python
"""
Display logged messages from an ULog file
"""

from __future__ import print_function

import argparse
import re
import os

from .core import ULog
from .px4  import PX4ULog
import matplotlib.pyplot as plt
from   matplotlib import colors as mcolors
import numpy as np
#pylint: disable=invalid-name

def main():
    parser = argparse.ArgumentParser(description='Display logged messages from an ULog file')
    parser.add_argument('filename', metavar='file.ulg', help='ULog input file')

    args = parser.parse_args()
    ulog_file_name = args.filename

    angular_rate_analyz(ulog_file_name)
    #accel_analysis(ulog_file_name)

def angular_rate_analyz(ulog_file_name):
    #time = [445.5, 465.5] #log_01
    #time = [428.25, 446.25] #log_12
    #time = [87.8, 105.8] #log_18
    time = [75, 113] #log_27
    filter_freq = [4.8, 5.2]
    filter_2freq = [15, 17]
    plt.figure(1, figsize=(18, 9))
    font2 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 12,
    }

    PX4 = PX4ULog()

    rr_fdb = PX4.read_data(ulog_file_name,'sensor_combined',0,'gyro_rad[0]',time[0],time[1])
    rr_fdb[0,0]
    Y_rr_fdb =PX4.fft_a(rr_fdb[0,:], rr_fdb[1,:] * 57.3)
    pr_fdb = PX4.read_data(ulog_file_name,'sensor_combined',0,'gyro_rad[1]',time[0],time[1])
    Y_pr_fdb =PX4.fft_a(pr_fdb[0,:], pr_fdb[1,:] * 57.3)
    yr_fdb = PX4.read_data(ulog_file_name,'sensor_combined',0,'gyro_rad[2]',time[0],time[1])
    Y_yr_fdb =PX4.fft_a(yr_fdb[0,:], yr_fdb[1,:] * 57.3)

    rr_sp = PX4.read_data(ulog_file_name,'vehicle_rates_setpoint',0,'roll',time[0],time[1])
    Y_rr_sp =PX4.fft_a(rr_sp[0,:], rr_sp[1,:] * 57.3)
    pr_sp = PX4.read_data(ulog_file_name,'vehicle_rates_setpoint',0,'pitch',time[0],time[1])
    Y_pr_sp =PX4.fft_a(pr_sp[0,:], pr_sp[1,:] * 57.3)
    yr_sp = PX4.read_data(ulog_file_name,'vehicle_rates_setpoint',0,'yaw',time[0],time[1])
    Y_yr_sp =PX4.fft_a(yr_sp[0,:], yr_sp[1,:] * 57.3)

    plt.subplot(3,2,1)
    plt.plot(rr_fdb[0,:], abs(rr_fdb[1,:]), color='r', linestyle='-', label = 'rollrate_fdb')
    plt.plot(rr_sp[0,:], abs(rr_sp[1,:]), color='b', linestyle='-', label = 'rollrate_sp')
    plt.legend()
    plt.subplot(3,2,3)
    plt.plot(Y_pr_fdb[0,:], abs(Y_pr_fdb[1,:]), color='r', linestyle='-', label = 'pitchrate_fdb')
    plt.legend()
    plt.subplot(3,2,5)
    plt.plot(Y_yr_fdb[0,:], abs(Y_yr_fdb[1,:]), color='r', linestyle='-', label = 'yawrate_fdb')
    plt.legend()
    plt.subplot(3,2,2)
    plt.plot(Y_rr_sp[0,:], abs(Y_rr_sp[1,:]), color='c', linestyle='-', label = 'rollrate_sp')
    plt.legend()
    plt.subplot(3,2,4)
    plt.plot(Y_pr_sp[0,:], abs(Y_pr_sp[1,:]), color='c', linestyle='-', label = 'pitchrate_sp')
    plt.legend()
    plt.subplot(3,2,6)
    plt.plot(Y_yr_sp[0,:], abs(Y_yr_sp[1,:]), color='c', linestyle='-', label = 'yawrate_sp')
    plt.legend()

    fig_name = ulog_file_name[0:6] + '-' + 'spectrum.jpg'
    #plt.savefig(fig_name, dpi=600)
    plt.show()

def accel_analysis(ulog_file_name):
    time = [558, 620] #log_27
    filter_freq = [4.8, 5.2]
    filter_2freq = [15, 17]
    plt.figure(1, figsize=(18, 9))
    font2 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 12,
    }

    PX4 = PX4ULog()

    plt.subplot(3,2,2)
    acc_bx_fdb = PX4.read_data(ulog_file_name,'sensor_accel',0, 'x',time[0], time[1])
    Y  =PX4.fft_a(acc_bx_fdb[0,:], acc_bx_fdb[1,:])
    plt.plot(Y[0,:], abs(Y[1,:]), color='r', linestyle='-', label = 'ax_fdb')
    plt.legend()
    plt.subplot(3,2,4)
    acc_by_fdb = PX4.read_data(ulog_file_name,'sensor_accel',0, 'y',time[0],time[1])
    acc_by_fdb -= np.mean(acc_by_fdb[1,:])
    Y  =PX4.fft_a(acc_by_fdb[0,:], acc_by_fdb[1,:])
    plt.plot(Y[0,:], abs(Y[1,:]), color='r', linestyle='-', label = 'ay_fdb')
    plt.legend()
    plt.subplot(3,2,6)
    acc_bz_fdb = PX4.read_data(ulog_file_name,'sensor_accel',0, 'z',time[0],time[1]) + 9.8
    Y  =PX4.fft_a(acc_bz_fdb[0,:], acc_bz_fdb[1,:])
    plt.plot(Y[0,:], abs(Y[1,:]), color='r', linestyle='-', label = 'az_fdb')
    plt.legend()

    fig_name = ulog_file_name[0:6] + '-' + 'spectrum.jpg'
    plt.savefig(fig_name, dpi=600)
    plt.show()