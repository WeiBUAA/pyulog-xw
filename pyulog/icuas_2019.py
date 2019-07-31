#! /usr/bin/env python

"""
Convert a ULog file into CSV file(s)
"""

from __future__ import print_function

import argparse
import re
import os
import matplotlib.pyplot as plt
from   matplotlib import colors as mcolors
import numpy as np
import cvxpy as cvx

from scipy.interpolate import spline
from scipy import signal
from .core import ULog


#pylint: disable=too-many-locals, invalid-name, consider-using-enumerate
def Tr(b):
    if len(b.shape) >= 2 :
        return b.T

    return b.reshape(1, b.shape[0])

def Loadtxt(a):
    b = np.loadtxt(a)
    return Tr(b)

def main():
    """Command line interface"""

    parser = argparse.ArgumentParser(description='Convert ULog to CSV')
    parser.add_argument('filename', metavar='file.ulg', help='ULog input file')

    parser.add_argument(
        '-m', '--messages', dest='messages',
        help=("Only consider given messages. Must be a comma-separated list of"
              " names, like 'sensor_combined,vehicle_gps_position'"))
    parser.add_argument('-d', '--delimiter', dest='delimiter', action='store',
                        help="Use delimiter in CSV (default is ',')", default=',')


    parser.add_argument('-o', '--output', dest='output', action='store',
                        help='Output directory (default is same as input file)',
                        metavar='DIR')

    args = parser.parse_args()

    if args.output and not os.path.isdir(args.output):
        print('Creating output directory {:}'.format(args.output))
        os.mkdir(args.output)

    #sweep(args.filename, args.output, 0, 210)
    #compare_pr()
    #vib_14hz_test(args.filename)
    #forward_test(args.filename)
    #notch_test(args.filename)
    #plot_aero()
    #height_test()
    acc_filter(args.filename)

    #system identification of ILC
    global sample_time
    sample_time = 2.5
def plot_aero():
    degree   = np.arange(91)
    CL_0     = np.loadtxt('CL_0.txt')/0.16
    CD_0     = np.loadtxt('CD_0.txt')/0.16
    
    font2 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 14,
    }
    plt.figure(1, figsize=(8, 5))
    plt.subplot(1,2,1)
    plt.plot(degree, CL_0)
    plt.xlim((0, 90))
    plt.ylim((0, 1.5))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('AOA('+r'${}^\circ$'+')',font2)
    plt.title(r'$C_{L}$',font2)
    #plt.ylabel(r'$C_{L}$',font2)

    fig_name = 'plot_cd_cl.jpg'
    plt.savefig(fig_name)

    plt.subplot(1,2,2)
    plt.plot(degree, CD_0)
    plt.xlim((0, 90))
    plt.ylim((0, 2.0))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('AOA('+r'${}^\circ$'+')',font2)
    plt.title(r'$C_{D}$',font2)
    #plt.ylabel(r'$C_{D}$',font2)

    fig_name = 'plot_cd_cl.jpg'
    plt.savefig(fig_name)

    plt.show()

def read_data(ulog_file_name,massage,channal,time_start,time_end):
    ulog = ULog(ulog_file_name, massage)
    d    = ulog.data_list
    
    if (channal == 'q_d'): #| (massage == 'sensor_accel')):
        ind_channel = 1
    else:
        ind_channel = 0
    print(d[0].name)

    times = d[ind_channel].data['timestamp'] / 1000000.0 #second
    #times = times - times[0]
    index = np.where((times>=time_start)&(times<=time_end))
    time_sub = times[index]
    print(time_sub.shape)
    q = np.zeros(shape=(len(time_sub),4))

    if ((channal == 'q') or (channal == 'q_d')):
        for i in range(0,4):
            data_name = channal + '['+str(i)+']'
            ori_data = d[ind_channel].data[data_name]
            #print(ori_data)
            q[:,i] = ori_data[index]
        roll  = np.arcsin(2.0 * (q[:,0] * q[:,1] + q[:,2] * q[:,3]))*57.3
        pitch = np.arctan2(-2.0 * (q[:,1] * q[:,3] - q[:,0] * q[:,2]), 1.0 - 2.0 * (q[:,1] * q[:,1] + q[:,2] * q[:,2]))*57.3
        yaw   = np.arctan2(-2.0 * (q[:,1] * q[:,2] - q[:,0] * q[:,3]), 1.0 - 2.0 * (q[:,1] * q[:,1] + q[:,3] * q[:,3]))*57.3
        dat   = np.vstack((time_sub,roll,pitch,yaw))
        
    else:
        data_name = channal
        pitchsp = d[ind_channel].data[data_name] #degree
        pitchsp_sub = pitchsp[index]
        dat = np.vstack((time_sub,pitchsp_sub))
    return dat
def fft_a(time,samp_sub):
    # for FFT analysis
    time_sub = time - time[0]
    n = len(time_sub)
    samp_fre = (n-1) / (time_sub[n-1] - time_sub[0])
    print(samp_fre)
    k = np.arange(n)
    T = n / samp_fre
    frq = k / T # two sides frequency range
    frq = frq[range(n/2)] # one side frequency range

    Y = np.fft.fft(samp_sub)/n # fft computing and normalization
    Y = Y[range(n/2)]
    return np.vstack((frq,Y))
def height_test():
    fig = plt.figure(1, figsize=(8, 5))
    font2 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 14,
    }
 
    ax1 = fig.add_subplot(111)
    after = read_data('after.ulg','vehicle_local_position','z',33, 37.45)
    before = read_data('before.ulg','vehicle_local_position','z',62.3, 66.75)
    height_a = (after[1,:])*(-1)
    height_b = (before[1,:])*(-1) - 20.0
    ax1.plot(after[0,:]-after[0,0], height_a,color='b', label = 'With Height Control')
    ax1.plot(before[0,:]-before[0,0], height_b,color='g', label = 'Without Height Control')
    ax1.set_xlabel('Time(s)',font2)
    ax1.set_ylabel('Height(m)',font2)
    ax1.set_ylim([25.0,50.0])
    plt.legend(prop=font2,loc= 'lower left')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
     
    ax2 = ax1.twinx()  # this is the important function
    atti_data = read_data('after.ulg','vehicle_attitude_setpoint','q_d',33, 37.45)
    ax2.plot(atti_data[0,:]-atti_data[0,0], atti_data[2,:]*1.05, color='r', label = 'PitchSP')
    ax2.set_ylabel('Angle('+r'${}^\circ$'+')',font2)
    plt.legend(prop=font2,loc= 'upper right')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    fig_name = 'plot_height.jpg'
    plt.savefig(fig_name)
    plt.show()

def notch_test(ulog_file_name):
    plt.figure(1, figsize=(9, 7))
    font2 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 14,
    }
    plt.subplot(3,1,1)
    with_notch = read_data(ulog_file_name,'vehicle_rates_setpoint','pitch',110, 150)
    b,a = signal.butter(1,2.0*83.0/250.0,'low')
    sf = signal.filtfilt(b,a,with_notch[1,:])
    Y1=fft_a(with_notch[0,:], with_notch[1,:])
    plt.plot(Y1[0,:], abs(Y1[1,:]), color='r', linestyle='-', label = 'PitchSP')
    #plt.ylim((0, 0.5))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylabel('PitchSP',font2)

    plt.subplot(3,1,2)
    without_notch = read_data(ulog_file_name,'vehicle_attitude','pitchspeed',110, 150)
    Y2=fft_a(without_notch[0,:], without_notch[1,:])
    plt.plot(Y2[0,:], abs(Y2[1,:]), color='r', linestyle='-', label = 'Pitch')
    #plt.ylim((0, 4))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Frequency(hz)',font2)
    plt.ylabel('Pitch',font2)

    plt.subplot(3,1,3)
    index = np.arange(300)
    closeloop = Y2[1,index]/Y1[1,index];
    plt.plot(Y2[0,index], abs(closeloop[:]), color='r', linestyle='-', label = 'SP-to-fdb')
    #plt.ylim((0, 4))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Frequency(hz)',font2)
    plt.ylabel('Pitch',font2)

    fig_name = str(15)
    plt.savefig(fig_name)
    plt.show()
    
'''
    plt.figure(2, figsize=(9, 7))
    plt.subplot(2,1,1)
    plt.plot(with_notch[0,:], sf)
    #plt.xlim((150,160))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylabel('With Notch and Low-Pass',font2)

    plt.subplot(2,1,2)
    plt.plot(without_notch[0,:], without_notch[1,:])
    #plt.xlim((70,80))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Time(s)',font2)
    plt.ylabel('Without Notch and Low-Pass',font2)

    fig_name = 'notch_test_time.jpg'
    plt.savefig(fig_name)
    '''

def vib_14hz_test(ulog_file_name):
    plt.figure(1, figsize=(9, 7))
    font2 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 14,
    }
    time_start=0
    time_end=250
    with_notch = read_data(ulog_file_name,'vehicle_attitude','pitchspeed',time_start, time_end)
    b,a = signal.butter(1,2.0*38.0/250.0,'low')
    sf = signal.filtfilt(b,a,with_notch[1,:])*57
    plt.plot(with_notch[0,:], sf, label = 'Feedback')

    cmd = read_data(ulog_file_name,'vehicle_rates_setpoint','pitch',time_start, time_end)
    plt.plot(cmd[0,:], cmd[1,:]*57, label = 'Command')

    plt.xlim((time_start, time_end))
    plt.ylim((-80,130))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Time(s)',font2)
    plt.ylabel('Pitch Angular Velocity(degree/s)',font2)

    plt.legend(prop=font2,loc= 'upper right')
    fig_name = 'notch_test_time.jpg'
    plt.savefig(fig_name)
    plt.show()


def forward_test(ulog_file_name):
    channel_name  = 'pitch'
    rpy_dic       = {'pitch':'[1]', 'roll':'[0]', 'yaw':'[2]'}
    rpy_index     = rpy_dic[channel_name]

    plt.figure(1, figsize=(9, 9))
    font2 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 12,
    }

    plt.subplot(4,1,1)
    atti_data = read_data(ulog_file_name,'vehicle_attitude_setpoint','q_d',60.5,90.5)
    plt.plot(atti_data[0,:], atti_data[2,:], label = 'PitchSP')
    atti_data = read_data(ulog_file_name,'vehicle_attitude','q',60.5,90.5)
    plt.plot(atti_data[0,:], atti_data[2,:], label = 'Pitch')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylabel('Pitch(degree)',font2)
    plt.legend(prop=font2,loc= 'lower right')

    plt.subplot(4,1,2)
    atti_data = read_data(ulog_file_name,'vehicle_attitude_setpoint','q_d',86,116)
    plt.plot(atti_data[0,:]-25.5, atti_data[1,:], label = 'RollSP')
    atti_data = read_data(ulog_file_name,'vehicle_attitude','q',86,116)
    plt.plot(atti_data[0,:]-25.5, atti_data[1,:], label = 'Roll')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylabel('Roll(degree)',font2)
    plt.legend(prop=font2,loc= 'lower right')

    plt.subplot(4,1,3)
    atti_data = read_data(ulog_file_name,'vehicle_attitude_setpoint','q_d',88,118)
    plt.plot(atti_data[0,:]-27.5, atti_data[3,:], label = 'YawSP')
    atti_data = read_data(ulog_file_name,'vehicle_attitude','q',88,118)
    plt.plot(atti_data[0,:]-27.5, atti_data[3,:], label = 'Yaw')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Time(s)',font2)
    plt.ylabel('Yaw(degree)',font2)
    plt.legend(prop=font2,loc= 'lower right')

    plt.subplot(4,1,4)
    after = read_data('after.ulg','vehicle_local_position','z',20, 50)
    height_a = ((after[1,:])*(-1) + 100.0)*0.4
    time = after[0,:]-after[0,0]+atti_data[0,0]-27.5
    plt.plot(time, height_a,color='b', label = 'With Height Control')
    plt.ylim((45,55))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Time(s)',font2)
    plt.ylabel('Altitude(m)',font2)

    #plt.xlim((65, 85))
    fig_name = 'forward_test.jpg'
    plt.savefig(fig_name,dpi=600)
    plt.show()


def sweep(ulog_file_name, output, time_start, time_end):
    """
    :param ulog_file_name: The ULog filename to open and read
    :param messages: A list of message names
    :param output: Output file path
    :param delimiter: CSV delimiter

    :return: None
    """
    messages   = 'vehicle_rates_setpoint,vehicle_attitude,actuator_controls_0'
    msg_filter = messages.split(',') if messages else None
    
    num_circle = 0;
    channel_name  = 'pitch'
    rpy_dic       = {'pitch':'[1]', 'roll':'[0]', 'yaw':'[2]'}
    rpy_index     = rpy_dic[channel_name]


    for i in range(0,len(msg_filter)):
        ulog = ULog(ulog_file_name, msg_filter[i])
        d    = ulog.data_list

        num_circle = num_circle + 1
        print(num_circle, d[0].name)
        data_name_dic = {'vehicle_rates_setpoint':channel_name, 'vehicle_attitude': channel_name + 'speed', 'actuator_controls_0': 'control' + rpy_index}
        plot_name_dic = {'vehicle_rates_setpoint':channel_name +'rate_cmd', 'vehicle_attitude': channel_name + 'rate_fdb','actuator_controls_0':channel_name+'rate_ctrloutput'}
        data_name     = data_name_dic[d[0].name]

        times = d[0].data['timestamp'] / 1000000.0 #second
        times = times - times[0]

        # get origin data
        pitchsp = d[0].data[data_name] * 57.3 #degree

        # get time subsection
        index = np.where((times>=time_start)&(times<=time_end))
        
        time_sub = times[index]
        #print(times)
        pitchsp_sub = pitchsp[index]

        # for FFT analysis
        time_sub = time_sub - time_sub[0]
        n = len(time_sub)
        samp_fre = (n-1) / (time_sub[n-1] - time_sub[0])
        k = np.arange(n)
        T = n / samp_fre
        frq = k / T # two sides frequency range
        frq = frq[range(n/2)] # one side frequency range

        Y = np.fft.fft(pitchsp_sub)/n # fft computing and normalization
        Y = Y[range(n/2)]
        
        plt.figure(0,figsize=(9, 4))
        plt.subplot(3,1,num_circle)
        plt.plot(frq, abs(Y), color='r', linestyle='-')
        plot_name = plot_name_dic[d[0].name]
        plt.title(plot_name)

    
        leg_dic = {'vehicle_attitude': r'$\omega$', 'actuator_controls_0': r'$u_{in}$'}
        if data_name != 'pitch':
            plt.figure(1, figsize=(9, 4.5))
            plt.plot(time_sub, pitchsp_sub, label = leg_dic[d[0].name])
            font2 = {'family' : 'Times New Roman',
            'weight' : 'normal',
            'size'   : 14,
            }
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.xlabel('time(s)',font2)
            plt.legend(prop=font2)
            fig_name = 'sysident-in-out.jpg'
            plt.savefig(fig_name)
            #plt.title(plot_name)
        
    #chirp signal
    plt.show()
    
    plt.figure(2, figsize=(9, 4.5))
    sweep_input = read_data(ulog_file_name,'actuator_controls_0','sweep_input',50, 210)
    plt.subplot(3,1,1)
    plt.plot(sweep_input[0,:],sweep_input[1,:])
    plt.subplot(3,1,2)
    Y=fft_a(sweep_input[0,:], sweep_input[1,:])
    plt.plot(Y[0,:], abs(Y[1,:]))
    plt.subplot(3,1,3)
    plt.psd(sweep_input[1,:], 512, 250)

    #plt.title(plot_name)
    font2 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 14,
    }
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('time(s)', font2)
    fig_name = 'chirp_sig.jpg' 
    plt.savefig(fig_name)
    
    plt.show()

def compare_pr():
    time = [73.5, 79.5]
    plt.figure(1, figsize=(9, 6))
    font2 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 12,
    }

    ulog_file_name = 'after_lpsp.ulg'
    '''
    plt.subplot(2,1,1)
    atti_data = read_data(ulog_file_name,'vehicle_attitude_setpoint','q_d',time[0],time[1])
    plt.plot(atti_data[0,:], atti_data[2,:], label = 'PitchSP')
    atti_data = read_data(ulog_file_name,'vehicle_attitude','q',time[0],time[1])
    plt.plot(atti_data[0,:], atti_data[2,:], label = 'Pitch')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylabel('Pitch(degree)',font2)
    plt.legend(prop=font2,loc= 'lower right')
    '''

    #plt.subplot(2,1,2)
    with_notch = read_data(ulog_file_name,'vehicle_attitude','pitchspeed',time[0],time[1])
    b,a = signal.butter(1,2.0*90/250.0,'low')
    sf = signal.filtfilt(b,a,with_notch[1,:])*57*0.94
    plt.plot(with_notch[0,:], sf, label = 'Feedback')
    cmd = read_data(ulog_file_name,'vehicle_rates_setpoint','pitch',time[0],time[1])
    plt.plot(cmd[0,:], cmd[1,:]*57, label = 'Command')
    plt.xlim((time[0],time[1]))
    plt.ylim((-220,260))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Time(s)',font2)
    plt.ylabel('Pitch Angular Velocity(degree/s)',font2)
    plt.legend(prop=font2,loc= 'upper right')

    fig_name = 'compare_pr.jpg'
    plt.savefig(fig_name, dpi=600)
    plt.show()    

def acc_filter(ulog_file_name):
    plt.figure(1, figsize=(18, 7))
    font2 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 14,
    }

    time = [136.0, 190.0]#[131.25, 136.0]

    plt.subplot(2,2,1)
    acc_fdb = read_data(ulog_file_name,'sensor_accel', 'z',time[0], time[1])
    
    Y=fft_a(acc_fdb[0,:], acc_fdb[1,:])
    plt.plot(Y[0,:], abs(Y[1,:]), color='r', linestyle='-', label = 'With Notch')
    #plt.ylim((0, 0.5))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylabel('acc_z_orig_spectrum',font2)

    plt.subplot(2,2,2)
    topic   = 'vehicle_attitude'
    channal = ['pitchspeed', 'rollspeed', 'yawspeed']

    pr_fdb = read_data(ulog_file_name,topic,channal[0],time[0], time[1])
    rr_fdb = read_data(ulog_file_name,topic,channal[1],time[0], time[1])
    yr_fdb = read_data(ulog_file_name,topic,channal[2],time[0], time[1])

    Y=fft_a(pr_fdb[0,:], rr_fdb[1,:])
    plt.plot(Y[0,:], abs(Y[1,:]), color='r', linestyle='-', label = 'With Notch')
    #plt.ylim((0, 4))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Frequency(hz)',font2)
    plt.ylabel(topic,font2)

    fig_name = 'notch_test.jpg'
    plt.savefig(fig_name)
    
    #plt.figure(2, figsize=(9, 7))
    plt.subplot(2,2,3)
    plt.plot(acc_fdb[0,:], acc_fdb[1,:])

    lp_frequency = 18.0;
    b,a = signal.butter(1,2.0*lp_frequency/250.0,'low')
    acc_filt = signal.filtfilt(b,a,acc_fdb[1,:])

    notch_b, notch_a = signal.iirnotch(12.7, 3, 250.0)
    acc_filt2 = signal.filtfilt(notch_b,notch_a,acc_filt)
    plt.plot(acc_fdb[0,:], acc_filt2, label = 'With Notch')

    b1,a1 = signal.butter(1,2.0*lp_frequency/250.0,'low')
    pr_filt = signal.filtfilt(b1,a1,pr_fdb[1,:])
    plt.plot(pr_fdb[0,:], pr_filt)

    b2,a2 = signal.butter(1,2.0*lp_frequency/250.0,'low')
    rr_filt = signal.filtfilt(b2,a2,rr_fdb[1,:])
    plt.plot(rr_fdb[0,:], rr_filt)

    b3,a3 = signal.butter(1,2.0*lp_frequency/250.0,'low')
    yr_filt = signal.filtfilt(b3,a3,yr_fdb[1,:])
    plt.plot(yr_fdb[0,:], yr_filt)
    #plt.xlim((150,160))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylabel('acc and angular rates',font2)

    plt.subplot(2,2,4)
    plt.plot(pr_fdb[0,:], pr_fdb[1,:])
    #plt.xlim((70,80))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Time(s)',font2)
    plt.ylabel('Without Notch and Low-Pass',font2)

    fig_name = 'notch_test_time.jpg'
    plt.savefig(fig_name)

    plt.show()