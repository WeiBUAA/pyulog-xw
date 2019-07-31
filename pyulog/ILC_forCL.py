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
from .icuas_2019 import read_data
from .icuas_2019 import fft_a

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

    #calculate the fft spectrum of rate control loop
    #rate_fft(args.filename, args.output, 30.0, 160.0)

    
    #system identification of ILC
    global sample_time
    sample_time = 6.0
    #ILC_sysident()
    ILC_cal_F_matrix()
    #ILC()
    #acc_filter(args.filename)

def get_cur_y(num):
    ulogname = 'ILC-' + str(num) + '.ulg'
    ulog1 = ULog(ulogname, 'vtol_vehicle_status')
    ulog2 = ULog(ulogname, 'vehicle_local_position')
    d1    = ulog1.data_list
    d2    = ulog2.data_list
    #print(d1[1].name, d2[0].name,len(d1[1].data['timestamp']),len(d2[0].data['timestamp']))

    times_trans = d1[0].data['timestamp'] / 1000000.0 #second
    ticks_trans = d1[0].data['in_transition_to_fw']
    
    index       = np.where(ticks_trans >= 1)
    print(index)
    time_start  = (times_trans[index])[0]
    
    
    # get altitude error inside the trans time
    times_local      = d2[0].data['timestamp'] / 1000000.0 #second
    alt              = d2[0].data['vz'] * (-1.0)
    lenth            = int(sample_time * 125)
    index            = np.where(times_local >= time_start)
    times_local_sub  = times_local[index]
    times_local_sub  = times_local_sub[np.arange(lenth)] - times_local_sub[0]
    alt_ref          = alt[index]
    alt_ref          = alt_ref[np.arange(lenth)] - alt_ref[0]

    return alt_ref


def ILC():
    #** set the number of iterations **#
    colors = ['k','r','y','g','b','c','m','pink','lime','burlywood']
    degree_samp = np.array([0,10,15,20,25,30,35,40,50,60,75,90]) #[0-,10-,15-,20,25,30,35,40,50,60,75,90-];
    degree      = np.arange(91)
    Y_NUM = int(sample_time * 125)
    U_NUM = len(degree_samp)
    
    F      = np.loadtxt('F.txt')
    #F[:,4] = - F[:,4]
    #** step1: read the global variables from dir **#
    for i in range(0, 1):
        Num_iteration = i
        print(i)
        CL_0     = np.loadtxt('CL_0.txt')
        u_norm   = np.loadtxt('CL_INPUT.txt');
        y_norm   = np.loadtxt('err_0.txt');
        
        
    #** step2: use Kalman Filter to estimate the disturbance **#
        process_var = 0.05;
        observe_var = 0.1;
        if (Num_iteration == 0):
            KF_P      = np.zeros(Y_NUM)
            KF_O      = np.zeros(Y_NUM)
            KF_K      = np.zeros(Y_NUM)
            KF_Dist   = np.zeros(Y_NUM)
            delt_y    = np.zeros(Y_NUM)
            delt_u    = np.zeros(U_NUM)

            np.savetxt('KF_P.txt',    KF_P,    fmt='%.3e')
            np.savetxt('KF_O.txt',    KF_O,    fmt='%.3e')
            np.savetxt('KF_K.txt',    KF_K,    fmt='%.3e')
            np.savetxt('KF_Dist.txt', KF_Dist, fmt='%.3e')
            np.savetxt('delt_y.txt',  delt_y,  fmt='%.3e')
            np.savetxt('delt_u.txt',  delt_u,  fmt='%.3e')

            Num_iter    = 1

            KF_P_cur    = KF_P
            KF_Dist_cur = KF_Dist
            u_cur       = delt_u[np.arange(U_NUM)]
            y_cur       = delt_y

            plt.figure(11)
            plt_time = np.arange(Y_NUM)/125.0
            plt.plot(plt_time, y_norm, color=colors[i], label='y_norm')
            plt.legend(loc='upper right')

        else:
            KF_P      = np.loadtxt('KF_P.txt')
            KF_O      = np.loadtxt('KF_O.txt')
            KF_K      = np.loadtxt('KF_K.txt')
            KF_Dist   = np.loadtxt('KF_Dist.txt')
            delt_y    = np.loadtxt('delt_y.txt')
            delt_u    = np.loadtxt('delt_u.txt')

            Num_iter    = KF_P.shape[1]

            KF_P_cur    = KF_P[:, Num_iter - 1]
            KF_Dist_cur = KF_Dist[:, Num_iter - 1]
            u_cur       = delt_u[np.arange(U_NUM), Num_iter - 1]
            y_cur       = get_cur_y(Num_iteration) - y_norm

            plt.figure(11)
            plt.plot(plt_time, y_cur + y_norm, color=colors[i], linestyle='-', label=str(i))
            plt.legend(loc='upper right')

        KF_P_NEXT    = KF_P_cur + process_var
        KF_O_NEXT    = KF_P_NEXT + observe_var
        KF_K_NEXT    = KF_P_NEXT / KF_O_NEXT
        KF_P_NEXT    = (1 - KF_K_NEXT) * (KF_P_NEXT)
        KF_Dist_NEXT = KF_Dist_cur + KF_K_NEXT * (y_cur - np.dot(F, u_cur) - KF_Dist_cur)

    #** step3: use CVX tools to calculate the nex input sequence u(i+1) **#

        u      = cvx.Variable(U_NUM)
        weight = 1.0
        linear_time_weight = np.identity(Y_NUM) * np.arange(Y_NUM) / 100.0
        obj    = cvx.norm((y_norm + F * u + KF_Dist_NEXT) * linear_time_weight, 2) + cvx.norm(weight * u, 2)
        prob   = cvx.Problem(cvx.Minimize(obj), [u <= 0.5, u >= -0.5])
        prob.solve()

        CL_NEXT = u.value + u_norm
        CL_NEXT = np.append(CL_NEXT, CL_NEXT[U_NUM - 1] * 0.3333)
        CL_NEXT = np.append(CL_NEXT, 0.0)
        
        CL_IN_NEXT = spline(degree_samp, CL_NEXT, degree)
        np.savetxt('CL_IN_NEXT.txt', CL_IN_NEXT.T, fmt='%.4f', delimiter='',newline=',', header='', footer='', comments='')

    #** step4: save all the global variables **#
        plt.figure(12)
        #plt.plot(degree_samp, CL_NEXT, marker='x')
        plt.plot(degree, CL_IN_NEXT, label=str(i))
        plt.plot(degree_samp, CL_NEXT, marker='o')
        plt.plot(degree, CL_0)

        KF_Dist = np.c_[KF_Dist, KF_Dist_NEXT]
        KF_P    = np.c_[KF_P, KF_P_NEXT]
        KF_O    = np.c_[KF_O, KF_O_NEXT]
        KF_K    = np.c_[KF_K, KF_K_NEXT]
        delt_u  = np.c_[delt_u, u.value]
        delt_y  = np.c_[delt_y, y_cur]

        np.savetxt('KF_P.txt',    KF_P,    fmt='%.3e')
        np.savetxt('KF_O.txt',    KF_O,    fmt='%.3e')
        np.savetxt('KF_K.txt',    KF_K,    fmt='%.3e')
        np.savetxt('KF_Dist.txt', KF_Dist, fmt='%.3e')
        np.savetxt('delt_y.txt',  delt_y,  fmt='%.3e')
        np.savetxt('delt_u.txt',  delt_u,  fmt='%.3e')
    
    plt.legend(loc='upper right')
    plt.show()

def ILC_sysident():
    colors = ['k','r','y','g','b','c','m','pink','lime','burlywood','azure','lightcyan','teal']

    data_choose = [['023', 120.6],['023', 120.6],['023', 120.6],
                   ['14', 136.0],['023', 173.66],['023', 205.64],['14', 178.95],['654', 126.65],['654', 86.65],['87', 256.30],['87', 212.30], #sys_ident: 12345678
                   ['0', 282.85]] #20190130ILC
    #data_choose = [[0, 201.5],[1, 118.3],[1, 118.3],[1, 118.3],[5, 92.0],[5, 92.0],[5, 92.0],[6, 132.0],[6, 132.0],[8, 239.0]] #20190130ILC
    #data_choose = [[0, 201.5],[1, 118.3],[2, 162.6],[3, 162.6],[5, 92.0],[5, 92.0],[6, 132.0],[6, 132.0],[8, 239.0]]

    degree  = [0,10,15,20,25,30,35,40,50,60,75,90]  #[0-,10-,15-,20,25,30,35,40,50,60,75,90-];
    delt_CL = -0.025
    F       = np.zeros(shape=(int(sample_time * 125), len(degree)))

    #plt.figure(2)
    plt.figure(figsize=(24, 12))

    for i in range(0,7):#len(degree)):
        #print(d1[1].name, d2[0].name,len(d1[1].data['timestamp']),len(d2[0].data['timestamp']))
        txt_num = str(data_choose[i][0])
        ulog1 = ULog(txt_num + '.ulg', 'vtol_vehicle_status')
        ulog2 = ULog(txt_num + '.ulg', 'vehicle_local_position')
        d1    = ulog1.data_list
        d2    = ulog2.data_list

        times_trans = d1[1].data['timestamp'] / 1000000.0 #second
        ticks_trans = d1[1].data['in_transition_to_fw']
        index       = np.where(times_trans >= data_choose[i][1])
        times_trans = times_trans[index]
        ticks_trans = ticks_trans[index]
        index       = np.where(ticks_trans >= 0.5)
        time_start  = (times_trans[index])[0]
        print(time_start)

        # get altitude error inside the trans time
        times_local      = d2[0].data['timestamp'] / 1000000.0 #second
        #alt              = d2[0].data['vz'] * (-1.0)
        alt              = d2[0].data['az'] * (-1.0)

        #filter
        if i == 3:
            plt.subplot(3,1,1)
            Y=fft_a(times_local,alt)
            plt.plot(Y[0,:], abs(Y[1,:]), color='r', linestyle='-', label = '0')
            plt.legend()

        lenth            = int(sample_time * 125)
        index            = np.where(times_local >= time_start)
        times_local_sub  = times_local[index]
        times_local_sub  = times_local_sub[np.arange(lenth)] - times_local_sub[0]
        alt_ref          = alt[index]
        alt_ref          = alt_ref[np.arange(lenth)] - alt_ref[0]

        if i == 0:
            np.savetxt("err_0.txt", Tr(alt_ref), fmt='%.3e');
            alt_err = alt_ref;
            legend_err = []
            legend_F   = []
            
            plt.subplot(3, 1, 2)
            plt.plot(times_local_sub, alt_ref, color=colors[i])
            print('No.', i, '  color: ',colors[i])
            legend_err.append(str(i))
            plt.legend(legend_err)
        else:
            #alt_err      = np.append(alt_err, alt_ref, axis=0)
            F[:,i-1]   = (alt_ref - alt_err) / delt_CL

            plt.subplot(3, 1, 2)
            plt.plot(times_local_sub, alt_ref, color=colors[i])
            print('No.', i, '  color: ',colors[i])
            legend_err.append(str(i))
            plt.legend(legend_err)

            plt.subplot(3, 1, 3)
            plt.plot(times_local_sub, F[:,i-1], color=colors[i])
            legend_F.append(str(i))
            plt.legend(legend_F)

    np.savetxt("F.txt", F, fmt='%.3e')

    fig_name = './'+'system identification altitude error.jpg' 
    plt.savefig(fig_name)
    plt.show()

'''
def rate_fft(ulog_file_name, output, time_start, time_end):
    """
    :param ulog_file_name: The ULog filename to open and read
    :param messages: A list of message names
    :param output: Output file path
    :param delimiter: CSV delimiter

    :return: None
    """
    messages   = 'vehicle_rates_setpoint,vehicle_attitude,actuator_controls_0'
    msg_filter = messages.split(',') if messages else None

    ulog = ULog(ulog_file_name, msg_filter)
    data = ulog.data_list

    output_file_prefix = ulog_file_name
    # strip '.ulg'
    if output_file_prefix.lower().endswith('.ulg'):
        output_file_prefix = output_file_prefix[:-4]

    # write to different output path?
    if output:
        base_name = os.path.basename(output_file_prefix)
        output_file_prefix = os.path.join(output, base_name)

    num_circle = 0;
    channel_name  = 'pitch'
    rpy_dic       = {'pitch':'[1]', 'roll':'[0]', 'yaw':'[2]'}
    rpy_index     = rpy_dic[channel_name]

    plt.figure(figsize=(24, 12))

    for d in data:
        fmt = '{0}_{1}_{2}.csv'
        output_file_name = fmt.format(output_file_prefix, d.name, d.multi_id)
        fmt = 'Writing {0} ({1} data points)'
        print(fmt.format(output_file_name, len(d.data['timestamp'])))
        #print(ulog.data_list[1])

        num_circle = num_circle + 1
        print(num_circle)
        data_name_dic = {'vehicle_rates_setpoint':channel_name, 'vehicle_attitude': channel_name + 'speed', 'actuator_controls_0': 'control' + rpy_index}
        plot_name_dic = {'vehicle_rates_setpoint':channel_name +'rate_cmd', 'vehicle_attitude': channel_name + 'rate_fdb','actuator_controls_0':channel_name+'rate_ctrloutput'}
        data_name     = data_name_dic[d.name]

        times = d.data['timestamp'] / 1000000.0 #second
        times = times - times[0] #start at  

        # get origin data
        pitchsp = d.data[data_name] * 57.3 #degree

        # get time subsection
        index = np.where((times>=time_start)&(times<=time_end))
        time_sub = times[index]
        print(times)
        pitchsp_sub = pitchsp[index]

        # for FFT analysis
        n = len(time_sub)
        samp_fre = (n-1) / (time_sub[n-1] - time_sub[0])
        k = np.arange(n)
        T = n / samp_fre
        frq = k / T # two sides frequency range
        frq = frq[range(n/2)] # one side frequency range

        Y = np.fft.fft(pitchsp_sub)/n # fft computing and normalization
        Y = Y[range(n/2)]

        plt.subplot(3,1,num_circle)
        plt.plot(frq, abs(Y), color='r', linestyle='-')
        plot_name = plot_name_dic[d.name]
        plt.title(plot_name)

    fig_name = './'+ channel_name +'rate_FFT' + '_' + ulog_file_name + '.jpg' 
    plt.savefig(fig_name)
    plt.show()
'''

def acc_filter(ulog_file_name):
    plt.figure(1, figsize=(9, 7))
    font2 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 14,
    }
    plt.subplot(2,1,1)
    with_notch = read_data(ulog_file_name,'sensor_accel','y',38, 88)
    Y=fft_a(with_notch[0,:], with_notch[1,:])
    plt.plot(Y[0,:], abs(Y[1,:]), color='r', linestyle='-', label = 'With Notch')
    #plt.ylim((0, 0.5))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylabel('With Notch and Low-Pass',font2)

    plt.subplot(2,1,2)
    b,a = signal.butter(4,2.0*5.0/250.0,'low')
    sf = signal.filtfilt(b,a,with_notch[1,:])
    Y=fft_a(with_notch[0,:], sf)
    plt.plot(Y[0,:], abs(Y[1,:]), color='r', linestyle='-', label = 'With Notch')
    #plt.ylim((0, 0.5))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylabel('With Notch and Low-Pass',font2)

    fig_name = 'notch_test.jpg'
    plt.savefig(fig_name)
    

    plt.figure(2, figsize=(9, 7))
    plt.subplot(2,1,1)
    plt.plot(with_notch[0,:], sf)
    #plt.xlim((150,160))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylabel('With Notch and Low-Pass',font2)

    plt.subplot(2,1,2)
    plt.plot(with_notch[0,:], with_notch[1,:])
    #plt.xlim((70,80))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Time(s)',font2)
    plt.ylabel('Without Notch and Low-Pass',font2)

    fig_name = 'notch_test_time.jpg'
    plt.savefig(fig_name)

    plt.show()

def ILC_cal_F_matrix():
    kp = -0.1
    kv = -0.2
    samp_t = 1.0/50.0 # input 50 point per second
    samp_N = 250 # overall number of ILC input
    A  = np.array([[0, 1],[kp, kv]])
    B  = np.array([[0],[1]])
    G  = np.array([1, 0])
    A_d = 1+A*samp_t
    B_d = B*samp_t
    #print(A_d)
    #print(B_d)
    F  = np.zeros(shape=(250, 250))
    for j in range(0,samp_N): #F(i,j)
        for i in range(0,samp_N):
            if i<=j:
                F[i,j] = 0.0
            else:
                F[i,j] = G.dot((np.power(A_d, i-j).dot(B_d)))