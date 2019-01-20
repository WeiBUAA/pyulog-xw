#! /usr/bin/env python

"""
Convert a ULog file into CSV file(s)
"""

from __future__ import print_function

import argparse
import re
import os
import matplotlib.pyplot as plt
import numpy as np

from .core import ULog


#pylint: disable=too-many-locals, invalid-name, consider-using-enumerate
def Tr(a):
    return a.reshape(1,a.shape[0])

def Loadtxt(a):
    b = np.loadtxt('CL_0.txt')
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
    #messages_list = 'vehicle_rates_setpoint,vehicle_attitude,actuator_controls_0'
    #rate_fft(args.filename, messages_list, args.output, args.delimiter)

    
    #system identification of ILC
    #ILC_sysident()

    ILC()

def ILC():
#** set the number of iterations **#
    Num_iteration = 0

#** step1: read the global variables from docs **#
    CL_0  = Loadtxt('CL_0.txt')
    F     = Loadtxt('F.txt')
    err_0 = Loadtxt('err_0.txt')

    print(err_0.shape, CL_0.shape)
#** step2: get the current input sequence u(i) and altitude error (output) sequece err[i] **#

#** step3: use Kalman Filter to estimate the disturbance **#
    process_var = 0.05;
    observe_var = 0.1;
    if (os.path.isfile("KF_P.txt") == 0):
        KF_P      = err_0 * err_0;
        KF_O      = np.zeros(shape=(1, 250))
        KF_K      = np.zeros(shape=(1, 250))
        KF_Dist   = err_0
        np.savetxt('KF_P.txt', KF_P)
        np.savetxt('KF_O.txt', KF_O)
        np.savetxt('KF_K.txt', KF_K)
        np.savetxt('KF_Dist.txt', KF_Dist)
    #print(Disturb_est)
    else:
        KF_P      = Loadtxt('KF_P.txt')
        KF_O      = Loadtxt('KF_O.txt')
        KF_K      = Loadtxt('KF_K.txt')
        KF_Dist   = Loadtxt('KF_Dist.txt')

        Num_iter  = KF_P.shape[0]

        if(KF_P.shape[0] <= Num_iteration + 1):

            KF_P_NEXT = KF_P[Num_iter - 1, :] + process_var
            KF_O_NEXT = KF_P_NEXT + observe_var
            KF_K_NEXT = KF_P_NEXT / KF_O_NEXT
            KF_P_NEXT = (1 - KF_K_NEXT) * (KF_P_NEXT)

            KF_P.append(KF_P_NEXT, axis=0)
            KF_O.append(KF_O_NEXT, axis=0)
            KF_K.append(KF_K_NEXT, axis=0)

            KF_Dist_NEXT = KF_Dist[Num_iter - 1, :] + KF_K_NEXT * (err[Num_iteration, :] - dot(F, CL_IN[Num_iteration - 1], :) - KF_Dist[Num_iter - 1, :])

            KF_Dist.append(KF_Dist_NEXT, axis=0)




#** step4: calculate the nex input sequence u(i+1) **#

#** step5: save all the global variables **#
    #Disturb_est = np.append(Disturb_est, np.zeros((1, 250)), axis = 0)


def ILC_sysident():
    degree  = [0,10,15,20,30,40,50,60,80,90]
    delt_CL = [-0.05,-0.008993,-0.0113775497483709,-0.011808,-0.01064,-0.010411,-0.010024,-0.0087125,-0.05]
    F       = np.zeros(shape=(250, 9))

    plt.figure(figsize=(24, 12))

    for i in range(0,10):
        txt_num = str(i)
        ulog1 = ULog(txt_num + '.ulg', 'vtol_vehicle_status')
        ulog2 = ULog(txt_num + '.ulg', 'vehicle_local_position')
        d1    = ulog1.data_list
        d2    = ulog2.data_list
        #print(d1[1].name, d2[0].name,len(d1[1].data['timestamp']),len(d2[0].data['timestamp']))

        times_trans = d1[1].data['timestamp'] / 1000000.0 #second
        ticks_trans = d1[1].data['ticks_since_trans']
        index       = np.where(ticks_trans >= 1)
        time_start  = (times_trans[index])[0]
        
        # get altitude error inside the trans time
        times_local      = d2[0].data['timestamp'] / 1000000.0 #second
        alt              = d2[0].data['z'] * (-1.0)
        lenth            = 250
        index            = np.where(times_local >= time_start)
        times_local_sub  = times_local[index]
        times_local_sub  = times_local_sub[np.arange(lenth)] - times_local_sub[0]
        alt_ref          = alt[index]
        alt_ref          = alt_ref[np.arange(lenth)] - alt_ref[0]

        if i == 0:
            np.savetxt("err_0.txt", Tr(alt_ref));
            alt_err = alt_ref;
        else:
            #alt_err      = np.append(alt_err, alt_ref, axis=0)
            F[:,i-1]     = (alt_ref - alt_err) / delt_CL[i-1]

            plt.subplot(2, 1, 1)
            plt.plot(times_local_sub, alt_ref, color='r', linestyle='-')

            plt.subplot(2, 1, 2)
            plt.plot(times_local_sub, F[:,i-1], color='b', linestyle='-')

    np.savetxt("F.txt", F)

    fig_name = './'+'system identification altitude error.jpg' 
    plt.savefig(fig_name)
    plt.show()



def rate_fft(ulog_file_name, messages, output, delimiter):
    """
    Coverts and ULog file to a CSV file.

    :param ulog_file_name: The ULog filename to open and read
    :param messages: A list of message names
    :param output: Output file path
    :param delimiter: CSV delimiter

    :return: None
    """

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
    channel_name  = 'roll'
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
        times = times - times[0] #start at 0s

        # get origin data
        pitchsp = d.data[data_name] * 57.3 #degree

        # get time subsection
        #index = np.where((times<38.8)&(times>=15.1))
        #index = np.where((times<80.7)&(times>=40.0))
        #index = np.where((times<162.0)&(times>=74.0)) #20190107outdoor-waypoint
        index = np.where((times<97.5)&(times>=57.5))
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