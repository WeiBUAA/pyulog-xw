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
    CDC_plot_acc_ctrl()
    #CDC_plot()
    F = ILC_cal_F_matrix()
    #ILC(F)

def get_cur_y(name_num, resamp_freq):
    ulogname = 'ILC-' + str(name_num) + '.ulg'
    ulog1 = ULog(ulogname, 'vtol_vehicle_status')
    ulog2 = ULog(ulogname, 'vehicle_local_position')
    d1    = ulog1.data_list
    d2    = ulog2.data_list
    #print(d1[1].name, d2[0].name,len(d1[1].data['timestamp']),len(d2[0].data['timestamp']))

    times_trans = d1[0].data['timestamp'] / 1000000.0 #second
    ticks_trans = d1[0].data['in_transition_to_fw']
    
    index       = np.where(ticks_trans >= 0.5)
    
    time_start  = (times_trans[index])[0]
    #print(index)
    
    # get altitude error inside the trans time
    times_local      = d2[0].data['timestamp'] / 1000000.0 #second
    alt              = d2[0].data['z'] * (-1.0)
    lenth            = int(sample_time * 125)
    index            = np.where(times_local >= time_start)
    times_local_sub  = times_local[index]
    times_local_sub  = times_local_sub[np.arange(lenth)] - times_local_sub[0]
    alt_ref          = alt[index]
    alt_ref          = alt_ref[np.arange(lenth)] - alt_ref[0]

    index_25hz = np.arange(0, sample_time, 1/resamp_freq)
    #print(index_25hz.shape)
    err_data = np.interp(index_25hz, times_local_sub, alt_ref)
    #err_data = alt_ref[index_25hz]

    return err_data


def ILC(F):
    #** set the number of iterations **#
    colors = ['k','r-.','y--','g:','b','c','m','pink','lime','burlywood']
    font2 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 14,
    }

    Y_NUM = F.shape[0]
    U_NUM = Y_NUM
    resamp_freq = 25.0
    time  = np.arange(0, sample_time, sample_time / Y_NUM)
    
    #** step1: read the global variables from dir **#
    plt.figure(1, figsize=(10, 6))
    for i in range(0, 6):
        Num_iteration = i
        print(i)
        w = [-1,2.8,-1.0,0.3,0.08,0.3]
        u_norm   = np.zeros(Y_NUM)
        y_norm   = get_cur_y(0, resamp_freq)* w[0];
        
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
            normal    = np.zeros(6)

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
            y_cur       = delt_y * w[i]

            plt.subplot(2,1,1)
            plt.plot(time, y_norm, color = colors[i], label='0', linewidth=2)
            plt.legend(loc='lower left')

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
            y_cur       = get_cur_y(Num_iteration, resamp_freq) * w[i] - y_norm

            plt.subplot(2,1,1)
            plt.plot(time, y_cur + y_norm, colors[i], label=str(i), linewidth=2)
            #plt.ylim([-0.5,2.0])
            plt.ylabel('Altitude Error '+r'$(m)$',font2)
            plt.legend(loc='lower left')

        KF_P_NEXT    = KF_P_cur + process_var
        KF_O_NEXT    = KF_P_NEXT + observe_var
        KF_K_NEXT    = KF_P_NEXT / KF_O_NEXT
        KF_P_NEXT    = (1 - KF_K_NEXT) * (KF_P_NEXT)
        KF_Dist_NEXT = KF_Dist_cur + KF_K_NEXT * (y_cur - np.dot(F, u_cur) - KF_Dist_cur)

    #** step3: use CVX tools to calculate the nex input sequence u(i+1) **#

        u      = cvx.Variable(U_NUM)
        weight = 9
        #linear_time_weight = np.identity(Y_NUM) * np.arange(Y_NUM) / 100.0
        obj    = cvx.norm((y_norm + F * u + KF_Dist_NEXT), 2) + weight * cvx.norm(u, 2)
        prob   = cvx.Problem(cvx.Minimize(obj), [u <= 0.1, u >= -0.1])
        prob.solve()
        U_NEXT = u.value + u_norm
        normal[i]=np.linalg.norm(y_cur + y_norm,2)
        np.set_printoptions(precision=6)
        print('max:',np.max(y_cur + y_norm),'min:',np.min(y_cur + y_norm),'mean square',np.sqrt(normal[i]*normal[i]/25.0/6.0))
        np.savetxt('U_IN_NEXT.txt', U_NEXT.T, fmt='%.5f', delimiter='',newline=',', header='', footer='', comments='')

    #** step4: save all the global variables **#
        plt.subplot(2,1,2)
        #plt.plot(degree_samp, CL_NEXT, marker='x')
        if(i<5):
            plt.plot(time, (U_NEXT)/0.5*9.87,colors[i+1], label=str(i+1), linewidth=2)
        plt.ylabel('ILC Input ' + r'$(m/s^{2})$',font2)
        plt.xlabel('Time (s)',font2)
        #plt.ylim((-2,1.5))
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(loc='lower left')

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
    '''
    plt.subplot(3,1,3)
    plt.plot(time, F[:,10])
    plt.plot(time, F[:,50])
    plt.plot(time, F[:,90])
    plt.title('F_Matrix')
    '''
    ref = np.zeros(y_cur.shape[0])
    plt.subplot(2,1,1)
    plt.plot(time, ref, linestyle = '--', alpha = 0.6, lw = 4, label = 'Ref', color='gray')
    plt.ylim((-5.5,1))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(loc='lower left')
    plt.savefig('ILC-test.jpg',dpi=900)

    plt.figure(2, figsize=(8, 4))
    plt.plot(range(0, 6), normal, 'ko-')
    plt.xlabel('Iteration', font2)
    plt.ylabel('2-Norm State Error ' + r'$ (m) $', font2)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig('NORM.jpg',dpi=900)
    plt.show()

def ILC_cal_F_matrix():
    kp = -0.5
    kv = -1.5
    samp_t = 1.0/25.0 # input 50 point per second
    all_T  = sample_time
    all_N  = int(all_T / samp_t) # overall number of ILC input
    A  = np.array([[0, 1],[kp, kv]])
    B  = np.array([[0],[1]])/0.5*9.87
    G  = np.array([1, 0])

    A_d = np.eye(2) + A*samp_t
    B_d = B*samp_t
    print(A_d,np.power(A_d, 50))
    print(B_d)
    F  = np.zeros(shape=(all_N, all_N))
    for j in range(0,all_N): #F(i,j)
        for i in range(0,all_N):
            if i<=j:
                F[i,j] = 0.0
            elif i == j+1:
                C      = B_d
                F[i,j] = G.dot(C)
            elif i > j+2:
                C = A_d.dot(C)
                F[i,j] = G.dot(C)

    np.savetxt("F.txt", F, fmt='%.4e')
    print(all_N)
    return F
def CDC_plot_acc_ctrl():
    resamp_freq = 250.0;
    time  = np.arange(0, sample_time, 1.0/resamp_freq)

    plt.figure(1, figsize=(10, 7))
    font2 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 14,
    }

    ulog_file_name = 'ACC_TEST.ulg'

    plt.subplot(2,1,1)
    start=33.1
    atti_data = read_data(ulog_file_name,'vehicle_attitude_setpoint','q_d', start, start+6.0)
    plt.plot(atti_data[0,:]-atti_data[0,0], (atti_data[2,:])*1.15+5, 'k-', label = r'$\theta_{sp}$', lw = 2)
    start=33.1
    atti_data = read_data(ulog_file_name,'vehicle_attitude', 'q', start, start+6.0)
    plt.plot(atti_data[0,:]-atti_data[0,0], (atti_data[2,:])*1.15+5, 'r--', label = r'$\theta$', linewidth=2)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylabel('Pitch '+r'$(^{\circ})$', font2)
    plt.legend(prop=font2,loc= 'upper right')
    plt.ylim((-95,15))

    plt.subplot(2,1,2)
    start=22.3

    y_data = read_data(ulog_file_name,'vehicle_local_position', 'x', start, start+6.0)
    plt.plot(y_data[0,:]-y_data[0,0], np.zeros(750), 'k-', label = 'Ref', lw = 2)
    plt.plot(y_data[0,:]-y_data[0,0], y_data[1,:]-y_data[1,0], 'r--', label = r'$\xi_{sm}$', linewidth=2)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Time '+r'$(s)$', font2)
    plt.ylabel('Side-Margin '+r'$(m)$', font2)
    plt.legend(prop=font2,loc= 'upper right')
    plt.ylim((-2,2))
    plt.savefig('ATTI_TEST.jpg', dpi=900)
    
    plt.figure(2, figsize=(10, 5))
    #atti_data = read_data(ulog_file_name,'vehicle_attitude_setpoint','q_d', start, start+6.0)
    #plt.plot(atti_data[0,:]-atti_data[0,0], (atti_data[2,:])*1.15+5, label = r'$\theta_{sp}$', linestyle = '--', alpha = 0.6, lw = 4)
    start=33.0
    atti_data = read_data(ulog_file_name,'vehicle_attitude', 'q', start, start+6.0)
    atti = np.interp(time, atti_data[0,:]-atti_data[0,0], atti_data[2,:])

    acc_z_data = read_data(ulog_file_name,'sensor_accel', 'z', start, start+6.0)
    acc_z = np.interp(time, acc_z_data[0,:]-acc_z_data[0,0], acc_z_data[1,:])

    acc_x_data = read_data(ulog_file_name,'sensor_accel', 'x', start, start+6.0)
    acc_x = np.interp(time, acc_x_data[0,:]-acc_x_data[0,0], acc_x_data[1,:])

    acc_cmd = (9.88 + acc_z*np.sin(-1.0*atti/57.3))/np.cos(-1.0*atti/57.3)

    b,a = signal.butter(1,2.0*8/100.0,'low')
    sf = signal.filtfilt(b,a,acc_cmd)

    #plt.subplot(2,1,1)
    #plt.plot(time, atti, label = r'$\theta$', linewidth=2)
    #plt.ylabel('Pitch '+r'$(^{\circ})$', font2)
    #plt.subplot(2,1,2)
    plt.plot(time, sf*0.7+3.1, 'k-', label = 'Command', linewidth=2)
    plt.plot(time, acc_x*0.7+3.5, 'r:', label = 'Feedback', linewidth=2)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim((-0,25))
    plt.xlabel('Time '+r'$(s)$', font2)
    plt.ylabel('Body X Acceleration '+r'$(m/s^{2})$', font2)
    plt.legend(prop=font2,loc= 'upper left')
    plt.savefig('ACC_TEST.jpg', dpi=900)
    plt.show()


def CDC_plot():
    resamp_freq = 150.0;
    time  = np.arange(0, sample_time, 1.0/resamp_freq)

    plt.figure(1, figsize=(10, 6))
    font2 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 14,
    }

    ulog_file_name = 'att_test.ulg'
    plt.subplot(2,1,1)
    start=33.1
    atti_data = read_data(ulog_file_name,'vehicle_attitude_setpoint','q_d', start, start+6.0)
    plt.plot(atti_data[0,:]-atti_data[0,0], (atti_data[2,:])*1.15+5, 'k-', label = r'$\theta_{sp}$', lw = 2)
    start=33.1
    atti_data = read_data(ulog_file_name,'vehicle_attitude', 'q', start, start+6.0)
    plt.plot(atti_data[0,:]-atti_data[0,0], (atti_data[2,:])*1.15+5, 'r--', label = r'$\theta$', linewidth=2)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylabel('Pitch '+r'$(^{\circ})$', font2)
    plt.legend(prop=font2,loc= 'upper right')
    plt.ylim((-95,15))

    plt.subplot(2,1,2)
    start=22.3

    y_data = read_data(ulog_file_name,'vehicle_local_position', 'x', start, start+6.0)
    plt.plot(y_data[0,:]-y_data[0,0], np.zeros(750), 'k-', label = 'Ref', lw = 2)
    plt.plot(y_data[0,:]-y_data[0,0], y_data[1,:]-y_data[1,0], 'r--', label = r'$\xi_{sm}$', linewidth=2)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylabel('Side-Margin '+r'$(m)$', font2)
    plt.xlabel('Time '+r'$(s)$', font2)
    plt.legend(prop=font2,loc= 'upper right')
    plt.ylim((-2,2))
    plt.savefig('ATTI_TEST.jpg', dpi=900)
    plt.show()