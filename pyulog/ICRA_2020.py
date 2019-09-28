#! /usr/bin/env python

"""
Convert a ULog file into CSV file(s)
"""

from __future__ import print_function

import argparse
import re
import os

from .core import ULog
from .px4  import PX4ULog
import matplotlib.pyplot as plt
from   matplotlib import colors as mcolors
plt.rcParams['text.usetex'] = True
import numpy as np
font_size = 24

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

    args = parser.parse_args()

    #plot_hover_efficiency()
    plot_hover_accuracy()
    #plot_gap()

def plot_hover_efficiency():
    ulog_name = 'hover.ulg'
    time = [300, 500]
    #time = [275, 400]

    plt.figure(1, figsize=(9, 6))
    font2 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 15,
    }

    PX4 = PX4ULog()
    current = PX4.read_data(ulog_name, 'battery_status', 0, 'current_a', time[0], time[1])
    voltage = PX4.read_data(ulog_name, 'battery_status', 0, 'voltage_v', time[0], time[1])
    #pos_x   = PX4.read_data(ulog_name, 'vehicle_local_position', 0, 'x', time[0], time[1])
    #pos_y   = PX4.read_data(ulog_name, 'vehicle_local_position', 0, 'y', time[0], time[1])
    #pos_z   = PX4.read_data(ulog_name, 'vehicle_local_position', 0, 'z', time[0], time[1])
    #att   = PX4.read_data(ulog_name, 'vehicle_attitude', 0, 'q', time[0], time[1])
    #attsp = PX4.read_data(ulog_name, 'vehicle_attitude_setpoint', 0, 'q_d', time[0], time[1])

    plt.subplot(3,1,1)
    plt.plot(voltage[0,:], voltage[1,:], color='r', linestyle='-', label = 'Current ('+'$A$'+')')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid(True)
    plt.ylim((10, 20))
    plt.ylabel('Voltage ($V$)',font2)

    plt.subplot(3,1,2)
    plt.plot(current[0,:], current[1,:], color='r', linestyle='-', label = 'Current ('+'$A$'+')')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=14)
    plt.grid(True)
    plt.ylim((17, 27))
    plt.ylabel('Current ($A$)',font2)

    plt.subplot(3,1,3)
    plt.plot(current[0,:], voltage[1,:]*current[1,:], color='r', linestyle='-', label = 'Current ('+'$A$'+')')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid(True)
    plt.ylim((250, 400))
    plt.ylabel('Power ($W$)',font2)

    plt.xlabel('Time ($s$)',font2)

    fig_name = 'hover_efficient.pdf'
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.show()

def plotstyle(plt):
    font2 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 15,
    }
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(True)
    plt.legend(prop=font2)

def plot_hover_accuracy():
    ulog_name = 'perk.ulg'
    time = [2628.8, 2639.7]
    #time = [275, 400]

    plt.figure(1, figsize=(12, 6.8))
    
    font2 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 18,
    }

    PX4 = PX4ULog()
    #current = PX4.read_data(ulog_name, 'battery_status', 0, 'current_filtered_a', time[0], time[1])
    #voltage = PX4.read_data(ulog_name, 'battery_status', 0, 'voltage_v', time[0], time[1])

    pos_x   = PX4.read_data(ulog_name, 'vehicle_local_position', 0, 'x', time[0], time[1])
    pos_y   = PX4.read_data(ulog_name, 'vehicle_local_position', 0, 'y', time[0], time[1])
    pos_z   = PX4.read_data(ulog_name, 'vehicle_local_position', 0, 'z', time[0], time[1])
    pos_xd  = PX4.read_data(ulog_name, 'vehicle_visual_odometry', 0, 'x', time[0], time[1])
    pos_yd  = PX4.read_data(ulog_name, 'vehicle_visual_odometry', 0, 'y', time[0], time[1])
    pos_zd  = PX4.read_data(ulog_name, 'vehicle_visual_odometry', 0, 'z', time[0], time[1])
    att   = PX4.read_data(ulog_name, 'vehicle_attitude', 0, 'q', time[0], time[1])
    attsp = PX4.read_data(ulog_name, 'vehicle_attitude_setpoint', 0, 'q_d', time[0], time[1])
    att[0,:] = att[0,:]-2550
    attsp[0,:] = attsp[0,:]-2550
    #roll_filt = PX4.band_smoother(att[0,:], att[1,:], [6,20])

    plt.subplot(3,2,1)
    plt.plot(pos_x[0,:]-2550, pos_x[1,:]-0.72, color='r', linestyle='-', label = 'Actual')
    plt.plot(pos_xd[0,:]-2550, pos_xd[1,:]*0-0.15, color='b', linestyle='-', label = 'Desired')
    plt.ylim((-1.5, 2))
    plotstyle(plt)
    plt.ylabel('Position X ($m$)',font2)
    plt.subplot(3,2,3)
    plt.plot(pos_y[0,:]-2550, pos_y[1,:], color='r', linestyle='-', label = 'Actual')
    plt.plot(pos_yd[0,:]-2550, pos_yd[1,:]*0+1, color='b', linestyle='-', label = 'Desired')
    plt.ylim((0, 3))
    plotstyle(plt)
    plt.ylabel('Position Y ($m$)',font2)
    plt.subplot(3,2,5)
    plt.plot(pos_z[0,:]-2550, pos_z[1,:], color='r', linestyle='-', label = 'Actual')
    plt.plot(pos_zd[0,:]-2550, pos_zd[1,:], color='b', linestyle='-', label = 'Desired')
    plt.ylim((-2, 0))
    plotstyle(plt)
    plt.xlabel('Time ($s$)',font2)
    plt.ylabel('Position Z ($m$)',font2)

    plt.subplot(3,2,2)
    plt.plot(att[0,:], att[1,:], color='r', linestyle='-', label = 'Actual')
    plt.plot(attsp[0,:], attsp[1,:], color='b', linestyle='-', label = 'Desired')
    plt.ylim((-45, 65))
    plotstyle(plt)
    plt.ylabel('Roll ($^\circ$)',font2)
    plt.subplot(3,2,4)
    plt.plot(att[0,:], att[2,:], color='r', linestyle='-', label = 'Actual')
    plt.plot(attsp[0,:], attsp[2,:], color='b', linestyle='-', label = 'Desired')
    plt.ylim((-10, 30))
    plotstyle(plt)
    plt.ylabel('Pitch ($^\circ$)',font2)
    plt.subplot(3,2,6)
    plt.plot(att[0,:], att[3,:]-2, color='r', linestyle='-', label = 'Actual')
    plt.plot(attsp[0,:], attsp[3,:]-2, color='b', linestyle='-', label = 'Desired')
    plt.ylim((-70, 0))
    plotstyle(plt)
    plt.xlabel('Time ($s$)',font2)
    plt.ylabel('Yaw ($^\circ$)',font2)

    fig_name = 'perk.eps'
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.show()

def plot_gap():
    ulog_name = 'gap.ulg'
    time = [2121, 2128.5]
    offset = 2000
    #time = [275, 400]

    plt.figure(1, figsize=(12, 6.8))
    font2 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 18,
    }

    PX4 = PX4ULog()
    #current = PX4.read_data(ulog_name, 'battery_status', 0, 'current_a', time[0], time[1])
    #voltage = PX4.read_data(ulog_name, 'battery_status', 0, 'voltage_v', time[0], time[1])
    pos_x   = PX4.read_data(ulog_name, 'vehicle_local_position', 0, 'x', time[0], time[1])
    pos_y   = PX4.read_data(ulog_name, 'vehicle_local_position', 0, 'y', time[0]-0.2, time[1]-0.2)
    pos_z   = PX4.read_data(ulog_name, 'vehicle_local_position', 0, 'z', time[0], time[1])
    pos_xd  = PX4.read_data(ulog_name, 'vehicle_visual_odometry', 0, 'x', time[0], time[1])
    pos_yd  = PX4.read_data(ulog_name, 'vehicle_visual_odometry', 0, 'y', time[0], time[1])
    pos_zd  = PX4.read_data(ulog_name, 'vehicle_visual_odometry', 0, 'z', time[0], time[1])
    att   = PX4.read_data(ulog_name, 'vehicle_attitude', 0, 'q', time[0], time[1])
    attsp = PX4.read_data(ulog_name, 'vehicle_attitude_setpoint', 0, 'q_d', time[0], time[1])
    att[0,:] = att[0,:]-offset
    attsp[0,:] = attsp[0,:]-offset
    #roll_filt = PX4.band_smoother(att[0,:], att[1,:], [6,20])

    plt.subplot(3,2,3)
    plt.plot(pos_x[0,:]-offset, pos_x[1,:]-0.8, color='r', linestyle='-', label = 'Actual')
    plt.plot(pos_xd[0,:]-offset, (pos_xd[1,:]-0.8)*0.3, color='b', linestyle='-', label = 'Desired')
    plt.ylim((-0.5, 1.5))
    plotstyle(plt)
    plt.ylabel('Position Y ($m$)',font2)
    plt.subplot(3,2,1)
    plt.plot(pos_y[0,:]-offset+0.2, pos_y[1,:]-2, color='r', linestyle='-', label = 'Actual')
    plt.plot(pos_yd[0,:]-offset, pos_yd[1,:]-2.05, color='b', linestyle='-', label = 'Desired')
    plt.ylim((-4.0, -1.5))
    plotstyle(plt)
    plt.ylabel('Position X ($m$)',font2)
    plt.subplot(3,2,5)
    plt.plot(pos_z[0,:]-offset, pos_z[1,:], color='r', linestyle='-', label = 'Actual')
    plt.plot(pos_zd[0,:]-offset, pos_zd[1,:]*0-1.15, color='b', linestyle='-', label = 'Desired')
    plt.ylim((-1.5, 0.5))
    plotstyle(plt)
    plt.xlabel('Time ($s$)',font2)
    plt.ylabel('Position Z ($m$)',font2)

    plt.subplot(3,2,2)
    plt.plot(att[0,:], att[1,:], color='r', linestyle='-', label = 'Actual')
    plt.plot(attsp[0,:], attsp[1,:], color='b', linestyle='-', label = 'Desired')
    plt.ylim((-20, 40))
    plotstyle(plt)
    plt.ylabel('Roll ($^\circ$)',font2)
    plt.subplot(3,2,4)
    plt.plot(att[0,:], att[2,:], color='r', linestyle='-', label = 'Actual')
    plt.plot(attsp[0,:], attsp[2,:], color='b', linestyle='-', label = 'Desired')
    plt.ylim((-20, 40))
    plotstyle(plt)
    plt.ylabel('Pitch ($^\circ$)',font2)
    plt.subplot(3,2,6)
    plt.plot(att[0,:], att[3,:]-2, color='r', linestyle='-', label = 'Actual')
    plt.plot(attsp[0,:], attsp[3,:]-2, color='b', linestyle='-', label = 'Desired')
    plt.ylim((-95, 0))
    plt.xlabel('Time ($s$)',font2)
    plt.ylabel('Yaw ($^\circ$)',font2)
    plotstyle(plt)

    fig_name = 'gap.pdf'
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.show()

