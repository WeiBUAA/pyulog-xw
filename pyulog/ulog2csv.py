#! /usr/bin/env python

"""
Convert a ULog file into CSV file(s)
"""

from __future__ import print_function

import argparse
import os

from .core import ULog

#pylint: disable=too-many-locals, invalid-name, consider-using-enumerate

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

    convert_ulog2csv(args.filename, args.messages, args.output, args.delimiter)


def convert_ulog2csv(ulog_file_name, messages, output, delimiter):
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
        index = np.where((times>=9.0)&(times<=100.0))
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
