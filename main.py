import numpy as np
import mysql.connector
from mysql.connector import Error
import time
import datetime
import logging
import csv
import random
import sys
from markov import MarkovChain
from samplegen import SampleGen
from sim import Simulation
from converter import Converter

# From the normal Windows cmd, I have manually installed the following packages using
# pip3 install <package> command:
# - numpy
# - mysql
# - matplotlib
# - statsmodels

# *********************************************************************
# script parameters
# *********************************************************************
day_window_device = {}
# stores one Markov model calculated for each (window, device) tuple
window_device = {}
# distinct states, i.e. c, d and r
number_of_states = 3
# length of a total window [s]
window_length = 6 * 3600
# duration [s] between state observations
time_between_samples = 2
# max duration [s] between subsequent states before inserting a disconnected state
max_duration_between_states = 10

# init time
now = datetime.datetime.today()
today = now.strftime("%Y%m%d_%H_%M_%S")

# files that are used to store the results of the script
logger_path = today + "_logfile.log"
results_path = today + "_results.csv"

# specificies the look of log messages
LOG_FORMAT = "%(levelname)s %(asctime)s - %(message)s"
# overwrite logfile on program execution
FILE_MODE = "w"

# params used to connect to a database
db_param_host = '<add host>'
db_param_database = '<add database>'
db_param_user = '<add user>'
db_param_password = '<add password>'

# *********************************************************************
# create logger
# *********************************************************************
logging.basicConfig(filename = logger_path, level = logging.DEBUG, format = LOG_FORMAT, filemode = FILE_MODE)
logger = logging.getLogger()

logger.info("Starting script execution...")

logger.info("")
logger.info("Script parameters:")
logger.info("window_length: " + str(window_length) + " s")
logger.info("time_between_samples: " + str(time_between_samples) + " s")
logger.info("max_duration_between_states: " + str(max_duration_between_states) + " s")
logger.info("")

start_script_execution = time.time()

# *********************************************************************
# load records from database
# *********************************************************************

converter = Converter(window_length, time_between_samples, max_duration_between_states)

logger.info("Loading records from database...")

start_loading_records = time.time()

# queries

query_state_sequences = ("select "
 "  window "
 ", device "
 ", date_of_request "
 ", window_start "
 ", window_end "
 ", state_sequence "
 "from v "
 "order by 1, 2, 3"
 )

try:

    connection = mysql.connector.connect(host=db_param_host,
                                         database=db_param_database,
                                         user=db_param_user,
                                         password=db_param_password)
    if connection.is_connected():
        cursor = connection.cursor()
        cursor.execute(query_state_sequences)
        for row in cursor.fetchall():
            # row[0] = window, row[1] = device
            if (row[0], row[1]) not in window_device:
                window_device[(row[0], row[1])] = 0
            date = row[2]
            # window-specific parameters
            start = str(row[3])
            end = str(row[4])
            # concatenated sequence of WLANs observed on the current day
            wlan_sequence = converter.convertToSequence(start, end, row[5])
            key = (date, row[0], row[1])
            day_window_device[key] = (start, end, wlan_sequence)
except Error as e:
    logger.info("Error while connecting to MySQL")
    quit()
finally:
    if (connection.is_connected()):
        cursor.close()
        connection.close()

end_loading_records = time.time()
loading_records_duration = str(end_loading_records - start_loading_records)

logger.info("Loading records from database finished: " + loading_records_duration +" s")

logger.info("Calculating the models...")
start_calculation = time.time()

# *********************************************************************
# calculate the models for all <window, device> pair
# *********************************************************************

total_number_of_models = len(window_device)
number_of_calculated_models = 0

for wd, item in window_device.items():  # wd -> (window, device)

    window = wd[0]
    device = wd[1]

    # *********************************************************************
    # calculate the initial WLAN
    # *********************************************************************

    a = time.time()

    # stores how often each WLAN occurs at the beginning of a day
    number_of_occurences_as_initial_wlan = np.zeros(346)
    # stores the initial WLAN
    initial_cluster = 0
    # determine the initial WLAN for the current window-device combination 
    for dwd, data in day_window_device.items():
        # skip if either window or device do not match
        if dwd[1] != window or dwd[2] != device:
            continue
        # determines the sequence of WLANs as ordered array
        wlan_sequence = data[2]
        wlans = wlan_sequence.split(',')
        # determines the initial WLAN given the sequence of the current day
        initial_cluster = 'd'
        for x in range(len(wlans)):
            if wlans[x] == 'd':
                continue
            initial_cluster = wlans[x]
            break
        # update frequencies
        if initial_cluster == 'd':
            number_of_occurences_as_initial_wlan[0] += 1
        else:
            number_of_occurences_as_initial_wlan[int(initial_cluster[1:])] += 1
    # computes the initial WLAN of the window-device combination
    max_value = np.amax(number_of_occurences_as_initial_wlan)
    initial_cluster = 0
    for i, value in enumerate(number_of_occurences_as_initial_wlan, start=0):
        if value == max_value:
            initial_cluster = i
            break

    logger.info("init number_of_occurences_as_initial_wlan: "+ str((time.time() - a)) + " s")

    # used to collect Markov models computed on a daily basis for the above <window, device> combination
    models_per_day = []

    # keys to remove after computation to reduce subsequent computations complexity
    keys_to_remove = []

    for dwd, data in day_window_device.items():  # dwd -> (day, window, device)
        # skip if either window or device do not match
        if dwd[1] != window or dwd[2] != device:
            continue

        # add (day, window, device) tuple to be able to remove them after the loop
        keys_to_remove.append(dwd)
        
        # *********************************************************************
        # replace occurences of states c<start_index> by c and c<!start_index> by r in state_sequence
        # *********************************************************************
        a = time.time()

        # determines the sequence of WLANs as ordered array
        wlan_sequence = data[2]
        wlans = wlan_sequence.split(',')
        
        # convert sequence of WLANs into a sequence of states
        sequence = ''
        for r, s in enumerate(wlans):
            if s.startswith('c'):
                if int(s[1:]) == initial_cluster:
                    wlans[r] = 'c'
                else:
                    wlans[r] = 'r'
            sequence += wlans[r] +','
        if sequence.endswith(','):
            sequence = sequence[0:len(sequence) - 1]

        # determine frequency for d,c and r from the sequence
        mapping = {"d": 0, "c": 1, "r": 2}
        number_of_occurrences = np.zeros(number_of_states)
        for state in wlans:
            index = mapping[state]
            number_of_occurrences[index] += 1

        # determines transition matrix of the day
        matrix = np.zeros((number_of_states, number_of_states))
        # determines the states occurring in the current state sequence
        distinct_states = np.unique(wlans)
        # used to count state transitions
        transitions = sequence.replace(',', '')
        # iterate through all combinations of state transitions s -> t
        for s in distinct_states:
            for t in distinct_states:
                # count number of occurrences of substring s + t
                number_of_transitions_s_to_t = 0
                for pos in range(len(transitions)):
                    if transitions[pos:].startswith(s + t):
                        number_of_transitions_s_to_t += 1
                # perform ratio calculation
                index_s = mapping[s]
                index_t = mapping[t]
                if number_of_occurrences[index_s] > 0:
                    matrix[index_s][index_t] = number_of_transitions_s_to_t / number_of_occurrences[index_s]                     

        a = time.time()

        # *********************************************************************
        # tackle special case for last state in the sequence using maximum likelihood
        # *********************************************************************

        # make matrix right stochastic by applying maximum likelihood
        end_state = wlans[len(wlans) - 1]
        row_of_end_state = 0
        if end_state == 'c':
            row_of_end_state = 1
        elif end_state == 'r':
            row_of_end_state = 2
        index = np.argmax(matrix[row_of_end_state])
        matrix[row_of_end_state][:] = 0
        matrix[row_of_end_state][index] = 1

        # *********************************************************************
        # save current transition matrix
        # *********************************************************************
        model = MarkovChain(distinct_states, initial_cluster, matrix, number_of_occurrences, wlans, 0)
        models_per_day.append(model)

    # remove all entries from day_window_device dictionary that 
    # are equal to the current (window, device) pair. This is okay, 
    # because they are no longer required here. Thus, we save time 
    # for the computation of the subsequent (window, device) pairs, since 
    # the inner loop has less iterations. Also, we save memory.
    # print("keys to remove: "+ str(len(keys_to_remove)))
    # print("len of day_window_device: "+ str(len(day_window_device)))
    for key_to_remove in keys_to_remove:
        del day_window_device[key_to_remove]
    # print("len of day_window_device: "+ str(len(day_window_device)))

    a = time.time()

    # *********************************************************************
    # computation of the weighted transition matrix
    # *********************************************************************
    weighted_matrix = np.zeros((number_of_states, number_of_states))
    for r in range(number_of_states):
        # occurrences of state r from all rows corresponding to r throughout all matrices
        number_of_r_over_all_matrices = 0
        for m in models_per_day:
            number_of_r_over_all_matrices += m.number_of_occurrences[r]
        for m in models_per_day:
            # sum of all occurrences of state r in the current matrix
            number_of_r_matrix = m.number_of_occurrences[r]
            # weight computation for the row of the current matrix
            weight_row = 0
            if (number_of_r_over_all_matrices != 0):
                weight_row = number_of_r_matrix / number_of_r_over_all_matrices
            # multiply entire row using the weighting factor
            m.transition_matrix[r][:] *= weight_row
    
    # determine weighted transition matrix by summing all weighted matrices
    for m in models_per_day:
        weighted_matrix += m.transition_matrix

    # there are the following cases for a cluster being observed on distinct days:
    # 1. the cluster appears in all days
    # 2. the cluster appears in some but not all days
    # 3. the cluster does not appear in any day
    # Cases 1. and 2. are covered through the weighted average calculation above. 
    # But there is one problem. Case 3. would yield a row having a sum of zero in the 
    # weighted transition matrix. Since any row must sum up to one, this is invalid. 
    # To address this problem, the corresponding cluster is set to an "island". 
    # This is achieved by assigning one to the matrix scalar at (cluster index, cluster index). 
    # Hence, the row of the cluster ends up having a sum of one. Since the cluster cannot 
    # be reached, it is an "island". The following code marks the clusters:
    for i in range(number_of_states):
        row_sum = np.sum(weighted_matrix[i])
        if row_sum == 0:
            weighted_matrix[i][i] = 1

    # *********************************************************************
    # add resulting markov model to the <window, device> pair it belongs
    # *********************************************************************
    initial_distribution = np.zeros(number_of_states)
    # set probability of state c to 1. Each model should 
    # start in the connected state. This state cannot be d or r.
    initial_distribution[1] = 1
    window_device[wd] = MarkovChain(np.zeros(1), initial_cluster, weighted_matrix, np.zeros(1), ['d'], initial_distribution) 

    # number_of_calculated_models += 1
    # print("calculated: "+ str(number_of_calculated_models) + "/" + str(total_number_of_models))

end_calculation = time.time()
calculation_duration = str(end_calculation - start_calculation)
logger.info("Calculating the models finished: " + calculation_duration +" s")

logger.info("Evaluating the models...")
start_evaluation = time.time()

# *********************************************************************
# evaluate models
# *********************************************************************

# number of samples used to calculate the average measures
number_of_samples = [25, 50, 100]
# mode of the prediction, i.e., IDV (False) or DPM (True)
is_dpm = True
# used to make a sample reproducible
seeds = [10, 75, 200, 67, 124]
# number of states a sample contains
sample_length_values = [25, 50, 100, 200]
# used to collect the simulation results
results = []

if (is_dpm == False):
    is_dpm_text = "IDV"
else:
    is_dpm_text = "DPM"

logger.info("")
logger.info("Simulation parameters:")
logger.info("is_dpm: " + is_dpm_text)
logger.info("")

results = []

number_of_calculated_models = 0
total_number_of_models = total_number_of_models * 60

for n in number_of_samples:
    number_of_samples_text = str(n)
    for seed in seeds:
        seed_text = str(seed)
        for length in sample_length_values:
            sample_length_text = str(length)
            rand = random.Random()
            for wd, mc in window_device.items():
                w = wd[0]
                d = wd[1]
                sample_gen = SampleGen(mc, rand)
                simulator = Simulation(sample_gen)
                # time used to generate all samples
                rand.seed(seed)
                sample_generation_time = 0
                for i in range(n):
                    start = time.time()
                    sample_gen.sample(length)
                    end = time.time()
                    sample_generation_time += end - start
                rand.seed(seed)
                # evaluate the MC given the current combination of parameters
                start = time.time()
                # performance indicator 1 - prediction accuracy
                accuracy = simulator.runSimulation(mc, n, length, is_dpm)
                end = time.time()
                # performance indicator 2 - execution time
                ex_time = (end - start - sample_generation_time) / (n * 1.0)
                if ex_time < 0:
                    ex_time = 0
                result = [w, d, str(accuracy).replace('.', ','), str(ex_time).replace('.', ','), mc.TransitionMatrixToText(), mc.InitialStateDistributionToText(), mc.startState(), number_of_samples_text, sample_length_text, is_dpm_text, seed_text]
                results.append(result)
                number_of_calculated_models += 1
                print("simulated: "+ str(number_of_calculated_models) + "/" + str(total_number_of_models))

end_evaluation = time.time()
evaluation_duration = str(end_evaluation - start_evaluation)
logger.info("Simulating the models finished: " + evaluation_duration +" s")

logger.info("Storing simulation results...")
start_store = time.time()

# sorting all devices by window, device, n and length
results.sort(key = lambda x: (x[0], x[1], int(x[7]), int(x[8])))

csv.register_dialect('my_dialect',
delimiter = ';',
quoting = csv.QUOTE_NONE)

with open(results_path, FILE_MODE, newline='\n', encoding='utf-8') as f:
    writer = csv.writer(f, 'my_dialect')
    writer.writerow(['Window', 'Device', 'Accuracy', 'Execution time', 'Transition matrix', 'Initial state distribution', 'start state', 'Number of samples', 'Sample length', 'Is DPM', 'Seed'])
    for row in results:
        writer.writerow(row)
f.close()

end_store = time.time()
store_duration = str(end_store - start_store)
logger.info("Storing simulation results finished: " + store_duration +" s")

end_script_execution = time.time()
script_execution_duration = str(end_script_execution - start_script_execution)

logger.info("Script execution finished: "+ script_execution_duration + " s")

print("Script execution finished")
