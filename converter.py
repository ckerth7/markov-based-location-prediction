
import time
import datetime
import sys

class Converter:
    """ 
    Used to convert an abstract state sequence of the form 
    "state1;login_datetime1;logout_datetime1,state2;login_datetime2;logout_datetime2 ..." 
    into "state1, state2 ...". This conversion is achieved by two steps. First, disconnected states 
    are being inserted into the input state sequence as needed. Second, the altered state sequence 
    is being sampled. The sampling is necessary to ensure that the state observations occur periodically. 
    Otherwise, the time aspect would not be considered, since some states could occur 
    way less than others but might last longer. The sampled sequence of states is the result 
    of the conversion.
    """

    def __init__(self, window_length, time_between_samples, max_duration_between_states):
        """ 
        Initializes an instance of this class using the given parameters.
        The parameter max_duration_between_states is a threshold used to decide whether to insert 
        a disconnected state between two subsequent states. If the duration between two subsequent states exceeds 
        max_duration_between_states, a disconnected state is being inserted.
      
        Parameters: 
            window_length (int): Window length in seconds
            time_between_samples (int): Number of seconds between observations
            max_duration_between_states (int): Maximum duration between two subsequent states in seconds
        """
        self.max_duration_between_states = max_duration_between_states
        self.window_length = window_length
        self.time_between_samples = time_between_samples
        
    def convertToSequence(self, window_start, window_end, state_sequence):
        """ 
        Converts a state sequence of the form 
        "state1;login_datetime1;logout_datetime1,state2;login_datetime2;logout_datetime2 ..." 
        into a state sequence of the form "state1, state2 ...".
      
        Parameters: 
            window_start (time): Starting time of the window
            window_end (time): Ending time of the window
            state_sequence (str): State sequence of the form "state1;login_time1;logout_time1, state2;login_time2;logout_time2, ..."

        Returns:
        str: State sequence of the form "state1, state2, ..."
        """
        state_sequence = self.addDisconnectedStates(window_start, window_end, state_sequence)
        state_sequence = self.connectAdjacentStates(state_sequence)
        state_intervals = self.getStateIntervalsFromStateSequence(state_sequence)
        state_intervals = self.fillGapsBetweenIntervals(state_intervals)
        state_sequence = self.getStateSequenceBySamplingStateInveral(state_intervals)
        return state_sequence

    def getStateIntervalsFromStateSequence(self, state_sequence):
        """ 
        Creates a list where each entry consists of an interval and an associated state.
        This list is created by walking through the given state sequence and creating tuples 
        of the form (lower_bound_in_seconds, upper_bound_in_seconds, state). These tuples are added 
        to the resulting list. The list can be used to determine the state for a given point of time 
        in the sequence.
      
        Parameters: 
            state_sequence (str): State sequence of the form "state1;login_time1;logout_time1, state2;login_time2;logout_time2, ..."

        Returns:
        list: state_intervals (list<lower_bound_s (int), upper_bound_s (int), state (str)>): list of (interval, state) pairs ordered by time in ascending order
        """
        states = state_sequence.split(',')
        # keys are intervals, i.e., [start time - end_time[ of time, 
        # values are the states associated to the intervals
        state_intervals = []
        for state in states:
            details = state.split(';')
            cluster = details[0]
            start = datetime.datetime.strptime(details[1], '%H:%M:%S').time()
            end = datetime.datetime.strptime(details[2], '%H:%M:%S').time()
            lower_bound_in_seconds = start.hour * 3600 + start.minute * 60 + start.second
            upper_bound_in_seconds = end.hour * 3600 + end.minute * 60 + end.second
            element = (lower_bound_in_seconds, upper_bound_in_seconds, cluster)
            state_intervals.append(element)
        return state_intervals
        
    def getStateSequenceBySamplingStateInveral(self, state_intervals):
        """ 
        Determines a state sequence by sampling through the state_intervals in ascending order.

        Parameters: 
            state_intervals (list<lower_bound_s (int), upper_bound_s (int), state (str)>): list of (interval, state) pairs ordered by time in ascending order
  
        Returns: 
        str: Periodically sampled state sequence of the form "state1, state2, ..."
  
        """
        state_sequence = ''
        if (len(state_intervals) == 0):
            return state_sequence
        first_state = state_intervals[0]
        time_offset = first_state[0]
        number_of_samples = int(self.window_length / self.time_between_samples)
        for i in range(number_of_samples):
            point = time_offset + i * self.time_between_samples
            state_sequence += self.getStateByIntervalMatch(state_intervals, point) + ','
        if state_sequence.endswith(','):
            state_sequence = state_sequence[0:len(state_sequence) - 1]
        return state_sequence

    def getStateByIntervalMatch(self, state_intervals, point_in_time):
        """ 
        Determines the interval of time the given point falls into and returns 
        the state associated with the interval.

        Parameters: 
            state_intervals (list<lower_bound_s (int), upper_bound_s (int), state (str)>): list of interval and state pairs ordered by time in ascended order
            point_in_time (int): The point in time to search the interval for
  
        Returns: 
        str: The state that belongs to the first interval that is being matched or an empty string if no such interval exists
  
        """
        state = ''
        for state_interval in state_intervals:
            lower_bound = state_interval[0]
            upper_bound = state_interval[1]
            if (lower_bound <= point_in_time < upper_bound):
                state = state_interval[2]
                break
        # point_in_time can be equal to or exceed the upper_bound of the last interval. 
        # window_length = 23 * 3600 + 59 * 60 + 59 = 86399 seconds
        # number of samples = 10800
        # point in time = offset + i * time_between_samples
        # the point in time can exceed the the upper_bouond of the last interval: 
        # If this case occurs the last state is appended
        # The following code handles this special case.
        number_of_intervals = len(state_intervals)
        if (state == '' and number_of_intervals > 0 and point_in_time >= state_intervals[number_of_intervals - 1][1]):
            state = state_intervals[number_of_intervals - 1][2]
        return state

    def fillGapsBetweenIntervals(self, state_intervals):
        """ 
        Creates a new list of intervals by making non-overlapping intervals adjacent. 
        This covers two cases. First, if the same states are adjacent and have a "gap" between their intervals the 
        start of the following state is updated to the end of the first state. Second, if two distinct 
        states have a "gap", the gap in time is filled up with equal parts of the time between them.

        Parameters: 
            state_intervals (list<lower_bound_s (int), upper_bound_s (int), state (str)>): list of interval and state pairs ordered by time in ascended order
  
        Returns: 
        state_intervals (list<lower_bound_s (int), upper_bound_s (int), state (str)>): list of adjacent intervals
  
        """
        temp_results = []
        for i in range(len(state_intervals)):
            interval = state_intervals[i]
            if i == 0:
                temp_results.append([interval[0], interval[1], interval[2]])
                continue
            previous_interval = state_intervals[i - 1]
            # make intervals adjacent if they are seperated
            if previous_interval[1] == interval[0]:
                temp_results.append([interval[0], interval[1], interval[2]])
                continue
            previous_state = previous_interval[2]
            state = interval[2]
            # make adjacent intervals of same state seamingless
            if (previous_state == state):
                temp_results.append([previous_interval[1], interval[1], interval[2]])
            # extend both intervals to fill the gap between distinct states
            else:
                delta_t = interval[0] - previous_interval[1]
                a = int(delta_t / 2)
                temp_results[i - 1][1] += a
                temp_results.append([interval[0] - a, interval[1], interval[2]])
        results = []
        for interval in temp_results:
            results.append((interval[0], interval[1], interval[2]))
        return results

    def connectAdjacentStates(self, state_sequence):
        states = state_sequence.split(',')
        q = ""

        for i in range(len(states)):
            if i == 0:
                q += states[i]
                continue
            state1 = states[i - 1]
            state2 = states[i]
            details1 = state1.split(';')
            details2 = state2.split(';')
            cluster2 = details2[0]
            logout_time1 = datetime.datetime.strptime(details1[2], '%H:%M:%S').time().strftime("%H:%M:%S")
            login_time2 = datetime.datetime.strptime(details2[1], '%H:%M:%S').time().strftime("%H:%M:%S")
            logout_time2 = datetime.datetime.strptime(details2[2], '%H:%M:%S').time().strftime("%H:%M:%S")

            if logout_time1 != login_time2:
                q += ',' + cluster2 + ';' + logout_time1 + ';' + logout_time2
            else:
                q += ',' + cluster2 + ';' + login_time2 + ';' + logout_time2

        return q

  
    def addDisconnectedStates(self, window_start, window_end, state_sequence):
        """ 
        Extends an abstract state sequence of the form 
        "state1;login_datetime1;logout_datetime1,state2;login_datetime2;logout_datetime2 ..." by inserting  
        disconnected states if needed. The disconnected states are being inserted if the duration between  
        states exceeds max_duration_between_states. Also, the datetime values after the states are being replaced 
        by their time components only.
  
        Parameters: 
            window_start (time): Starting time of the window
            window_end (time): Ending time of the window
            state_sequence (str): Intial state sequence
  
        Returns: 
        str: State sequence modified by added disconnected states of the form 
        "state1;login_time1;logout_time1,state2;login_time2;logout_time2 ..."
        """
    
        window_start = datetime.datetime.strptime('2018-06-29 ' + window_start, '%Y-%m-%d %H:%M:%S')
        window_end = datetime.datetime.strptime('2018-06-29 ' + window_end, '%Y-%m-%d %H:%M:%S')

        ws = (window_start + datetime.timedelta(seconds = self.max_duration_between_states)).time()
        we = (window_end - datetime.timedelta(seconds = self.max_duration_between_states)).time()

        states = state_sequence.split(',')
        q = ""

        previous_logout_time = window_start
        i = 0

        for state in states:
            details = state.split(';')
            cluster = details[0]
            try:
                login_time = datetime.datetime.strptime(details[1][0:8], '%H:%M:%S').time()
                logout_time = datetime.datetime.strptime(details[2][0:8], '%H:%M:%S').time()
            # catch *all* exceptions
            except:
                e = sys.exc_info()[0]
                print("Error while accessing details", e)
                print("details:")
                print(details)
                print("cluster:")
                print(cluster)
                print("details[1]:")
                print(details[1])
                print("details[2]:")
                # print(details[2])
                quit()

            login_time = datetime.datetime.strptime(details[1][0:8], '%H:%M:%S').time()
            logout_time = datetime.datetime.strptime(details[2][0:8], '%H:%M:%S').time()

            # duration of current cluster
            cluster_scope = login_time.strftime("%H:%M:%S") + ';' + logout_time.strftime("%H:%M:%S")

            a = datetime.datetime(year = 2019, month = 6, day = 6, hour = login_time.hour, minute = login_time.minute, second=login_time.second)
            b = datetime.datetime(year = 2019, month = 6, day = 6, hour = previous_logout_time.hour, minute = previous_logout_time.minute, second=previous_logout_time.second)
            delta_t = a
            if (a > b):
                delta_t = a - b
            else:
                delta_t = b - a
            if i == 0:
                if login_time > ws:
                    # duration for disconnected state inserted before first cluster
                    t1 = window_start.strftime("%H:%M:%S") + ';' + login_time.strftime("%H:%M:%S")
                    q += 'd;' + t1 + ','
                q += cluster + ';' + cluster_scope
                # if there is only one cluster
                if len(states) == 1:
                    if logout_time < we:
                        # duration for disconnected state inserted after last cluster
                        t1 = logout_time.strftime("%H:%M:%S") + ';' + window_end.strftime("%H:%M:%S")
                        q += ',d;' + t1
            elif i > 0 and i < len(states) - 1:
                if delta_t.total_seconds() > self.max_duration_between_states:
                    t1 = previous_logout_time.strftime("%H:%M:%S") + ';' + login_time.strftime("%H:%M:%S")
                    q += ',d;' + t1
                q += ',' + cluster + ';' + cluster_scope
            else:
                if delta_t.total_seconds() > self.max_duration_between_states:
                    t1 = previous_logout_time.strftime("%H:%M:%S") + ';' + login_time.strftime("%H:%M:%S")
                    q += ',d;' + t1
                q += ',' + cluster + ';' + cluster_scope
                if logout_time < we:
                    t1 = logout_time.strftime("%H:%M:%S") + ';' + window_end.strftime("%H:%M:%S")
                    q += ',d;' + t1
            i += 1
            previous_logout_time = logout_time
        return q
