import serial
import time
import json
import numpy as np
import matplotlib.pyplot as plt

port_name = "COM3"
serial_port = serial.Serial(port_name, 115200)

json_path = f'Experiments/Data/temp_estimation.json'
    
class NonLinearTemperatureEstimator():
    def __init__(self):
        self.room_temp = 22
        self.max_alloy_temp = 70
        self.max_sensor_temp = 40

    def estimate(self, meas_temp):
        normalized_sensor = (meas_temp - self.room_temp) / (self.max_sensor_temp - self.room_temp)
        estimated_temp = self.room_temp + normalized_sensor * (self.max_alloy_temp - self.room_temp)

        return round(estimated_temp, 4)
    
class StiffnessController:
    def __init__(self):
        self.states = [0, 0]  # Initial state

        self.liquid_threshold = 63
        self.solid_threshold = 53

        self.temp = [0, 0]
        self.temp_filtered = [0, 0]

        self.temp_estimators = [
            NonLinearTemperatureEstimator(),
            NonLinearTemperatureEstimator()
        ]

        self.experiment_start_time = None


    def control_loop(self, current_states: list, target_states: list):
        self.states = current_states

        meas = []
        meas_filtered = []
        time_list = []

        if self.experiment_start_time is None:
            self.experiment_start_time = time.perf_counter()

        while True:
            actions = self.getActions(target_states)

            if actions == (0, 0):
                break

            status = self.readTemperature()
            if not status:
                continue

            meas.append(self.temp[0])
            meas_filtered.append(self.temp_filtered[0])

            experiment_current_time = time.perf_counter()
            elapsed_time = experiment_current_time - self.experiment_start_time
            elapsed_time = round(elapsed_time, 4)

            time_list.append(elapsed_time)

            self.applyActions(actions)

        print(f'Elapsed time: {elapsed_time}')
        print()

        return meas, meas_filtered, time_list

    def getActions(self, target_states):
        actions = (self.getAction(self.states[0], target_states[0]),
                   self.getAction(self.states[1], target_states[1]))
        
        return actions

    def getAction(self, state, target_state):
        if state == target_state:
            return 0
        elif state == 0 and target_state == 1:
            return 1
        else:
            return -1
        
    def applyActions(self, actions):
        for i in range(len(actions)):
            self.applyAction(i, actions[i])

    def applyAction(self, i, action):
        if action == 1:
            if self.temp_filtered[i] >= self.liquid_threshold:
                self.states[i] = 1
            else:
                print(f'Switching segment {i+1} to soft...')
                print(f'Current temp: {self.temp[i]}')
                print(f'Current temp filtered: {self.temp_filtered[i]}')
                print()
        if action == -1:
            if self.temp_filtered[i] <= self.solid_threshold:
                self.states[i] = 0
            else:
                
                print(f'Switching segment {i+1} to rigid...')
                print(f'Current temp: {self.temp[i]}')
                print(f'Current temp filtered: {self.temp_filtered[i]}')
                print()


    def readTemperature(self) -> bool:
            response = serial_port.readline()
            response = response.decode('ascii', errors="ignore")

            try:
                temperature = float(response[1:])

                i = -1
                if response[0] == 'A':
                    i = 0
                if response[0] == 'B':
                    i = 1

                if i != -1:
                    self.temp[i] = temperature
                    self.temp_filtered[i] = self.temp_estimators[i].estimate(temperature)
                else:
                    return False
            except ValueError:
                return False
            
            return True
    
def sendCommands(commands) -> None:
    msg = "s" + "".join(f"{command}\n" for command in commands)
    serial_port.write(msg.encode())

def run_exp() -> None:
    sc = StiffnessController()

    s = [1, 0]
    commands = [0] * 4 + s + [1]
    sendCommands(commands)
    meas1, meas_filtered1, time_list1 = sc.control_loop([0, 0], s)

    s = [0, 0]
    commands = [0] * 4 + s + [1]
    sendCommands(commands)
    meas2, meas_filtered2, time_list2 = sc.control_loop([1, 0], s)

    meas = meas1 + meas2
    meas_filtered = meas_filtered1 + meas_filtered2
    time_list = time_list1 + time_list2

    data_json = {
        'measured': meas,
        'filtered':meas_filtered,
        'time': time_list
    }

    with open(json_path, 'w') as f:
        json.dump(data_json, f, indent=2)

    print()
    print("JSON file 'tracking_morph_data.json' has been created.")

    plt.plot(time_list, meas, label='measured')
    plt.plot(time_list, meas_filtered, label='filtered')

    plt.legend()
    plt.axis('equal')
    plt.show()

 # Apply moving average filter (n = 3)
def moving_average(data, n=3):
    cumsum = np.cumsum(np.insert(data, 0, 0)) 
    return (cumsum[n:] - cumsum[:-n]) / float(n)

def analyse_exp() -> None:
    data = None

    try:
        with open(json_path, "r") as json_file:
            data = json.load(json_file)
    except FileNotFoundError:
        print(f"Error: {json_path} not found.")
        return
    
    if data is not None:
        meas = data['measured']
        filtered = data['filtered']
        elapsed_time = data['time']

        measured_temp = moving_average(meas)
        estimated_temp = moving_average(filtered)

        # Adjust time list to match the length of filtered data
        elapsed_time = elapsed_time[1:-1]
        import seaborn as sns

        # Set the style for a clean, modern look
        plt.style.use('seaborn-v0_8-whitegrid')

        # Create a custom color palette
        colors = sns.color_palette("colorblind", 4)

        # Create a figure with 2 subplots in a row
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Measured vs Estimated Temperature
        ax1.plot(elapsed_time, measured_temp, label='Measured', color=colors[0], linewidth=2)
        ax1.plot(elapsed_time, estimated_temp, label='Estimated', color=colors[3], linewidth=2)

        # Find the first occurrence of measured temperature = 62
        target_temp = 62
        target_index = next((i for i, temp in enumerate(estimated_temp) if temp >= target_temp), None)

        if target_index is not None:
            target_temp = estimated_temp[target_index]
            target_time = elapsed_time[target_index]
            circle = plt.Circle((target_time, target_temp), 0.6, color='black', linewidth=2, label='Melting point')
            ax1.add_artist(circle)
            circle.set_zorder(2)

        ax1.set_xlabel('Time [s]', fontsize=12)
        ax1.set_ylabel('Temperature [°C]', fontsize=12)
        ax1.set_title('Measured vs Estimated Temperature', fontsize=14, fontweight='bold')
        ax1.set_aspect('equal', adjustable="datalim")
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='both', which='major', labelsize=10)

        # Histogram of Errors
        ax2.hist(estimated_temp - measured_temp, bins=20, rwidth=0.8, color=colors[0])

        ax2.set_xlabel('Estimation Error', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Distribution of Estimation Errors', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='both', which='major', labelsize=10)

        plt.tight_layout()
        plt.show()

        fig.savefig('Experiments/Figures/temp_estimation.pdf', dpi=300, format="pdf", transparent=True, bbox_inches='tight')
    

if __name__ == "__main__":
    # run_exp()
    analyse_exp()

