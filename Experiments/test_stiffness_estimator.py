import serial
import time
import json
import matplotlib.pyplot as plt

port_name = "COM3"
serial_port = serial.Serial(port_name, 115200)
    
class NonLinearTemperatureEstimator():
    def __init__(self):
        self.room_temp = 22
        self.max_alloy_temp = 70
        self.max_sensor_temp = 40
        self.cooling_factor = 0.01
        self.last_estimate = self.room_temp

    def estimate(self, meas_temp, action):
        # if action == 1:
        #      # Non-linear mapping for heating phase
        #     normalized_sensor = (meas_temp - self.room_temp) / (self.max_sensor_temp - self.room_temp)
        #     estimated_temp = self.room_temp + normalized_sensor * (self.max_alloy_temp - self.room_temp)
        # elif action == -1:
        #     estimated_temp = self.last_estimate - (self.last_estimate - meas_temp) * self.cooling_factor
        # else:
        #     estimated_temp = meas_temp

        normalized_sensor = (meas_temp - self.room_temp) / (self.max_sensor_temp - self.room_temp)
        estimated_temp = self.room_temp + normalized_sensor * (self.max_alloy_temp - self.room_temp)

        self.last_estimate = estimated_temp
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

            status = self.readTemperature(actions)
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


    def readTemperature(self, actions) -> bool:
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
                    self.temp_filtered[i] = self.temp_estimators[i].estimate(temperature, actions[i])
                else:
                    return False
            except ValueError:
                return False
            
            return True
    
def sendCommands( commands) -> None:
    msg = "s" + "".join(f"{command}\n" for command in commands)
    serial_port.write(msg.encode())
    

if __name__ == "__main__":
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

    json_path = f'Data/temp_estimation.json'
    with open(json_path, 'w') as f:
        json.dump(data_json, f, indent=2)

    print()
    print("JSON file 'tracking_morph_data.json' has been created.")

    plt.plot(time_list, meas, label='measured')
    plt.plot(time_list, meas_filtered, label='filtered')

    plt.legend()
    plt.axis('equal')
    plt.show()


