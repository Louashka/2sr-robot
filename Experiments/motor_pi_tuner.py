import serial
import os
import time
import pandas as pd
import random

portName = "/dev/tty.usbserial-0001"
serial_port = serial.Serial(portName, 115200)

dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'Data/motor_log.csv')

kp = 0.4  # Initial guess for P gain
ki = 0.0  # Initial guess for I gain
kd = 0.0
kp_increment = 0.1  # Increment for P gain

new_kp_start_time = None
new_kp_duration = 3

last_velocity_error = 0  # Store the last velocity error (velocity - target_velocity_velocity)
oscillation_detected = False  # Flag to indicate if an oscillation is detected
oscillation_start_time = None  # Time when the first oscillation was detected
total_oscillations = 0  # Total number of oscillations detected
oscillations_required = 50  # Number of oscillations to detect before calculating Tu
oscillations_threshold = 20

target_velocity = 5
start_time = time.time()

# Function to send data to the microcontroller
def send_to_microcontroller(kp, ki, kd, setpoint, flag):
    msg = "s"

    commands = [kp, ki, kd, setpoint, flag]

    for command in commands:
        msg += str(command) + '\n'

    serial_port.write(msg.encode())

# Function to read data from the microcontroller
def read_from_microcontroller():
    response = serial_port.readline().decode('ascii', errors="ignore").strip()
    # print(response)

    try:
        motor_velocity = float(response)
    except ValueError:
        return False
    
    return motor_velocity

def save_data(velocity):
    current_time = time.time() - start_time

    data = {'velocity': velocity, 'time': [current_time]}
    
    # print(data)
    df = pd.DataFrame(data)

    df.to_csv(filename, mode='a', index=False, header=False)
    # print('Save data')

# Function to detect oscillations
def detect_oscillations(current_velocity, target_velocity):
    global last_velocity_error, oscillation_detected, oscillation_start_time, total_oscillations
    current_velocity_error = current_velocity - target_velocity
    # Check if the velocity error has changed sign (crossed the setpoint)
    if current_velocity_error * last_velocity_error < 0:  # Sign change detected
        if not oscillation_detected:
            # First oscillation detected
            oscillation_detected = True
            oscillation_start_time = time.time()
        else:
            # Subsequent oscillation detected
            total_oscillations += 1
    last_velocity_error = current_velocity_error

def ziegler_nichols():
    global new_kp_start_time, oscillation_detected, total_oscillations, kp

    new_kp_start_time = time.time()
    print(f"New kp: {kp}")
    send_to_microcontroller(kp, ki, kd, target_velocity, 1)

    while True:
        # Read data from the microcontroller
        send_to_microcontroller(kp, ki, kd, target_velocity, 0)
        response = read_from_microcontroller()

        if not response:
            # print('Wrong data')
            continue

        # save_data(response)

        if oscillation_start_time is not None:
            if time.time() - oscillation_start_time >= oscillations_threshold:
                oscillation_detected = False
                total_oscillations = 0

        # Detect oscillations
        detect_oscillations(response, target_velocity)

        # If enough oscillations have been detected, calculate Tu and adjust gains
        if total_oscillations >= oscillations_required:
            # Calculate ultimate period (Tu)
            Tu = (time.time() - oscillation_start_time) / total_oscillations
            # Calculate ultimate gain (Ku) based on the current P gain inducing oscillations
            Ku = kp
            # Apply Ziegler-Nichols formula to set the gains
            # kp_tuned = 0.2 * Ku
            # ki_tuned = 0.4 * kp_tuned / Tu
            # kd_tuned = 0.066 * ki_tuned * Tu

            kp_tuned = 0.6 * Ku
            ki_tuned = 1.2 * kp_tuned / Tu
            kd_tuned = 0.075 * ki_tuned * Tu

            # kp_tuned = 0.45 * Ku
            # ki_tuned = 0.54 * kp_tuned / Tu
            # kd_tuned = 0

            # Send the tuned gains to the microcontroller
            send_to_microcontroller(kp_tuned, ki_tuned, kd_tuned, target_velocity, 1)
            print(f"Tuned P gain: {kp_tuned}")
            print(f"Tuned I gain: {ki_tuned}")
            print(f"Tuned D gain: {kd_tuned}")

            time.sleep(5)
            
            return kp_tuned, ki_tuned, kd_tuned

        # print(time.time() - new_kp_start_time)

        # Increment P gain to induce oscillation if not yet oscillating
        if not oscillation_detected and time.time() - new_kp_start_time >= new_kp_duration:
            kp += kp_increment
            new_kp_start_time = time.time()

            print(f"New kp: {kp}")
            send_to_microcontroller(kp, ki, kd, target_velocity, 1)

def run(kp_tuned, ki_tuned, kd_tuned):
    global target_velocity

    counter = 1
    trials = 10
    duration = 5
    offset = 0

    send_to_microcontroller(kp_tuned, ki_tuned, kd_tuned, 0, 1)
    send_to_microcontroller(kp_tuned, ki_tuned, kd_tuned, 0, 0)
    print('Set PID')
    time.sleep(2)

    trial_start_time = time.time()
    last_time = 0

    target_velocity = random.randint(-10, 11)
    send_to_microcontroller(kp_tuned, ki_tuned, kd_tuned, target_velocity, 0)
    print(f'Set new velocity: {target_velocity}')

    while True:
        if counter > trials:
            break

        send_to_microcontroller(kp_tuned, ki_tuned, kd_tuned, target_velocity, 0)
        response = read_from_microcontroller()

        if response == -1:
            offset += time.time() - last_time
            continue

        save_data(response)
        # print(response)

        time_counter = time.time() - trial_start_time
        # print(time_counter)

        if time_counter >= duration + offset:
            target_velocity = random.randint(-10, 11)
            send_to_microcontroller(kp_tuned, ki_tuned, kd_tuned, target_velocity, 0)
            print(f'Set new velocity: {target_velocity}')

            trial_start_time = time.time()
            counter += 1
            offset = 0

        last_time = time.time()

if __name__ == "__main__":
    f = open(filename, "w+")
    f.close()

    # kp_tuned, ki_tuned, kd_tuned = ziegler_nichols()
    
    # run(kp_tuned, ki_tuned, kd_tuned)


    while True:
        # send_to_microcontroller(0.3, 0, 0, 5, 0)
        velocity = read_from_microcontroller()
        if velocity:
            save_data(velocity)

        print(velocity)
    
        
# if __name__ == "__main__":
#     msg = "s"

#     kp = 0.12 
#     ki = 1.473 
#     kd = 0.00

#     send_to_microcontroller(kp, ki, kd, 0, 1)
#     send_to_microcontroller(kp, ki, kd, 0, 1)

#     send_to_microcontroller(kp, ki, kd, -3, 1)
#     send_to_microcontroller(kp, ki, kd, -3, 1)

