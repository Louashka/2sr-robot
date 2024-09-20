import serial

port_name = "COM3"
serial_port = serial.Serial(port_name, 115200)

while True:
    response = serial_port.readline()
    response = response.decode('ascii', errors="ignore")

    print(response)