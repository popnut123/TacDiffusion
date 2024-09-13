import socket
import threading
import time
from helper_functions.DataBuffer_Class import DataBuffer
import onnxruntime
import torch
import struct

def udp_model_receiver(ip_host, port_host, ip_target, port_target, device, ort_session, model_train_timeslot=7):
    """
    Function to receive data via UDP, process it with a model, and send the results back via UDP.
    
    Args:
        ip_host (str): IP address for receiving data.
        port_host (int): Port for receiving data.
        ip_target (str): IP address for sending processed data.
        port_target (int): Port for sending processed data.
        device (str): Device to use for PyTorch (e.g., 'cuda' or 'cpu').
        ort_session (onnxruntime.InferenceSession): ONNX Runtime inference session.
        model_train_timeslot (int): Time slot for the model training, in milliseconds.
    """
    
    # Define UDP sockets for sending and receiving
    udp_socket_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_socket_receive = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_socket_receive.bind((ip_host, port_host))

    print(f"Listening on {ip_host}:{port_host}...")
    print(f"Sending to {ip_target}:{port_target}...")

    data_buffer = DataBuffer()

    data_buffer.horizon_prev = model_train_timeslot
    print(f'**** data_buffer.horizon_prev: {data_buffer.horizon_prev}ms ****')

    send_count = 0  # Count the number of packets sent

    def receive_data():
        """
        Continuously receive data from the UDP socket, process it, and store it in the buffer.
        """
        while True:
            try:
                data, addr = udp_socket_receive.recvfrom(1024)
                # print(f'Received message in original: {data}')

                if data == b'end':
                    print('Finished receiving data!')
                    data_buffer.empty_buffer()
                else:
                    # Decode bytes to string and parse the string into a list of floats
                    data_str = data.decode('utf-8')
                    data_str = data_str.strip('[]')  # Remove leading and trailing brackets
                    message = [float(x) for x in data_str.split(',')]
                    data_buffer.add_data(message)

            except Exception as e:
                print(f"Error receiving data: {e}")

    def send_data():
        """
        Continuously process data from the buffer using the model and send it via UDP.
        """
        nonlocal send_count
        while True:
            message = data_buffer.get_data()
            if message is not None:
                # Process data using the model
                with torch.no_grad():
                    x_eval = torch.Tensor(message).type(torch.FloatTensor).to(device)
                    x_eval_ = x_eval.repeat(1, 1).cpu().numpy()
                    y_pred_ = ort_session.run(['output'], {'input': x_eval_})[0]

                # Prepare data for sending
                payload = y_pred_.flatten().tolist()
                counter = 0
                format_str = "<6b" + str(len(payload)) + "f4b"  # Format string for struct packing
                data_to_send = struct.pack(format_str, 127, 127, 127, 127, counter, len(payload) * 4, *payload, 126, 126, 126, 126)
                udp_socket_send.sendto(data_to_send, (ip_target, port_target))
                send_count += 1  # Increment the count of sent packets

    def print_send_rate():
        """
        Print the rate of data packets sent per second.
        """
        nonlocal send_count
        while True:
            time.sleep(1)
            print(f"Data sent in the last second: {send_count} packets")
            send_count = 0  # Reset the counter

    # Start the data receiving thread
    receive_thread = threading.Thread(target=receive_data)
    receive_thread.daemon = True  # Ensure the thread exits when the main thread exits
    receive_thread.start()

    # Start the send rate printing thread
    rate_thread = threading.Thread(target=print_send_rate)
    rate_thread.daemon = True  # Ensure the thread exits when the main thread exits
    rate_thread.start()

    # Keep the main thread running to allow receiving and sending threads to continue working
    try:
        send_data()
    except KeyboardInterrupt:
        print("Receiver stopped.")
    finally:
        udp_socket_send.close()
        udp_socket_receive.close()

# Example usage

model_size = 512
model_train_timeslot = 7
print(f'!!! Model: {model_size} - {model_train_timeslot}')

if model_size == 128 and model_train_timeslot == 7:
    model_name = 'output/model_128.onnx'
elif model_size == 256 and model_train_timeslot == 7:
    model_name = 'output/model_256.onnx'
elif model_size == 512 and model_train_timeslot == 7:
    model_name = 'output/model_512.onnx'
elif model_size == 1024 and model_train_timeslot == 7:
    model_name = 'output/model_1024.onnx'
else:
    raise ValueError('No suitable model found!')
print(f'Model name: {model_name}')

ip_host = "0.0.0.0"  # IP address of the model computer
port_host = 1501

ip_target = "10.157.175.246"  # IP address of the robot computer
port_target = 2333

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'Device: {device}')
ort_session = onnxruntime.InferenceSession(model_name)
start_time = time.time()
udp_model_receiver(ip_host, port_host, ip_target, port_target, device, ort_session, model_train_timeslot=model_train_timeslot)
