import cv2

def list_ports():
    """
    Test the ports and returns a tuple with the available ports and the ones that are working.
    """
    non_working_ports = []
    dev_port = 0
    working_ports = []
    available_ports = []
    while dev_port < 10: # Test first 10 ports
        cap = cv2.VideoCapture(dev_port)
        if not cap.isOpened():
            non_working_ports.append(dev_port)
            print(f"Port {dev_port} is not working.")
        else:
            is_reading, img = cap.read()
            w = cap.get(3)
            h = cap.get(4)
            if is_reading:
                print(f"Port {dev_port} is working and reads images ({w} x {h})")
                working_ports.append(dev_port)
            else:
                print(f"Port {dev_port} is present but does not read images ({w} x {h})")
                available_ports.append(dev_port)
        cap.release()
        dev_port += 1
    return available_ports, working_ports, non_working_ports

if __name__ == "__main__":
    print("Checking for available cameras...")
    available, working, non_working = list_ports()
    print(f"\nSummary:")
    print(f"Working ports (Camera found and frame read): {working}")
    print(f"Available ports (Camera found but no frame read): {available}")
    print(f"Non-working ports: {non_working}")
