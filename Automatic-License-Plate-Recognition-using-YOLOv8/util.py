import string
import easyocr
import cv2
import numpy as np

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S'}


# Mapping dictionaries for Indian state codes to state names
dict_state_codes = {
    'AN': 'Andaman and Nicobar Islands', 'AP': 'Andhra Pradesh', 'AR': 'Arunachal Pradesh',
    'AS': 'Assam', 'BR': 'Bihar', 'CH': 'Chandigarh', 'CT': 'Chhattisgarh',
    'DN': 'Dadra and Nagar Haveli', 'DD': 'Daman and Diu', 'DL': 'Delhi',
    'GA': 'Goa', 'GJ': 'Gujarat', 'HR': 'Haryana', 'HP': 'Himachal Pradesh',
    'JK': 'Jammu and Kashmir', 'JH': 'Jharkhand', 'KA': 'Karnataka', 'KL': 'Kerala',
    'LA': 'Ladakh', 'LD': 'Lakshadweep', 'MP': 'Madhya Pradesh', 'MH': 'Maharashtra',
    'MN': 'Manipur', 'ML': 'Meghalaya', 'MZ': 'Mizoram', 'NL': 'Nagaland',
    'OR': 'Odisha', 'PY': 'Puducherry', 'PB': 'Punjab', 'RJ': 'Rajasthan',
    'SK': 'Sikkim', 'TN': 'Tamil Nadu', 'TG': 'Telangana', 'TR': 'Tripura',
    'UP': 'Uttar Pradesh', 'UK': 'Uttarakhand', 'WB': 'West Bengal'
}


def get_region(text):
    """
    Get the region (Indian state) based on the first two characters of the license plate.

    Args:
        text (str): License plate text.

    Returns:
        str: State name if found, 'Unknown' otherwise.
    """
    if text and len(text) >= 2:
        state_code = text[:2].upper()
        # Handle common OCR errors for state codes
        state_code = state_code.replace('0', 'O').replace('1', 'I')
        return dict_state_codes.get(state_code, 'Unknown')
    return 'Unknown'


def write_csv(results, output_path):
    """
    Write the results to a CSV file using the csv module for safety.
    Includes error handling for Permission Denied.
    """
    import csv
    
    # Define columns
    header = ['car_id', 'timestamp', 'license_number', 'lane', 'speed', 'region']
    
    try:
        f = open(output_path, 'w', newline='')
    except PermissionError:
        import time
        timestamp = int(time.time())
        output_path = output_path.replace('.csv', f'_{timestamp}.csv')
        print(f"ERROR: Permission denied. Saving to {output_path} instead.")
        f = open(output_path, 'w', newline='')

    writer = csv.DictWriter(f, fieldnames=header)
    writer.writeheader()

    for frame_nmr in sorted(results.keys()):
        for obj_id in results[frame_nmr].keys():
            if 'license_plate' in results[frame_nmr][obj_id] and \
               'text' in results[frame_nmr][obj_id]['license_plate']:
                
                data = results[frame_nmr][obj_id]
                
                row = {
                    'car_id': obj_id,
                    'timestamp': data.get('timestamp', '00:00:00'),
                    'license_number': data['license_plate']['text'],
                    'lane': data.get('lane', 'N/A'),
                    'speed': data.get('speed', '0'),
                    'region': data.get('region', 'Unknown')
                }
                writer.writerow(row)
    
    f.close()


def license_complies_format(text):
    """
    Check if the license plate text complies with common Indian formats.
    Supports:
    - 7-char: [AA 11 AAA]
    - 8-char: [AA 11 1111]
    - 9-char: [AA 11 A 1111]
    - 10-char: [AA 11 AA 1111]
    """
    length = len(text)
    if length not in [7, 8, 9, 10]:
        return False

    # All Indian plates start with 2 letters (State Code)
    if not ((text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and \
            (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys())):
        return False

    # Second block is always 2 digits (District Code)
    if not ((text[2] in string.digits or text[2] in dict_char_to_int.keys()) and \
            (text[3] in string.digits or text[3] in dict_char_to_int.keys())):
        return False

    # Last block is always 4 digits (Unique Number)
    last_4 = text[-4:]
    for char in last_4:
        if not (char in string.digits or char in dict_char_to_int.keys()):
            return False

    # Middle part (if any) should be letters
    if length == 9: # AA 11 A 1111
        if not (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()):
            return False
    elif length == 10: # AA 11 AA 1111
        if not ((text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()) and \
                (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys())):
            return False
    elif length == 7: # AA 11 AAA (Standard format)
        if not ((text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()) and \
                (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys()) and \
                (text[6] in string.ascii_uppercase or text[6] in dict_int_to_char.keys())):
            return False

    return True


def format_license(text):
    """
    Format the license plate text by converting characters based on position.
    """
    license_plate_ = ''
    length = len(text)
    
    # State code (Letters)
    license_plate_ += dict_int_to_char.get(text[0], text[0])
    license_plate_ += dict_int_to_char.get(text[1], text[1])
    
    # District code (Digits)
    license_plate_ += dict_char_to_int.get(text[2], text[2])
    license_plate_ += dict_char_to_int.get(text[3], text[3])
    
    if length == 7: # AA 11 AAA
        license_plate_ += dict_int_to_char.get(text[4], text[4])
        license_plate_ += dict_int_to_char.get(text[5], text[5])
        license_plate_ += dict_int_to_char.get(text[6], text[6])
    elif length == 8: # AA 11 1111
        license_plate_ += dict_char_to_int.get(text[4], text[4])
        license_plate_ += dict_char_to_int.get(text[5], text[5])
        license_plate_ += dict_char_to_int.get(text[6], text[6])
        license_plate_ += dict_char_to_int.get(text[7], text[7])
    elif length == 9: # AA 11 A 1111
        license_plate_ += dict_int_to_char.get(text[4], text[4])
        license_plate_ += dict_char_to_int.get(text[5], text[5])
        license_plate_ += dict_char_to_int.get(text[6], text[6])
        license_plate_ += dict_char_to_int.get(text[7], text[7])
        license_plate_ += dict_char_to_int.get(text[8], text[8])
    elif length == 10: # AA 11 AA 1111
        license_plate_ += dict_int_to_char.get(text[4], text[4])
        license_plate_ += dict_int_to_char.get(text[5], text[5])
        license_plate_ += dict_char_to_int.get(text[6], text[6])
        license_plate_ += dict_char_to_int.get(text[7], text[7])
        license_plate_ += dict_char_to_int.get(text[8], text[8])
        license_plate_ += dict_char_to_int.get(text[9], text[9])

    return license_plate_


def read_license_plate(license_plate_crop):
    """
    Read the license plate text from the given cropped image.
    Concatenates multiple detections to handle split reads (e.g. 'TN33' and 'J1364').
    """
    # Enhancement: Convert to gray and increase contrast
    gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
    xp = [0, 64, 128, 192, 255]
    fp = [0, 16, 128, 240, 255]
    x = np.arange(256)
    table = np.interp(x, xp, fp).astype('uint8')
    enhanced = cv2.LUT(gray, table)
    
    # Try multiple reads and pick the best concatenation
    best_text = ""
    best_score = 0
    
    for img in [enhanced, license_plate_crop]:
        detections = reader.readtext(img, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        if not detections:
            continue
            
        # Join all detected text pieces horizontally
        # Sort by x-coordinate to ensure correct order
        detections.sort(key=lambda x: x[0][0][0])
        
        full_text = "".join([d[1] for d in detections])
        full_text = full_text.upper().replace(' ', '').replace('-', '').replace('.', '').replace(',', '')
        
        # Average score of all pieces
        avg_score = sum([d[2] for d in detections]) / len(detections)
        
        print(f"DEBUG: Joined OCR text: {full_text} (avg score: {avg_score:.2f})")
        
        if license_complies_format(full_text):
            return format_license(full_text), avg_score
        
        # If not compliant, keep the longest one we've found so far
        if len(full_text) > len(best_text):
            best_text = full_text
            best_score = avg_score

    if len(best_text) >= 4:
        return best_text, best_score

    return None, None


def get_car(license_plate, vehicle_track_ids):
    """
    Retrieve the vehicle coordinates and ID based on the license plate coordinates.
    Using center-point check for more robustness in assignment.

    Args:
        license_plate (tuple): Tuple containing the coordinates of the license plate (x1, y1, x2, y2, score, class_id).
        vehicle_track_ids (list): List of vehicle track IDs and their corresponding coordinates.

    Returns:
        tuple: Tuple containing the vehicle coordinates (x1, y1, x2, y2) and ID.
    """
    x1, y1, x2, y2, score, class_id = license_plate
    
    # Calculate plate center
    px_center = (x1 + x2) / 2
    py_center = (y1 + y2) / 2

    foundIt = False
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]

        # Check if plate center is inside car bbox
        if xcar1 < px_center < xcar2 and ycar1 < py_center < ycar2:
            car_indx = j
            foundIt = True
            break

    if foundIt:
        return vehicle_track_ids[car_indx]

    return -1, -1, -1, -1, -1
