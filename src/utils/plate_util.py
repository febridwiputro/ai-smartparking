import re
from src.utils.util import most_freq, check_db, parking_space_vehicle_counter
from src.config.logger import Logger
from src.Integration.service_v1.controller.plat_controller import PlatController


logger = Logger("plate_util", is_save=False)

db_plate = PlatController()

def correct_similar_characters(input_char, similar_characters_mapping):
    """
    Check if a character has visually similar alternatives.
    Return the list of potential matches, including the original character.
    """
    return similar_characters_mapping.get(input_char, [input_char])


def match_plate_no(plate_no):
    """
    Match plate_no with a like method in the database.
    """
    similar_characters_mapping = {
        "0": ["O", "Q", "D"],
        "1": ["I", "l"],
        "2": ["Z"],
        "5": ["S"],
        "6": ["G"],
        "8": ["B"],

        "O": ["0"],
        "Q": ["0"],
        "D": ["0"],
        "I": ["1"],
        "S": ["5"],
        "G": ["6"],
        "B": ["8"],
    }

    def calculate_similarity(input_plate, db_plate):
        matches = 0
        total_length = len(input_plate)
        for i in range(total_length):
            if input_plate[i] == db_plate[i]:
                matches += 1
            elif input_plate[i] in similar_characters_mapping and db_plate[i] in similar_characters_mapping[input_plate[i]]:
                matches += 1
        return (matches / total_length) * 100

    res_plate_no = ""
    similarity = 0
    len_plate_no = len(plate_no)
    plate_no_list = db_plate.get_all_plat()

    if not plate_no[:2].isdigit():
        if len_plate_no == 8:
            match = re.match(r"^([A-Z]{2})(\d{4})([A-Z]{2})$", plate_no)

            if match:
                group1 = match.group(1)  # First 2 characters (e.g., "BP")
                group2 = int(match.group(2))  # Middle 4 digits (e.g., 7062)
                group3 = match.group(3)  # Last 2 characters (e.g., "MF")

                # res_plate_no = f"Group 1: {group1}, Group 2: {group2}, Group 3: {group3}"

                for plate_db in plate_no_list:
                    match_plate_db = re.match(r"^([A-Z]{2})(\d{4})([A-Z]{2})$", plate_db)

                    if match_plate_db:
                        plate_db_group1 = match_plate_db.group(1)
                        plate_db_group2 = int(match_plate_db.group(2))
                        plate_db_group3 = match_plate_db.group(3)

                        db_plate_combined = f"{plate_db_group1}{plate_db_group2}{plate_db_group3}"
                        input_plate_combined = f"{group1}{group2}{group3}"

                        similarity = calculate_similarity(input_plate_combined, db_plate_combined)

                        # if group3_0 == plate_db_group3_0:
                        #     print("plate_db match == ", plate_db)
                        #     if group3_1 == plate_db_group3_1:
                        #         print("Matching plate in database:", plate_db)
                        #         res_plate_no = f"Matched: {plate_db}"
                        #         break
                        #     else:

                        #         print(f"Partial match: Group3_1 mismatch ({group3_1} != {plate_db_group3_1})")

                        # # Check group1 using similar characters
                        # if all(
                        #     db_char in self.correct_similar_characters(user_char, similar_characters_mapping)
                        #     for user_char, db_char in zip(group1, plate_db_group1)
                        # ) and group2 == plate_db_group2:
                        #     if all(
                        #         db_char in self.correct_similar_characters(user_char, similar_characters_mapping)
                        #         for user_char, db_char in zip(group3, plate_db_group3)
                        #     ):
                        #         res_plate_no = f"Matched: {plate_db}"
                        #         break

                        if similarity >= 90:
                            print(f"Matching plate in database with {similarity}% similarity: {plate_db}")
                            # res_plate_no = f"Matched: {plate_db} ({similarity:.2f}% similarity)"
                            plate_no, similarity
                            break
                        else:
                            if similarity >= 80:
                                print(f"No match: {plate_db} has {similarity:.2f}% similarity.")
                                plate_no, similarity
                    else:
                        # print("Plate number in database does not match the expected format.")
                        plate_no, similarity
            else:
                plate_no, similarity
                print("Plate number format is invalid.")

        elif len_plate_no == 6:
            plate_no, similarity
            # BP1768

    else:
        if len_plate_no == 6:
            # Check if the first 4 characters are integers
            if plate_no[:4].isdigit():
                group2 = int(plate_no[:4])  # First 4 characters as integer
                group3 = plate_no[4:]      # Last 2 characters as string

                for plate_db in plate_no_list:
                    match_plate_db = re.match(r"^([A-Z]{2})(\d{4})([A-Z]{2})$", plate_db)

                    if match_plate_db:
                        plate_db_group2 = int(match_plate_db.group(2))
                        plate_db_group3 = match_plate_db.group(3)

                        db_plate_combined = f"{plate_db_group2}{plate_db_group3}"
                        input_plate_combined = f"{group2}{group3}"

                        similarity = calculate_similarity(input_plate_combined, db_plate_combined)

                        if similarity >= 90:
                            print(f"Matching plate in database with {similarity}% similarity: {plate_db}")
                            # res_plate_no = f"Matched: {plate_db} ({similarity:.2f}% similarity)"
                            plate_no, similarity
                            break
                        else:
                            if similarity >= 80:
                                print(f"No match: {plate_db} has {similarity:.2f}% similarity.")
                                plate_no, similarity

                        # if group2 == plate_db_group2:
                        #     if group3 == plate_db_group3:
                        #         print(f"Exact match found for 6-character plate: {plate_no}")
                        #         res_plate_no = plate_no
                        #         break
                        #     else:
                        #         print(f"Group3 mismatch: {group3} != {plate_db_group3}")
                        # else:
                        #     print(f"Group2 mismatch: {group2} != {plate_db_group2}")
            else:
                plate_no, similarity
                print("Invalid format: First 4 characters are not integers.")

    return plate_no, similarity

def process_plate_data(floor_id, cam_id, arduino_idx, car_direction, container_plate_no):
    """
    Processes plate number data and updates the parking status.
    """

    last_plate_no = ""
    res_match_plate_no = ""
    response_counter = {}

    plate_no_list = [data["plate_no"] for data in container_plate_no]
    plate_no_easyocr_list = [data["plate_no_easyocr"] for data in container_plate_no]
    plate_no_max = most_freq(plate_no_list)
    plate_no_easyocr_max = most_freq(plate_no_easyocr_list)
    status_plate_no = check_db(plate_no_max)
    status_plate_no_easyocr = check_db(plate_no_easyocr_max)

    res_plate_no, plate_no_perc = match_plate_no(plate_no=plate_no_max)
    res_plate_no_easyocr, plate_no_easyocr_perc = match_plate_no(plate_no=plate_no_easyocr_max)

    if res_plate_no and res_plate_no_easyocr:
        if plate_no_perc == 100 and plate_no_easyocr_perc == 100:
            res_match_plate_no = res_plate_no
        elif plate_no_perc >= plate_no_easyocr_perc:
            res_match_plate_no = res_plate_no
        else:
            res_match_plate_no = res_plate_no_easyocr

    elif res_plate_no and not res_plate_no_easyocr:
        if plate_no_perc >= 87:
            res_match_plate_no = res_plate_no
        else:
            res_match_plate_no = plate_no_max

    elif res_plate_no_easyocr and not res_plate_no:
        if plate_no_easyocr_perc >= 87:
            res_match_plate_no = res_plate_no_easyocr
        else:
            res_match_plate_no = plate_no_easyocr_max

    else:
        res_match_plate_no = plate_no_max

    print("res_plate_no: ", res_plate_no)
    print("res_plate_no_easyocr: ", res_plate_no_easyocr)

    plate_no_is_registered = True
    if status_plate_no:
        last_plate_no = res_match_plate_no

    elif status_plate_no_easyocr:
        last_plate_no = res_match_plate_no

    elif not status_plate_no or not status_plate_no_easyocr:
        logger.write(
            f"Warning, plate is unregistered, reading container text!! : {last_plate_no}",
            logger.WARN
        )
        last_plate_no = res_match_plate_no
        plate_no_is_registered = False

    response_counter = parking_space_vehicle_counter(
        floor_id=floor_id,
        cam_id=cam_id,
        arduino_idx=arduino_idx,
        car_direction=car_direction,
        plate_no=last_plate_no,
        container_plate_no=container_plate_no,
        plate_no_is_registered=plate_no_is_registered
    )

    container_plate_no = []

    return last_plate_no, response_counter