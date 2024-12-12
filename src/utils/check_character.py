import os, sys
import re

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(src_path)

from src.Integration.service_v1.controller.plat_controller import PlatController

db_plate = PlatController()

def match_plate_no(plate_no):
    """
    Match plate_no with similar plates from the database.
    """
    # Define similar characters mapping
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
        """Calculate similarity score and matched characters between input_plate and db_plate."""
        matches = 0
        matched_characters = ""
        for i in range(min(len(input_plate), len(db_plate))):
            if input_plate[i] == db_plate[i]:
                matches += 1
                matched_characters += input_plate[i]
            elif input_plate[i] in similar_characters_mapping and db_plate[i] in similar_characters_mapping[input_plate[i]]:
                matches += 1
                matched_characters += input_plate[i]
            else:
                matched_characters += "-"
        similarity = matches / max(len(input_plate), len(db_plate)) * 100  # Return percentage similarity
        return similarity, matched_characters

    # Fetch all plates from the database
    plate_no_list = db_plate.get_all_plat()

    # Regex to validate the plate format
    plate_regex = r"^[A-Z]{2}\d{4}[A-Z]{2}$"
    if not re.match(plate_regex, plate_no):
        return False, "Invalid plate number format"

    # Calculate similarity for each plate in the database
    similarity_scores = []
    for db_plate_no in plate_no_list:
        similarity, matched_chars = calculate_similarity(plate_no, db_plate_no)
        similarity_scores.append((db_plate_no, similarity, matched_chars))

    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    print("Close Matches:")
    for plate, similarity, matched_chars in similarity_scores:
        if similarity > 50:
            print(f"Plate: {plate} == {matched_chars}, Similarity: {similarity:.2f}%")

    best_match = similarity_scores[0] if similarity_scores else None
    if best_match and best_match[1] > 50:
        return True, f"Best Match: {best_match[0]} with {best_match[1]:.2f}% similarity"
    return False, "No close match found"

test_plate = "BP1125OZ"
result, message = match_plate_no(test_plate)
print("Result:", result)
print("Message:", message)