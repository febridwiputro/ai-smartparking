import re

class PlateProcessor:
    def __init__(self) -> None:
        self.middle_char_mapping = {
            'T': '1', 
            'I': '1', 
            'D': '0', 
            'B': '8',
            'Q': '0', 
            'J': '1', 
            'Z': '7'
        }

        self.suffix_char_mapping = {
            '0': 'Q', 
            '8': 'O'
        }

    def apply_mapping(self, text, mapping):
        """
        Replace characters in `text` based on the provided `mapping`.
        """
        return ''.join([mapping.get(char, char) for char in text])

    def match_char(self, plate):

        plate = re.sub(r"[^A-Za-z0-9]", "", plate)
        plate = plate.replace(" ", "").upper()

        if plate.startswith('8'):
            plate = 'B' + plate[1:]

        pattern = r"^(.{2})(.{0,4})(.*?)(.{2})$"

        def replace(match):
            prefix = match.group(1)
            middle = match.group(2)
            body = match.group(3)
            suffix = match.group(4)

            modified_middle = self.apply_mapping(middle, self.middle_char_mapping)

            if re.match(r"^[A-Z]{2}\d{4}$", f"{prefix}{modified_middle}"):
                modified_suffix = self.apply_mapping(suffix, self.suffix_char_mapping)
            else:
                modified_suffix = suffix

            modified_plate = f"{prefix}{modified_middle}{body}{modified_suffix}"
            match_special_case = re.match(r"(\d{4})(.*)(BP)$", modified_plate)
            if match_special_case:
                return f"BP{match_special_case.group(1)}{match_special_case.group(2)}"

            return modified_plate

        result = re.sub(pattern, replace, plate)
        result = result.strip()
        print("plate_no:", result)

        return result

run = PlateProcessor()
plate_no = "BP1030'KZ"
run.match_char(plate_no)