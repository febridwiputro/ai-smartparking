from src.model.matrix_model import CarCounterMatrix


class MatrixController:
    def __init__(self, arduino_matrix, total_car=18, max_car=18, debug=False):
        self.debug = debug
        self.matrix_model = CarCounterMatrix(matrix=arduino_matrix, total_car= total_car, max_car=max_car, debug=debug)

    def start(self, text="BP0000AA,H;"):
        self.matrix_model.start(text)

    def plus_car(self):
        # self.matrix_model.plus_car()
        self.matrix_model.minus_car()

    def minus_car(self):
        # self.matrix_model.minus_car()
        self.matrix_model.plus_car()

    def get_total(self):
        return self.matrix_model.total_car
    
    def write_arduino(self, text):
        self.matrix_model.edit_total_car(text)
