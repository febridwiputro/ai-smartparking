import time



class CarCounterMatrix:
    def __init__(self, matrix, total_car, max_car, debug):
        """

        Args:
            matrix: board arduino
            total_car: total mobil yang akan di hitung
            max_car: total keseluruhan mobil
        """

        self.matrix = matrix  # controller mobil
        # self.total_car = total_car
        self.total_car = total_car if total_car is not None else 0
        self.max_car = max_car
        self.debug = debug

    def start(self, text= "BP0000AA,H;"):
        self.edit_total_car(text)


    def plus_car(self):
        self.total_car += 1
        if self.total_car > self.max_car:
            self.total_car = self.max_car
        self.edit_total_car(self.total_car)

    def minus_car(self):
        self.total_car -= 1
        if self.total_car < 0:
            self.total_car = 0
        self.edit_total_car(self.total_car)

    def edit_total_car(self, total_car):
        if not self.debug:
            self.matrix.write(str(total_car))
            self.matrix.sending()


if __name__ == "__main__":
    from src.Integration.arduino import Arduino
    arduino = Arduino(baudrate=115200, driver="CP210")
    matrix = CarCounterMatrix(arduino, total_car=0, max_car=50, debug=False)
    for i in range(55):
        matrix.plus_car()
        print(matrix.total_car)
        time.sleep(1)