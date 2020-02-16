import requests
import time
import copy
import freesms

class Report:
    def __init__(self):
        self.report = ""

    def add(self, string):
        self.report += string + "\n"

    def send(self):
        print(self.report)
        return True


class Range:

    def __init__(self, minimum, maximum):
        self.minimum = min(minimum, maximum)
        self.maximum = max(minimum, maximum)

    def range(self):
        return abs(self.maximum - self.minimum)


class Set:
    """
    Needed:
    Local min, local max <-- magnitudes of drops and gains
    """

    def __init__(self, file_path, sym):
        self.sym = sym
        self.data = self.parse_data(file_path, sym)
        self.range = Range(0, 0)

        try:
            self.mean = self.calculate_mean()
            self.std_dev = self.calculate_std_dev()
        except ZeroDivisionError:
            self.mean = 0
            self.std_dev = 0

    def add(self, data):
        self.data.append(float(data['latestPrice']))
        if len(self.data) > 60 * 24:
            self.data = self.data[1:len(self.data)]
        diff_from_mean = self.dist(data['latestPrice'], self.mean)
        dist_from_std_dev = self.dist(diff_from_mean, self.std_dev)
        update_symbols(data)
        old_range = copy.copy(self.range)
        self.calibrate()

        if self.analyze(data['latestPrice'], diff_from_mean, dist_from_std_dev, old_range, self.range):
            report.send()

    def analyze(self, price, diff_from_mean, dist_from_std_dev, old_range, new_range):
        send = False

        percent_diff_std_dev = (dist_from_std_dev / self.std_dev) * 100
        direction = self.direction(price, 5)
        direction /= abs(direction)  # 1 or -1

        if percent_diff_std_dev > 60:
            send = True
            if direction > 0:
                report.add("Up: " + percent_diff_std_dev)
            elif direction < 0:
                report.add("Down: " + percent_diff_std_dev)

        if new_range.range > old_range * 1.5:
            send = True
            report.add("Range Increased by: " + (new_range / old_range))

    def direction(self, price, i):
        direction = 0
        if i == 0:
            return direction
        try:
            direction = price - self.get(self.length() - 5)
            return direction
        except IndexError:
            return direction(price, i - 1)

    def calibrate(self):
        self.update_range()
        self.std_dev = self.calculate_std_dev()
        self.mean = self.calculate_mean()

    def update_range(self):
        self.range.maximum = max(self.data)
        self.range.minimum = min(self.data)

    def print_attr(self):
        for attr in dir(self):
            print(attr + ": " + str(self.__getattribute__(attr)))

    def calculate_std_dev(self):
        dsum = 0.0
        for data in self.data:
            dsum += abs(data - self.mean)
        return dsum / len(self.data)

    def calculate_mean(self):
        dsum = 0.0
        for i in self.data:
            dsum += float(i)
        return dsum / len(self.data)

    def get_data(self):
        return self.data

    def length(self):
        return len(self.data)

    def get(self, i):
        return self.data[i]

    @staticmethod
    def parse_data(file_path, sym):
        file = open(file_path, "r")
        lines = file.readlines()
        i = find_line_for_symbol(sym, lines)

        data = lines[i].split(", ")
        if len(data) == 3:
            data = [data[1]]
        else:
            data = data[1:len(data) - 1]

        for j in range(len(data)):
            data[j] = float(data[j])

        return data

    @staticmethod
    def dist(this, that):
        return abs(float(this) - float(that))


alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ:.0123456789"
numbers = "1234567890"
token = "sk_861a4746f9784f83b68c4717e9e06a9b"

ONE_MINUTE = 60

symbols = []
report = Report()


# string input json data stage 0 parsed
def format_quotes(data):
    data += " "
    out = ""
    for i in range(len(data) - 1):
        if data[i + 1] in numbers and data[i] not in numbers:  # will never be negative (stock values)
            out += data[i] + "'"
        elif data[i + 1] not in numbers and data[i] in numbers:
            out += data[i] + "'"
        else:
            out += data[i]
    return out


# check iff string contains number
def is_number(string):
    for i in range(len(string)):
        if not string[i].isdigit():
            return False
    return True


# data is a list of values distinct by separance of numbers from words
def join_decimal_values(data):
    out = []
    conjoined = False
    for i in range(len(data) - 1):
        if conjoined:
            conjoined = False
            continue
        if is_number(data[i]) and is_number(data[i + 1]):
            out.append(data[i] + "." + data[i + 1])
            conjoined = True
        else:
            out.append(data[i])

    return out


# generates list with elements indicated by surrounding quote bars data stage 1 parsed
def extract_list(data):
    word_list = []
    on = False
    word = ""
    for i in range(len(data)):
        if data[i] == "'" and not on:
            on = True
            continue
        if data[i] == "'" and on:
            on = False
            word_list.append(word)
            word = ""
            continue
        if on:
            word += data[i]
    return word_list


# returns dictionary with parsed data, data stage 2 parsed
def associate_map(data_list):
    map = {}
    for i in range(0, len(data_list) - 1, 2):
        map.update({data_list[i]: data_list[i + 1]})
    return map


# generate data map, param = http request
def data_map(response):
    return associate_map(join_decimal_values(extract_list(format_quotes(str(response.json())))))


# add a target symbol
def add_symbol(symbol):
    if symbol not in symbols:
        symbols.append(symbol)

    stocks = open("Stocks.csv", "r")
    lines = stocks.readlines()

    new_lines = lines.copy()
    if not contains(lines, symbol):
        new_lines.append("\n" + symbol + ", ")

    stocks = open("Stocks.csv", "w+")
    for line in new_lines:
        stocks.write(line)


# checks if list contains symbol
def contains(file_lines, symbol):
    for i in range(len(file_lines)):
        if symbol in file_lines[i] and symbol[0] == file_lines[i][0]:
            return True
    return False


# finds symbol location in lines
def find_line_for_symbol(sym, lines):
    for i in range(len(lines)):
        if sym in lines[i]:
            return i


# remove blank new lines
def remove_open_spaces(lines):
    new_lines = []
    for line in lines:
        if line != "\n":
            new_lines.append(line)
    return new_lines


# update one line in the records file
def update_symbols(data):
    stocks = open("Stocks.csv", "r")
    lines = stocks.readlines()
    lines = remove_open_spaces(lines)
    i = find_line_for_symbol(data["symbol"], lines)
    lines[i] = lines[i].replace("\n", "")
    lines[i] = lines[i] + data["latestPrice"] + ", \n"
    stocks = open("Stocks.csv", "w")
    for line in lines:
        stocks.write(line)


# def add_new_data():
#     for sym in symbols:
#         url = "https://cloud.iexapis.com/stable/stock/" + sym + "/quote?token=" + token
#         response = requests.get(url)
#         data = data_map(response)
#         update_symbols(data)

# keep working on this
def add_new_data(data_set):
    url = "https://cloud.iexapis.com/stable/stock/" + data_set.sym + "/quote?token=" + token
    response = requests.get(url)
    data = data_map(response)
    data_set.add(data)


def main():
    add_symbol("FB")
    add_symbol("T")
    add_symbol("CMG")
    add_symbol("COST")
    add_symbol("FSCT")

    map = {}
    for sym in symbols:
        map[sym] = Set("Stocks.csv", sym)

    while True:

        for sym in symbols:
            # data_set = Set("Stocks.csv", sym)
            data_set = map[sym]
            add_new_data(data_set)
            # data_set.print_attr()

            # print("\n")
            # print("data:")
            # print(data_set.data)
            # print("mean: ", end="")
            # print(data_set.mean)
            # print("Standard Deviation: ", end="")
            # print(data_set.std_dev)

        time.sleep(ONE_MINUTE)


# deploy to amazon ec2 free
main()
