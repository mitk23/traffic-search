import datetime


class Logger:
    def __init__(self, fname=None):
        self.fname = fname

    def log(self, message):
        if self.fname is not None:
            self.log2file(message)

        message = self.__get_message(message)

        print(message)
        return None

    def log2file(self, message):
        message = self.__get_message(message)

        with open(self.fname, mode="a", encoding="utf-8") as f:
            print(message, file=f)

    def __get_message(self, message):
        now = datetime.datetime.now()
        message = f"{now} | {message}"
        return message
