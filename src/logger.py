import datetime


class Logger:
    def __init__(self, stdout=True, fname=None):
        self.stdout = stdout
        self.fname = fname

    def log(self, message):
        message = self.__get_message(message)

        if self.stdout:
            print(message)
        if self.fname:
            self.log2file(message)

    def log2file(self, message):
        with open(self.fname, mode="a", encoding="utf-8") as f:
            print(message, file=f)

    def __get_message(self, message):
        now = datetime.datetime.now()
        message = f"{now} | {message}"
        return message
