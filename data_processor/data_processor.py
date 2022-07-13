from patterns.singleton import singleton


@singleton
class DataProcessor:
    def __int__(self):
        pass

    @staticmethod
    def load_partition():
        # TODO ez lesz az adott adat partició betöltésért felelős
        pass

data_processor = DataProcessor()
