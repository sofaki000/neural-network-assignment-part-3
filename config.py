cifar_path ="C:\\Users\\Lenovo\\Desktop\\νευρωνικά δίκτυα\\neural-network-assignment-part-3\\data"


def get_datetime_for_filename():
    from datetime import datetime

    date = datetime.now()
    h = date.hour
    filename=f'm={date.month}_d={date.day}_h={h}_min={date.minute}'
    return filename