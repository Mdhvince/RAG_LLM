import configparser


def get_config():
    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation(), inline_comment_prefixes="#")
    config.read("config.ini")
    return config["DEFAULT"], config["VECTORSTORAGE"], config["SEARCH"], config["MODEL"]
