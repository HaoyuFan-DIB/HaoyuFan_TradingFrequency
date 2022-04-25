import os

RootPath = os.path.dirname(os.path.realpath(__file__))

with open("/Users/haoyufan/config.txt") as f:
    access_key_id = f.readline().replace("\n", "")
    secret_access_key = f.readline().replace("\n", "")