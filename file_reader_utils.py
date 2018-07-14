import json

def read_json_file(jsonfile):
    with open(jsonfile) as json_file:
        json_data = json.load(json_file)
        return json_data