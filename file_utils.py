import json

def read_json_file(jsonfile):
    with open(jsonfile) as json_file:
        json_data = json.load(json_file)
        return json_data


def read_text_file(text_file):
    with open(text_file, 'r') as train_bbx_file:
        content = train_bbx_file.readlines();
        return content

def save_to_file(save_file, data):
    with open(save_file, mode='wt', encoding='utf-8') as myfile:
        myfile.write("\n".join(data))
    myfile.close()