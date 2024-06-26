import requests
import base64
import urllib
import os
import pathlib
import json
import re
from PIL import Image
import io
import numpy as np
API_KEY = "RTdKtOMpN1cPf3DiV3875mnm"
SECRET_KEY = "sBTgMaYCX3XzWHQAPrLvk4qZXwTjahgF"
MODEL_API_KEY = "qkjQhbwOaKcrzqAuWBDg2axq"
MODEL_SECRET_KEY = "Ms5T5S6UrKx97hvU64wO4e3oL7xTXnI2"


def get_access_token():
    """
    获取OCR的access_token
    """
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {"grant_type": "client_credentials",
              "client_id": API_KEY, "client_secret": SECRET_KEY}
    return str(requests.post(url, params=params).json().get("access_token"))


def get_model_access_token():
    """
    获取大模型的access_token
    """
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {"grant_type": "client_credentials",
              "client_id": MODEL_API_KEY, "client_secret": MODEL_SECRET_KEY}
    return str(requests.post(url, params=params).json().get("access_token"))


def get_file_content_as_base64(path, urlencoded=False):
    """
    用base64解码图片
    """
    with open(path, "rb") as f:
        content = base64.b64encode(f.read()).decode("utf8")
        if urlencoded:
            content = urllib.parse.quote_plus(content)
    return content


def ocr_analysis():
    """
    识别表格信息并保存到json文件
    """
    print('ocr_analysis')
    url = "https://aip.baidubce.com/rest/2.0/ocr/v1/doc_analysis_office?access_token=" + \
        get_access_token()
    os.chdir(pathlib.Path(__file__).parent.joinpath('input_images').resolve())
    response_text_map = {}
    for root, dirs, files in os.walk(os.getcwd()):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                image = get_file_content_as_base64(file, True)
                payload = f'image={image}&detect_direction=false&line_probability=false&disp_line_poly=false&layout_analysis=false&recg_tables=true&recog_seal=false&recg_formula=false&erase_seal=false&disp_underline_analysis=false&char_probability=false'
                headers = {
                    'Content-Type': 'application/x-www-form-urlencoded',
                    'Accept': 'application/json'
                }
                response = requests.request(
                    "POST", url, headers=headers, data=payload)
                print(file+" has been analyzed")
                response_text_map[file.split('.')[0]] = response.text
    return response_text_map


def get_table_template(cells):
    """
    生成html表格的模板
    """
    html_table = '<table border="1">\n'
    current_row = 0
    html_table += '<tr>'
    for cell in sorted(cells, key=lambda x: (x['row_start'], x['col_start'])):
        row_start = cell['row_start']
        row_end = cell['row_end']
        col_start = cell['col_start']
        col_end = cell['col_end']
        words = cell['words']
        print(row_start, row_end, col_start, col_end, words)
        row_span = row_end - row_start
        col_span = col_end - col_start
        if (row_start > current_row):
            html_table += '</tr>\n<tr>'
            current_row = row_start
        if row_span > 1 or col_span > 1:
            html_table += f'<td rowspan="{row_span}" colspan="{col_span}">{words}</td>'
        else:
            html_table += f'<td>{words}</td>'
    html_table += '</tr>\n</table>'
    return html_table


def transit_json_to_html(data):
    """
    将json数据转换为html表格,并保存到html文件
    """
    table_data_lists = data.get('tables_result', [])
    html_table_template = ''
    if table_data_lists:
        for table_data in table_data_lists:
            table = table_data.get('body')
            html_table_template += get_table_template(table) + '<br>'
    return html_table_template


def ocr_recover(ocr_analysis_map):
    """
    版面恢复主体代码
    """
    print('ocr_recover')
    html_file_map = {}
    for file, json_file in ocr_analysis_map.items():
        html_file = file.split('.')[0] + '.html'
        html_path = pathlib.Path(__file__).parent.joinpath(
            '../output_html').joinpath(html_file).resolve()
        html_data = transit_json_to_html(json.loads(json_file))
        html_file_map[html_file] = html_data
        with open(html_path, 'w') as f:
            f.write(html_data)
        print(file+" has been recovered to "+html_file)
    return html_file_map


def get_prompt(html_data):
    """
    生成prompt模板并添加html,返回结果
    """
    url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/ernie_speed?access_token=" + \
        get_model_access_token()
    payload = json.dumps({
        "messages": [
            {
                "role": "user",
                "content": "这段html代码形成的表格的含税总金额是多少"+html_data+" 用{\"含税总金额\": 8615481.60}这种格式表示,直接写出计算结果即可"
            }
        ]
    })
    headers = {
        'Content-Type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    match = re.search(r'\\?"含税总金额\\?":\s*([0-9.]+)', response.text)
    if match:
        total_amount = float(match.group(1))
        result = {"含税总金额": total_amount}
    else:
        result = {}

    return str(result)


def ocr_extract(html_file_map):
    """
    提取表格中的含税总金额并保存到json文件
    """
    print('ocr_extract')
    os.chdir(pathlib.Path(__file__).parent.joinpath(
        '../output_html').resolve())
    ocr_prompt_map = {}
    for file, html_data in html_file_map.items():
        prompt_response = get_prompt(html_data)
        ocr_prompt_map[file.split('.')[0]] = prompt_response
        print(file+" finish prompt response")
    return ocr_prompt_map


def high_light_png(ocr_analysis_map, ocr_prompt_map):
    """
    高亮图片中的含税总金额
    """
    print('high_light_png')
    input_img_path = pathlib.Path(
        __file__).parent.joinpath('../input_images').resolve()
    output_img_path = pathlib.Path(
        __file__).parent.joinpath('../output_images').resolve()
    for root, dirs, files in os.walk(input_img_path):
        for img_file in files:
            if img_file.endswith('.jpg') or img_file.endswith('.png'):
                file = img_file.split('.')[0]
                with open(input_img_path.joinpath(img_file), 'rb') as img_file:
                    img_data = img_file.read()
                    image = Image.open(io.BytesIO(img_data))
                    img_array = np.array(image)
                    prompt_number = eval(
                        ocr_prompt_map.get(file)).get('含税总金额', 0)
                    data_lists = eval(ocr_analysis_map.get(file)
                                      ).get('results', [])
                    if data_lists:
                        for data in data_lists:
                            result = data.get('words')
                            if result:
                                if result.get('word') and re.match(r'^[0-9]+(\.[0-9]+)?$', result.get('word')):
                                    if prompt_number == float(result.get('word')):
                                        words_location = result.get(
                                            'words_location')
                                        top = words_location.get('top', 0)
                                        left = words_location.get('left', 0)
                                        width = words_location.get('width', 0)
                                        height = words_location.get(
                                            'height', 0)
                                        red = 255
                                        green = 255
                                        blue = 0
                                        for i in range(top, top + height):
                                            for j in range(left, left + width):
                                                if img_array[i][j][0] > 150 and img_array[i][j][1] > 150 and img_array[i][j][2] > 150:
                                                    img_array[i][j][0] = red
                                                    img_array[i][j][1] = green
                                                    img_array[i][j][2] = blue
                                        img = Image.fromarray(img_array)
                                        img.save(output_img_path.joinpath(
                                            file+'_highlighted.png'))
                                        print(file+" has been high lighted")


if __name__ == '__main__':
    # 版面分析OCR
    ocr_analysis_map = ocr_analysis()
    # 版面恢复
    html_file_map = ocr_recover(ocr_analysis_map)
    # 使用Prompt提取信息
    ocr_prompt_map = ocr_extract(html_file_map)
    # 高亮关键信息
    high_light_png(ocr_analysis_map, ocr_prompt_map)
