import sys
import re
import requests
import base64


def main():

    md_path ='./Logistic Regression( Gradient Descent）.md'
    with open(md_path, 'r', encoding='utf-8') as f:
        str = f.read()
    str += '\n\n\n\n\n\n'
    img_path = re.findall('!\[.*?\]\((.*?)\)', str)
    key = 0
    for path in img_path:
        if re.match('http.://.*', path):
            try:
                data = requests.get(url=path).content
            except Exception:
                print('[warn]请检查网络连接或图片链接(%s)是否有效' % path)
                continue
        else:
            try:
                with open(path, 'rb') as f:
                    data = f.read()
            except FileNotFoundError:
                print('[warn]请检查本地文件(%s)是否存在' % path)
                continue
        print('[success]%s' % path)
        base64_str = base64.b64encode(data).decode()
        str = str.replace('(%s)' % path, '[%d]' % key)
        str += '[%d]:data:image/jpg;base64,%s\n\n' % (key, base64_str)
        key += 1
    with open('result.md', 'w+', encoding='utf-8') as f:
        f.write(str)


main()