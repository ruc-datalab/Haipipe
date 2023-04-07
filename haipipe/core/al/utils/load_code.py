import json
def ipynb2str(ipypath):
    with open(ipypath, 'r', encoding='utf-8') as f:
        cells = json.load(f)['cells']
    wstr = ''
    for i in range(len(cells)):
        if cells[i]['cell_type'] == 'markdown':
            for j in cells[i]['source']:
                wstr += ('# ' + j)
            wstr += '\n\n'
        elif cells[i]['cell_type'] == 'code':
            if type(cells[i]['source']).__name__ == 'str':
                code_list = cells[i]['source'].split("\n")
                for line in code_list:
                    if len(line) != 0:
                        if line[0] == '%':
                            line = '#' + line[0]
                    wstr += line
                    wstr += '\n'
                wstr += '\n'
            else:
                for line in cells[i]['source']:
                    if len(line) != 0:
                        if line[-1] == '\n':
                            line = line[0:-1]
                    if len(line) != 0:
                        if line[0] == '%':
                            line = '#' + line[0]
                    wstr += line
                    wstr += '\n'
                wstr += '\n'
    code = cleaning_origin(wstr)
    return code

def py2str(filepath):
    with open(filepath, 'r') as f:
        code = f.read()
    return code

def cleaning_origin(code):
    lines = code.split('\n')
    res_str = ''
    for line in lines:
        line1 = line.strip()
        
        if '#' in line:
            num_p = 0
            num_pp = 0
            index = line.index("#")
            for char in line[0:index]:
                if char == "'":
                    num_p += 1
                if char == '"':
                    num_pp += 1
            if num_p %2 == 0 and num_pp %2 == 0:
                line = line[0:index]
        if len(line) != 0:
            line1 = line.strip()
            if line[-1] == '\\':
                res_str += line[0:-1]
            elif len(line1) > 0:
                if line1[-1] == ',':
                    res_str += line
                else:
                    res_str += line
                    res_str += '\n'
            else:
                res_str += line
                res_str += '\n'
    res_str = clean_brackets(res_str)
    return res_str

def clean_brackets(code):
    lines = code.split('\n')
    left_num = 0
    right_num = 0
    left_z_num = 0
    right_z_num = 0
    str_ = ''
    is_in_beizhu = False
    for line in lines:
        num_p = 0
        num_pp = 0
        if line.count('"""') % 2 == 1 or line.count("'''") %2 == 1:
            is_in_beizhu = not is_in_beizhu
        
        if not is_in_beizhu:
            for index,char in enumerate(line):
                if num_pp % 2 ==0 and num_p % 2 ==0:
                    if char == '#':
                        continue
                if char == '"':
                    if index > 0:
                        if line[index-1] != '\\':
                            if num_p % 2 ==0:
                                num_pp += 1
                    else:
                        if num_p % 2 ==0:
                                num_pp += 1
                if char == "'":
                    if index > 0:
                        if line[index-1] != '\\':
                            if num_pp % 2 ==0:
                                num_p += 1
                    else:
                        if num_pp % 2 ==0:
                                num_p += 1
                if char == '(':
                    if num_pp %2 ==0 and num_p %2 == 0:
                        left_num += 1
                if char == ')':
                    if num_pp %2 ==0 and num_p %2 == 0:
                        right_num += 1
                if char == '[':
                    if num_pp %2 ==0 and num_p %2 == 0:
                        left_z_num += 1
                if char == ']':
                    if num_pp %2 ==0 and num_p %2 == 0:
                        right_z_num += 1
                    
        if left_num == right_num and left_z_num == right_z_num:
            str_ += line
            str_ += '\n'
            left_num =0
            right_num = 0
        else:
            str_ += line
    return str_