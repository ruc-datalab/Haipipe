import ast
import astunparse
import re

class FindVariables(ast.NodeVisitor):
    def __init__(self):
        self.variables = []

    def visit_Name(self,node):
        self.variables.append(node.id)
        ast.NodeVisitor.generic_visit(self, node)

def remove_model(code):
    try:
        r_node = ast.parse(code)
    except:
        return code
    lines = code.split('\n')
    fit_caller = {}
    str_ = ''

    need_remove_variables = []
    need_remove_index = []

    for index,line in enumerate(lines): 
        if ('ProfileReport(' in line or 'cross_val_predict(' in line or 'plot(' in line or 'axes[' in line or 'axes.' in line or 'sns.' in line or '.HoloMap(' in line or '.boxcox(' in line or 'plt.' in line) and 'def ' not in line:
            line1 = line.strip()
            try:
                r_node = ast.parse(line1)
            except:
                continue
            if type(r_node.body[0]).__name__ == 'Assign':
                for tar in r_node.body[0].targets:
                    pred_variable = astunparse.unparse(tar)[0:-1]
                    if ',' in pred_variable:
                        pvars = pred_variable[1:-1].split(',')
                        for pvar in pvars:
                            need_remove_variables.append(pvar.strip())
                    else:
                        need_remove_variables.append(pred_variable)
            need_remove_index.append(index)

        if '.fit(' in line or '.fit_resample(' in line:
            line1 = line.strip()
            try:
                r_node = ast.parse(line1)
            except:
                continue
            
            assigned = r_node.body[0].value
            if type(assigned).__name__ == 'Attribute':
                assigned = assigned.value

            args = assigned.args
            if len(args) == 2:
                caller = astunparse.unparse(assigned.func.value)[0:-1]
                if '(' in caller and ')' in caller:
                    if type(r_node.body[0]).__name__ == 'Assign':
                        caller = astunparse.unparse(r_node.body[0].targets[0])[0:-1]
                    else:
                        if 'GridSearchCV' in line:
                            need_remove_index.append(index)
                        continue
                    func_name = astunparse.unparse(assigned.func.value)[0:-1].split('(')[0]
                    if caller not in fit_caller:
                        fit_caller[caller] = {}
                        fit_caller[caller]['index'] = []
                        fit_caller[caller]['func_name']= func_name
        
                if caller not in fit_caller:
                    fit_caller[caller] = {}
                    fit_caller[caller]['index'] = []
                fit_caller[caller]['index'].append(index)
                fit_caller[caller]['need_remove']= True
            else:
                for keyword in assigned.keywords:
                    if keyword.arg == 'y':
                        caller = astunparse.unparse(assigned.func.value)[0:-1]
                        if '(' in caller and ')' in caller:
                            if type(r_node.body[0]).__name__ == 'Assign':
                                caller = astunparse.unparse(r_node.body[0].targets[0])[0:-1]
                            else:
                                if 'GridSearchCV' in line:
                                    need_remove_index.append(index)
                                continue
                            func_name = astunparse.unparse(assigned.func.value)[0:-1].split('(')[0]
                            if caller not in fit_caller:
                                fit_caller[caller] = {}
                                fit_caller[caller]['index'] = []
                                fit_caller[caller]['func_name']= func_name
                
                        if caller not in fit_caller:
                            fit_caller[caller] = {}
                            fit_caller[caller]['index'] = []
                        fit_caller[caller]['index'].append(index)
                        fit_caller[caller]['need_remove']= True
                        break
        if '.best_score_' in line:
            for caller in fit_caller:
                if caller in line:
                    fit_caller[caller]['index'].append(index)
        if '.best_params_' in line:
            for caller in fit_caller:
                if caller in line:
                    fit_caller[caller]['index'].append(index)
        if '.best_estimator_' in line:
            for caller in fit_caller:
                if caller in line:
                    fit_caller[caller]['index'].append(index)
                    
    for index,line in enumerate(lines): 
        try:
            r_node = ast.parse(line).body[0]
        except:
            continue
        if type(r_node).__name__ == 'Assign':
            if astunparse.unparse(r_node.targets[0])[0:-1] in list(fit_caller.keys()):
                if type(r_node.value).__name__ != 'Call':
                    continue
                if type(r_node.value.func).__name__ == 'Name':
                    func_name = astunparse.unparse(r_node.value.func)[0:-1]
                    fit_caller[astunparse.unparse(r_node.targets[0])[0:-1]]['func_name']= func_name
                    
    
    for index,line in enumerate(lines):
        if 'from' in line and 'import' in line and 'feature_selection' in line:
            for caller in fit_caller:
                if 'func_name' in fit_caller[caller]:
                    if fit_caller[caller]['func_name'] in line:
                        fit_caller[caller]['need_remove'] = False
    for caller in fit_caller:
        if fit_caller[caller]['need_remove']==True:
            for ind in fit_caller[caller]['index']:
                need_remove_index.append(ind)
            need_remove_variables.append(caller)

    is_in_def = []
    def_info = {}
    next_line = []
    indent_num = []
    def_name = []
    for index,line in enumerate(lines):
        if len(is_in_def) > 0: 
            if index == def_info[def_name[-1]]['start'] + next_line[-1]:
                all_blank = True
                for char in line:
                    if ord(char) != 32:
                        all_blank=False
                        break
                if all_blank:
                    next_line[-1] += 1
                    continue
                indent_num[-1] = 0
                for char in line:
                    if char != ' ':
                        break
                    indent_num[-1] += 1
                # #print(indent_num)
            else:
                all_blank = True
                for char in line:
                    if ord(char) != 32:
                        all_blank=False
                        break
                if all_blank:
                    continue
                line_indent = 0
                for char in line:
                    if char != ' ':
                        break
                    line_indent += 1
                #print('line_indent', line_indent)
                #print('indent_num', indent_num[-1])
                if line_indent < indent_num[-1]:
                    #print(def_name[-1], "end")
                    def_info[def_name[-1]]['end'] = index -1
                    is_in_def = is_in_def[0:-1]
                    next_line = next_line[0:-1]
                    indent_num = indent_num[0:-1]
                    def_name = def_name[0:-1]
        if 'def ' in line:
            # #print(line)
            def_name.append(line.strip()[4:-1].split('(')[0])
            def_info[def_name[-1]] = {}
            def_info[def_name[-1]]['start'] = index
            is_in_def.append(True)
            next_line.append(1)
            indent_num.append(0)
            # continue
        if line == '':
            continue
        
        if line[0:7] == 'return ':
            # #print(def_name[-1], "return")
            def_info[def_name[-1]]['end'] = index
            is_in_def = is_in_def[0:-1]
            next_line = next_line[0:-1]
            indent_num = indent_num[0:-1]
            def_name = def_name[0:-1]
            # #print("return end")

    if len(is_in_def) > 0:
        def_info[def_name[-1]]['end'] = len(lines)-1
        is_in_def = is_in_def[0:-1]
        next_line = next_line[0:-1]
        indent_num = indent_num[0:-1]
        def_name = def_name[0:-1]
    str_ = ''

    for index,line in enumerate(lines):
        can_parse = True
        for key in def_info:
            all_drop = True
            if index == def_info[key]['end']+1:
                for ind in range(def_info[key]['start']+1, def_info[key]['end']+1):
                    all_blank = True
                    for char in lines[ind]:
                        if ord(char) != 32:
                            all_blank=False
                            break
                    if all_blank:
                        continue
                    if ind not in need_remove_index:
                        all_drop = False
                        break
                if all_drop:
                    def_info[key]['drop'] = True
                    need_remove_variables.append(key)
        try:
            r_node = ast.parse(line.strip()).body[0]
        except:
            can_parse = False
        if '.show(' in line or 'input(' in line or '.imshow(' in line:
            need_remove_index.append(index)
        if can_parse:
            if type(r_node).__name__ == 'Assign':
                for variable in need_remove_variables:
                    if '[' + variable + ']' in astunparse.unparse(r_node.targets)[0:-1]:
                        need_remove_index.append(index)
                        need_remove_variables.append(astunparse.unparse(r_node.targets)[0:-1].split('[')[0])
                    elif variable + '.' in astunparse.unparse(r_node.targets)[0:-1]:
                        need_remove_index.append(index)
                    elif variable + '[' in astunparse.unparse(r_node.targets)[0:-1]:
                        need_remove_index.append(index)
    
                    fv = FindVariables()
                    fv.visit(r_node.value)
                    if variable in fv.variables:
                        for taget in r_node.targets:
                            target_str = astunparse.unparse(taget)[0:-1]
                            if '(' == target_str[0] and ')' == target_str[-1]:
                                target_str = astunparse.unparse(taget)[0:-1][1:-1]
                            split_dou = target_str.split(',')
                            for target_var in split_dou:
                                target_var = target_var.strip()
                                if target_var not in need_remove_variables:
                                    need_remove_variables.append(target_var)
    
                        need_remove_index.append(index)
            else:
                fv = FindVariables()
                fv.visit(r_node)
                for variable in need_remove_variables:          
                    if variable in fv.variables:
                        need_remove_index.append(index)
                        for item in fv.variables:
                            if item not in need_remove_variables and item+'.append(' in line:
                                need_remove_variables.append(item)
                        break
            if line.strip()[0:7] == 'return ':
                try:
                    r_node = ast.parse(line.strip())
                except:
                    continue
            
                fv = FindVariables()
                fv.visit(r_node)
                variables = fv.variables
                for variable in need_remove_variables:
                    if variable in variables:
                        for key in def_info:
                            if index == def_info[key]['end']:
                                def_info[key]['drop'] = True
                                need_remove_variables.append(key)
                        break
    
   
   ##### deal with all for,if,while #ed
    is_indent = []
    indent_num = []
    indent_index = []
    need_remove = []
    next_line = []
    before_indent_num = []
    for index,line in enumerate(lines):
        
        if len(is_indent) > 0:
            if index == indent_index[-1]+next_line[-1]:
                all_blank = True
                for char in line:
                    if ord(char) != 32:
                        all_blank=False
                        break
                if all_blank:
                    next_line[-1] += 1
                    continue
                indent_num[-1] =0
                for char in line:
                    if ord(char) == 32:
                        indent_num[-1] += 1
                    elif ord(char) == 9:
                        indent_num[-1] += 4
                    else:
                        break
                if indent_num[-1] == before_indent_num[-1]:
                    is_indent = is_indent[0:-1]
                    indent_num = indent_num[0:-1]
                    need_remove = need_remove[0:-1]
                    indent_index = indent_index[0:-1]
                    next_line = next_line[0:-1]
            else:
                all_blank = True
                for char in line:
                    if ord(char) != 32:
                        all_blank=False
                        break
                if all_blank:
                    continue
                line_indent =0
                for char in line:
                    if ord(char) == 32:
                        line_indent += 1
                    elif ord(char) == 9:
                        line_indent += 4
                    else:
                        break
                if line_indent != indent_num[-1]:
                    is_indent = is_indent[0:-1]
                    is_all_in = True
                    indent_num = indent_num[0:-1]
                    need_remove = need_remove[0:-1]
                    for ind in range(indent_index[-1]+1, index):
                        all_blank = True
                        for char in lines[ind]:
                            if ord(char) != 32:
                                all_blank=False
                                break
                        if all_blank:
                            continue
    
                        if ind not in need_remove_index:
                            is_all_in=False
                            break
                    
                    if is_all_in:
                        need_remove_index.append(indent_index[-1])
                    indent_index = indent_index[0:-1]
                    next_line = next_line[0:-1]

                    while(len(is_indent)>0 and line_indent != indent_num[-1]):

                        is_indent = is_indent[0:-1]
                        is_all_in = True
                        indent_num = indent_num[0:-1]
                        need_remove = need_remove[0:-1]
                        for ind in range(indent_index[-1]+1, index):
                            all_blank = True
                            for char in lines[ind]:
                                if ord(char) != 32:
                                    all_blank=False
                                    break
                            if all_blank:
                                continue

                            if ind not in need_remove_index:
                                is_all_in=False
                                break
                        
                        if is_all_in:
                            need_remove_index.append(indent_index[-1])
                        indent_index = indent_index[0:-1]
                        next_line = next_line[0:-1]

            if len(need_remove) > 0:
                if need_remove[-1]:
                    need_remove_index.append(index)
        if line.strip()[0:4] == 'for ' or line.strip()[0:3] ==  'if ' or line.strip()[0:6] == 'while ' or line.strip()[0:5] == 'else:' or line.strip()[0:5] == 'elif ' or line.strip()[0:5] == 'with ' or line.strip()[0:4] == 'try:' or line.strip()[0:7] == 'except:' or line.strip()[0:7] == 'except ':
            is_indent.append(True)
            indent_index.append(index)
            indent_num.append(0)
            need_remove_by_v = False
            next_line.append(1)
            before_indent_num.append(0)
            for char in line:
                if ord(char) == 32:
                    before_indent_num[-1] += 1
                elif ord(char) == 9:
                    before_indent_num[-1] += 4
                else:
                    break
            for var in need_remove_variables:
                if var in line:
                    need_remove_by_v = True
                    break
            if len(need_remove) > 0:
                if need_remove[-1]:
                    need_remove_by_v = True
            need_remove.append(index in need_remove_index or need_remove_by_v)
            if need_remove[-1]:
                need_remove_index.append(index)
        
    is_indent = []
    indent_num = []
    indent_index = []

    next_line = 1
    for index,line in enumerate(lines): 
        is_def_drop = False
        is_for = False
        for key in def_info:
            if 'drop' in def_info[key]:
                if index >= def_info[key]['start'] and index <= def_info[key]['end']:
                    str_ += '#'
                    str_ += line
                    str_ += '\n'
                    is_def_drop = True
        if is_def_drop:
            continue
        if index in need_remove_index:
            str_ += '#'
            str_ += line
            str_ += '\n'
        else:
            str_ += line
            str_ += '\n'
    return str_

def remove_model_2(code):
    lines = code.split('\n')
    code_res = ''
    for line in lines:
        line = line.replace("LogisticRegression(","LogisticRegression(solver='liblinear',")
        line = line.replace("display(","#display(")
        code_res+=line+"\n"
    return code_res