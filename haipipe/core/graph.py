
import ast
import astunparse
from haipipe.core.read_ipynb import ipynb2py
import matplotlib.pyplot as plt
import os
import pickle
import pprint
import json

colorset = {
    0:'#00BFFF',
    1:'#7FFF00',
    2:'#EE82EE',
    3:'#ff0000',
    4:'#9932CC',
    5:'#2E8B57',
    6:'#0000CD',
    7:'#BDB76B',
    8:'#FF8C00',
    9:'#8B0000',
    10:'#558866'
    }

white_opes = [
    'fit_transform',
    'transform',
    'fit',
    'confusion_matrix',
    'sum',
    'remove',
    'append',
    'drop',
    'unstack',
    'reshape',
    'replace',
    'drop_duplicates',
    'groupby',
    'merge',
    'reset_index',
    'join',
    'sort_values',
    'concat',
    'get_dummies',
    'fillna',
    'dropna',
]

class Node():
    def __init__(self, sorted_id, varible_name):
        self.sorted_id = sorted_id
        self.childrens = []
        self.children_edges = []
        self.node_type = 0 # 0 is not useful, 1 is data
        self.varible_name = varible_name
        self.is_dataset_node = False
        self.is_model_node = False
    
    def add_edge(self, child, edge):
        self.childrens.append(child)
        self.children_edges.append(edge)
    def show(self):
        #print("-------Node "+str(self.sorted_id)+"--------")
        #print(self.varible_name)
        #print(self.sorted_id)
        child_str = ''
        for children in self.childrens:
            child_str += str(children.varible_name)
            child_str += ','
        #print(child_str)
        edge_str = ''
        for edge in self.children_edges:
            edge_str += str(edge.edge_id)
            edge_str += str(edge.func_name)
            edge_str += ','
        #print(edge_str)
        
class Edge():
    def __init__(self, edge_id, sorted_id, func_name, original_code, edge_type):
        self.edge_id = edge_id
        self.sorted_id = sorted_id
        self.edge_type = edge_type # 0 is black, 1 is white
        self.func_name = func_name
        self.original_code = original_code
        self.is_dataset_edge = False
        self.is_model_edge = False
        self.line_id = []
    def show(self):
        pass
        #print("-------Edge "+str(self.edge_id)+"--------")
        #print(self.sorted_id)
        #print(self.edge_type)
        #print(self.func_name)
        #print(self.original_code)


class NotebookGraph(ast.NodeVisitor):
    def __init__(self):
        super().__init__()
        self.colorset = colorset
        self.white_opes = white_opes

        self.data_varibles_to_current_node = {}
        self.defed_functions = []
        self.nodes = []
        self.result_nodes = []
        self.result_edges = []
        self.model_variable = []
        self.edges = []
        self.sorted_node_id = 0
        self.edge_id = 0
        self.sorted_edge_id = 0

        self.is_if = False
        self.is_for = False
        self.is_while = False
        self.is_with = False
        self.is_try = False
        self.is_subscript = False
        self.start_get_varible = False
        self.start_get_varible_with_sub = False
        self.start_get_varible_insub = False
        self.start_get_varible_flow = False
        self.should_create_white_edge = False
        self.should_create_black_edge = False
        self.white_edge_node = None
        self.get_varible_results = []

        self.current_flow_id = 0
        self.varible_flow = {}
        self.root_node = None
        self.code = ''

    def is_highlighted(self):
        for edge in self.result_edges:
            if edge.edge_type == 1:
                return True
        return False

    def build_graph(self):
        self.extract_all_user_defined_function(self.root_node)
        self.visit(self.root_node)
        self.nodes_sorted()
        self.point_dataset_node()
        self.point_model_node()

    def load_file(self, filepath):
        self.file_path = filepath
        if '.ipynb' in filepath:
            output_path = filepath.replace(".ipynb",'.py')
            # #print(output_path)
            output_path = output_path.replace("/home/datamanager/dataset/statsklearn/notebook/",'pycode/')
            #print(output_path)
            ipynb2py(filepath, output_path)
        # if '.py' in filepath:
        else:
            output_path = filepath
        with open(output_path, 'r') as src_file:
            src = src_file.read()
            self.code = src
        try:
            self.root_node = ast.parse(src, mode='exec')
        except:
            #print("parse fail")
            return "error"

    def look_edges(self):
        #print(len(self.edges))
        for edge in self.edges:
            edge.show()

    def look_nodes(self):
        #print(len(self.nodes))
        for node in self.nodes:
            node.show()
    def nodes_sorted(self):
        self.nodes.sort(key=lambda t:t.sorted_id)

    def layout(self, directory, all_nodes=True):
        filename = self.file_path.split("/")[-1]
        filename = filename.replace(".py",'')
        
        if all_nodes:

            g = Digraph(name = filename,filename = filename,directory = directory,format = "png")
            for node in self.nodes:
                g.node(str(node.sorted_id),str(node.varible_name))
            
            i = 0
            for node in self.nodes:
                for index,childnode in enumerate(node.childrens):
                    i+=1
                    g.edge(tail_name = str(node.sorted_id),head_name = str(childnode.sorted_id),label=str(node.children_edges[index].func_name),color=self.colorset[node.children_edges[index].edge_type],style="bold")
            # g.save()
            g.render(filename = 'example',directory = directory,format = "png",view=False)  
        else:

            g = Digraph(name = filename,filename = filename,directory = directory,format = "png")
            for node in self.result_nodes:
                g.node(str(node.sorted_id),str(node.varible_name))
            
            i = 0
            for node in self.result_nodes:
                for index,childnode in enumerate(node.childrens):
                    i+=1
         
                    g.edge(tail_name = str(node.sorted_id),head_name = str(childnode.sorted_id),label=str(node.children_edges[index].func_name),color=self.colorset[node.children_edges[index].edge_type],style="bold")
            # g.save()
            g.render(filename = 'example',directory = directory,format = "png",view=False)  

 
    def point_dataset_node(self):
        node_index = 0
        while node_index < len(self.nodes):
            for index,edge in enumerate(self.nodes[node_index].children_edges):
                if 'read_csv' in edge.original_code:
                    self.nodes[node_index].is_dataset_node = True
                    self.nodes[node_index].children_edges[index].is_dataset_edge = True
                    self.nodes[node_index].children_edges[index].edge_type = 3
                    self.nodes[node_index].children_edges[index].func_name = 'read_csv'
                    for childeinx, childe in enumerate(self.edges):
                        if childe.edge_id == self.nodes[node_index].children_edges[index].edge_id:
                            self.edges[childeinx].is_dataset_edge = True
                            self.edges[childeinx].is_dataset_edge = 3
                            self.edges[childeinx].is_dataset_edge = 'read_csv'
            if self.nodes[node_index].is_dataset_node == True:
                for index,child in enumerate(self.nodes[node_index].childrens):
                    for childinx , childn in enumerate(self.nodes):
                        if childn.sorted_id == child.sorted_id:
                            self.nodes[childinx].is_dataset_node=True

                    for childeinx, childe in enumerate(self.edges):
                        if childe.edge_id == self.nodes[node_index].children_edges[index].edge_id:
                            self.edges[childeinx].is_dataset_edge = True
                
                    self.nodes[node_index].children_edges[index].is_dataset_edge = True
                    self.nodes[node_index].childrens[index].is_dataset_node =True
            node_index += 1
            
    def delete_nodes(self):
        def merge_node(subsets, found_index, found_child_index):
            for item in subsets[found_index]:
                if item not in subsets[found_child_index]:
                    subsets[found_child_index].append(item)
            subsets.pop(found_index)
            return subsets
        subsets = {}
        max_id = 0
        for node in self.nodes:
            found_index = 0
            if node.is_model_node == True:
                continue
            is_in = False
            for index, key in enumerate(subsets):
                subset = subsets[key]
                if node.sorted_id in subset:
                    is_in = True
                    found_index = key
                    break
            if not is_in:
                new_set = []
                new_set.append(node.sorted_id)
                subsets[max_id] = new_set
                found_index = max_id
                max_id += 1
            for child in node.childrens:
                if child.is_model_node==True:
                    continue
                is_in = False

                for index,key in enumerate(subsets):
                    subset = subsets[key]
                    if child.sorted_id in subset:
                        is_in = True
                        found_child_index = key
                        break
                if is_in:
                    if found_child_index == found_index:
                        continue
                    else:
                        subsets = merge_node(subsets, found_index, found_child_index)
                 
                        for index, key in enumerate(subsets):
                            subset = subsets[key]
               
                            if node.sorted_id in subset:
                                is_in = True
                                found_index = key
               
                                break
               
                else:
                    subsets[found_index].append(child.sorted_id)

        res = []
        res_edge = []
        for key in subsets:
            has_read_edge = False
            subset = subsets[key]
            for id_ in subset:
                for node in self.nodes:
                    if node.sorted_id == id_:
                        found_node = node
                        break
                for edge in found_node.children_edges:
                    if edge.edge_type ==3:
                        has_read_edge = True
                        break
                if has_read_edge:
                    break
            if has_read_edge:
                res.append(subset)
    
        for subset in res:
            for id_ in subset:
                is_in = False
                for node in self.result_nodes:
                    if id_ == node.sorted_id:
                        is_in = True
                        break
                if is_in:
                    continue
                for node in self.nodes:
                    if node.sorted_id == id_:
                        self.result_nodes.append(node)
                        break
        sorted_id_list = []
        for node in self.result_nodes:
            sorted_id_list.append(node.sorted_id)

        for node_index, node in enumerate(self.result_nodes):
            rem_index = []
            for index,edge in enumerate(node.children_edges):
                if node.childrens[index].sorted_id not in sorted_id_list:
                    rem_index.append(index)
                    continue
                self.result_edges.append(edge)
            r_index = len(rem_index)-1
            while r_index >= 0:
                index = rem_index[r_index]
                r_index -= 1
                self.result_nodes[node_index].children_edges.remove(self.result_nodes[node_index].children_edges[index])
                self.result_nodes[node_index].childrens.remove(self.result_nodes[node_index].childrens[index])
      

    def update_edges(self):
        for node_index, node in enumerate(self.result_nodes):
            for edge_index, edge in enumerate(node.children_edges):
                is_in = False
                for ope in self.white_opes:
                    if ope in edge.func_name:
                        is_in = True
                        break
                if not is_in and edge.func_name != '-Assign-':
                    self.result_nodes[node_index].children_edges[edge_index].edge_type = 0
                    self.result_nodes[node_index].children_edges[edge_index].func_name = '-BLACK-'
                    for e_index, res_edge in enumerate(self.result_edges):
                        if res_edge.edge_id == edge.edge_id:
                            self.result_edges[e_index].edge_type = 0
                            self.result_edges[e_index].func_name = '-BLACK-'
                            break

    def point_model_node(self):
        node_index = 0
        
        for node_index, node in enumerate(self.nodes):
            for index,edge in enumerate(self.nodes[node_index].children_edges):
                if '.predict' in edge.func_name or 'cross_val_score' in edge.func_name:
                    if '.predict' in edge.func_name:
                        model_name = edge.func_name.split(".")[0]
                        self.model_variable.append(model_name)
                    if 'cross_val_score' in edge.func_name:
                        mode_node = ast.parse(edge.func_name)
                        mode_node = mode_node.body[0].value
                        model_name = self.get_variable_from_node(mode_node)
                        self.model_variable.append(model_name)

                    self.nodes[node_index].is_model_node = True
                    self.nodes[node_index].children_edges[index].is_model_edge = True
                    self.nodes[node_index].children_edges[index].edge_type = 5
                    for childeinx, childe in enumerate(self.edges):
                        if childe.edge_id == self.nodes[node_index].children_edges[index].edge_id:
                            self.edges[childeinx].is_model_edge = True
                            self.edges[childeinx].edge_type = 5
    
        node_index = 0
        while node_index < len(self.nodes):
            if self.nodes[node_index].varible_name in self.model_variable:
                self.nodes[node_index].is_model_node = True

            if self.nodes[node_index].is_model_node == True:
                for index,child in enumerate(self.nodes[node_index].childrens):
                    for childinx , childn in enumerate(self.nodes):
                        if childn.sorted_id == child.sorted_id:
                            self.nodes[childinx].is_model_node=True
    
                    for childeinx, childe in enumerate(self.edges):
                        if childe.edge_id == self.nodes[node_index].children_edges[index].edge_id:
                            self.edges[childeinx].is_model_edge = True
                            self.edges[childeinx].edge_type = 5
        
                    self.nodes[node_index].children_edges[index].is_model_edge = True
                    self.nodes[node_index].childrens[index].is_model_node =True
            node_index += 1
      
    def get_white_edge_line(self):
        code_list = self.code.split("\n")
        for index,node in enumerate(self.result_nodes):
            for edge_index,edge in enumerate(self.result_nodes[index].children_edges):
                if self.result_nodes[index].children_edges[edge_index].edge_type == 1: # white
                    found_edge_index = 0
                    for res_edge_id, res_edge in enumerate(self.result_edges):
                        if res_edge.edge_id == edge.edge_id:
                            found_edge_index = res_edge_id
                    
                    for line_index,line in enumerate(code_list):
                        replaced_line = self.result_nodes[index].children_edges[edge_index].original_code[0:-1].replace('\'','"')
                        replaced_line2 = replaced_line.replace('(- 1)','-1')
                        replaced_line3 = replaced_line2.replace(' ','')
                        replaced_line4 = self.result_nodes[index].children_edges[edge_index].original_code[0:-1].replace(' ','')
                        replace_line = line.replace(" ",'')
  
                        if self.result_nodes[index].children_edges[edge_index].original_code[0:-1] in line or replaced_line2 in line or replaced_line3 in line or replaced_line4 in line or replaced_line3 in replace_line or replaced_line4 in replace_line:
                            self.result_nodes[index].children_edges[edge_index].line_id.append(line_index)

    def show_dataset_graph(self):
        for node in self.nodes:
            if node.is_dataset_node:
                node.show()
                for edge in node.children_edges:
                    edge.show()

    def extract_all_user_defined_function(self,ast_tree):
        funcdefs = [n for n in ast.walk(ast_tree) if isinstance(n, ast.FunctionDef)]
        self.defed_functions += self.flat([self.get_func_names_from_funcdef(f) for f in funcdefs])
        
    def get_func_names_from_funcdef(self, funcdef_node):
            vars_info = []
            for arg in funcdef_node.args.args:
                vars_info.append(
                    (arg.arg, arg),
                )
            return vars_info
    
    def flat(self, some_list):
        return [item for sublist in some_list for item in sublist]

    def get_variable_from_node(self, node): 
        if type(node).__name__ == 'list':
            result = []
            for one_node in node:
                self.start_get_varible = True
                self.visit(one_node)

                result += self.get_varible_results
                self.get_varible_results = []
                self.start_get_varible = False

            new_result = []
            for i in result:
                if i in new_result:
                    continue
                new_result.append(i)
            return new_result
        else:
            self.start_get_varible = True
            self.visit(node)
            result = self.get_varible_results
            self.get_varible_results = []
            self.start_get_varible = False
            new_result = []
            for i in result:
                if i in new_result:
                    continue
                new_result.append(i)
            return new_result
    def get_variable_from_node_insub(self, node): 
        if type(node).__name__ == 'list':
            result = []
            for one_node in node:
                self.start_get_varible_insub = True
                self.start_get_varible = True
                self.visit(one_node)

                result += self.get_varible_results
                self.get_varible_results = []
                self.start_get_varible = False
                self.start_get_varible_insub = False

            new_result = []
            for i in result:
                if i in new_result:
                    continue
                new_result.append(i)
            return new_result
        else:
            self.start_get_varible_insub = True
            self.start_get_varible = True
            self.visit(node)
    
            result = self.get_varible_results
            self.get_varible_results = []
            self.start_get_varible = False
            self.start_get_varible_insub = False
            new_result = []
            for i in result:
                if i in new_result:
                    continue
                new_result.append(i)
            return new_result

    def get_variable_from_node_with_sub(self, node): 
        if type(node).__name__ == 'list':
            result = []
            for one_node in node:
                self.start_get_varible = True
                self.start_get_varible_with_sub = True
                self.start_get_varible_insub = True
                self.visit(one_node)
                result += self.get_varible_results
                self.get_varible_results = []
                self.start_get_varible_with_sub = False
                self.start_get_varible_insub = False
                self.start_get_varible = False

            new_result = []
            for i in result:
                if i in new_result:
                    continue
                new_result.append(i)
            return new_result
        else:
            self.start_get_varible = True
            self.start_get_varible_with_sub = True
            self.start_get_varible_insub = True
            self.visit(node)
            result = self.get_varible_results
            self.get_varible_results = []
            self.start_get_varible_with_sub = False
            self.start_get_varible_insub = False
            self.start_get_varible = False
            new_result = []
            for i in result:
                if i in new_result:
                    continue
                new_result.append(i)
            return new_result

    def get_variable_flow(self, node):
        self.start_get_varible_flow = True
        self.visit(node)
        self.start_get_varible_flow = False
        self.current_flow_id = 0

    def found_func_name(self,node):
        if type(node) == ast.Name:
            return node.id
        elif type(node) == ast.Attribute:
            func_name = self.found_func_name(node.value)
            func_name += "." + node.attr
            return func_name
        elif type(node) == ast.Str:
            return node.s
        elif type(node) == ast.Subscript:
            return self.found_func_name(node.value)
        elif type(node) == ast.Call:
            return self.found_func_name(node.func)
        

    ############################ create edges
    def create_black_edge(self, node, graph_node=None, one_target_name=None, targets = None, varibles=None):
        if varibles==None:
            varibles = self.get_variable_from_node_insub(node)
        else:
            varibles += self.get_variable_from_node_insub(node)
        varibles = [item for item in varibles if item in self.data_varibles_to_current_node.keys()]
        if len(varibles) is None:
            varibles = self.data_varibles_to_current_node.keys()
            
        if targets is not None:
            new_node = Node(self.sorted_node_id, "-Blank-Result-")
            self.sorted_node_id += 1
            for var in varibles:
                var_node = self.data_varibles_to_current_node[var]
                new_edge = Edge(self.edge_id,self.sorted_edge_id,"-BLACK-",astunparse.unparse(node),0)
                var_node.add_edge(new_node, new_edge)
                self.edge_id+=1
                self.edges.append(new_edge)
            self.sorted_edge_id+=1
            self.nodes.append(new_node)
            for target_value in targets:
                target_new_node = Node(self.sorted_node_id, target_value)
                self.sorted_node_id += 1
                if target_value not in self.data_varibles_to_current_node:
                    self.data_varibles_to_current_node[target_value] = []
                
                self.data_varibles_to_current_node[target_value] = target_new_node
                target_new_edge = Edge(self.edge_id, self.sorted_edge_id, '-Assign-', astunparse.unparse(node), 2)
                new_node.add_edge(target_new_node,target_new_edge)
                self.edge_id+=1
                self.edges.append(target_new_edge)
                self.nodes.append(target_new_node)
            self.sorted_edge_id+=1
            return 
                
        if graph_node is not None:
            if one_target_name is None:
                new_node = Node(self.sorted_node_id, graph_node.varible_name)
                self.data_varibles_to_current_node[graph_node.varible_name] = new_node
                self.sorted_node_id += 1
            else:
                new_node= Node(self.sorted_node_id, one_target_name)
                self.data_varibles_to_current_node[one_target_name] = new_node
                self.sorted_node_id += 1
            self.nodes.append(new_node)
            new_edge = Edge(self.edge_id, self.sorted_edge_id,'-BLACK-', astunparse.unparse(node), 0)
            self.sorted_edge_id += 1
            self.edge_id += 1
            graph_node.add_edge(new_node, new_edge)

            self.edges.append(new_edge)
            
            return

        for varible in varibles:
            if varible not in self.data_varibles_to_current_node.keys():
                continue
            varible_node = self.data_varibles_to_current_node[varible]
            new_node = Node(self.sorted_node_id, varible)
            self.sorted_node_id += 1
            new_edge = Edge(self.edge_id, self.sorted_edge_id,'-BLACK-', astunparse.unparse(node), 0)
            self.sorted_edge_id += 1
            self.edge_id += 1
            varible_node.add_edge(new_node, new_edge)
            self.nodes.append(new_node)
            self.edges.append(new_edge)
            self.data_varibles_to_current_node[varible] = new_node

    def create_white_edge(self, node, func_name):
        varible_tup_varibles = self.get_variable_from_node_with_sub(node)
        varibles = []
        for var_tuple in varible_tup_varibles:
            if var_tuple[1] == False:
                varibles.append(var_tuple[0])
            else:
                self.should_create_black_edge=True
                return
        varibles = [item for item in varibles if item in self.data_varibles_to_current_node.keys()]
        if len(varibles) != 0:    
            func_var = self.get_variable_from_node(node.func)
            func_var = [item for item in func_var if item in self.data_varibles_to_current_node]
            if len(func_var) == 0:
                new_node = Node(self.sorted_node_id, '-RESULT-'+func_name)
            elif len(func_var) == 1:
                new_node = Node(self.sorted_node_id, func_var[0])
            else:
                self.should_create_black_edge = True
                return
            self.sorted_node_id += 1
            varibles = [item for item in varibles if item in self.data_varibles_to_current_node.keys()]
            for varible in varibles:
                varible_node = self.data_varibles_to_current_node[varible]
                new_edge = Edge(self.edge_id, self.sorted_edge_id,func_name, astunparse.unparse(node), 1)
                self.edge_id += 1
                varible_node.add_edge(new_node, new_edge)
                self.edges.append(new_edge)

            self.sorted_edge_id += 1
            self.white_edge_node = new_node
        else:
            self.should_create_black_edge = True

        
    def create_assign_edge(self, target_varibles, node, graph_node=None):
        if graph_node is not None:
            for target in target_varibles:
                if target not in self.data_varibles_to_current_node:
                    self.data_varibles_to_current_node[target] = []
                new_node = Node(self.sorted_node_id, target)
                self.sorted_node_id += 1
                new_edge = Edge(self.edge_id, self.sorted_edge_id,'-Assign-', astunparse.unparse(node), 2)
                self.sorted_edge_id += 1
                self.edge_id += 1
                graph_node.add_edge(new_node, new_edge)
                self.data_varibles_to_current_node[target] = new_node
                self.nodes.append(new_node)
                self.edges.append(new_edge)

    ############################ ast visit functions

    def visit_Expr(self, node): # is a stmt
        if self.start_get_varible:
            self.generic_visit(node)
        else:
            if isinstance(node.value, ast.Call):
                self.should_create_white_edge = True
            elif (isinstance(node.value, ast.Assign) or isinstance(node.value, ast.AugAssign) or isinstance(node.value, ast.AnnAssign)) and isinstance(node.value.value, ast.Call):
                self.should_create_white_edge = True
            else:
                self.should_create_white_edge = False
                self.should_create_black_edge = True
        
            self.visit(node.value)
            if self.white_edge_node is not None:
                self.nodes.append(self.white_edge_node)
                self.data_varibles_to_current_node[self.white_edge_node.varible_name] = self.white_edge_node
                self.white_edge_node = None
            if self.should_create_black_edge:
                self.create_black_edge(node)
                self.should_create_black_edge = False
            return node

    
    def visit_Name(self, node):
        if self.start_get_varible:
            if self.start_get_varible_with_sub:
                self.get_varible_results.append((node.id, self.is_subscript))
            else:
                self.get_varible_results.append(node.id)
        else:
            self.generic_visit(node)

    def visit_Subscript(self, node):
        if self.start_get_varible:
            if self.start_get_varible_insub:
                self.is_subscript = True
                self.generic_visit(node)
                self.is_subscript = False
            else:
                self.is_subscript = True
                self.visit(node.value)
                self.is_subscript = False
        else:
            self.generic_visit(node)
    

    def visit_Call(self, node):
        if self.start_get_varible:
            self.generic_visit(node)
        else:
            func = node.func
            try:
                func_name = self.found_func_name(func)
            except:
                func_name = '-BLACK-'
            args_varibles = []
            for arg_node in node.args:
                args_varible = self.get_variable_from_node(arg_node)
                for arg_var in args_varible:
                    if arg_var in self.data_varibles_to_current_node.keys():
                        args_varibles += args_varible
            if func_name in self.defed_functions or not self.should_create_white_edge or func_name=='-BLACK-':
                return
            else:
                self.create_white_edge(node, func_name)
                self.should_create_white_edge=False

    def assign_func(self, node, assign_type = 1):
        if self.start_get_varible:
            self.generic_visit(node)
        else:
            if assign_type == 1:
                target_varibles = self.get_variable_from_node(node.targets)
            else:
                target_varibles = self.get_variable_from_node(node.target)
    
            if self.start_get_varible_flow:
                value_varibles = self.get_variable_from_node(node.value)

                
                for target_varible in target_varibles:
                    if target_varible not in self.varible_flow.keys():
                        self.varible_flow[target_varible] = []
                    for value_varible in value_varibles:
                        if self.is_for:
                            self.varible_flow[target_varible].append((value_varible, self.current_flow_id, 'for', self.expr_id['for']))
                        elif self.is_if:
                            self.varible_flow[target_varible].append((value_varible, self.current_flow_id, 'if', self.expr_id['if']))
                        elif self.is_try:
                            self.varible_flow[target_varible].append((value_varible, self.current_flow_id, 'try', self.expr_id['try']))
                        elif self.is_with:
                            self.varible_flow[target_varible].append((value_varible, self.current_flow_id, 'with', self.expr_id['with']))
                        elif self.is_while:
                            self.varible_flow[target_varible].append((value_varible, self.current_flow_id, 'while', self.expr_id['while']))
                        else:
                            self.varible_flow[target_varible].append((value_varible, self.current_flow_id, 'expr', 0))
                        self.current_flow_id += 1 
                    if len(self.varible_flow[target_varible]) == 0:
                        if self.is_for:
                            self.varible_flow[target_varible].append(("-CREATE-WITHOUT-VARIBLES-", self.current_flow_id, 'for', self.expr_id['for']))
                        elif self.is_if:
                            self.varible_flow[target_varible].append(("-CREATE-WITHOUT-VARIBLES-", self.current_flow_id, 'if', self.expr_id['if']))
                        elif self.is_try:
                            self.varible_flow[target_varible].append(("-CREATE-WITHOUT-VARIBLES-", self.current_flow_id, 'try', self.expr_id['try']))
                        elif self.is_with:
                            self.varible_flow[target_varible].append(("-CREATE-WITHOUT-VARIBLES-", self.current_flow_id, 'with', self.expr_id['with']))
                        elif self.is_while:
                            self.varible_flow[target_varible].append(("-CREATE-WITHOUT-VARIBLES-", self.current_flow_id, 'while', self.expr_id['while']))
                        else:
                            self.varible_flow[target_varible].append(("-CREATE-WITHOUT-VARIBLES-", self.current_flow_id, 'expr', 0))
                        self.current_flow_id += 1
            else:                
                value_varibles = self.get_variable_from_node(node.value)
                value_varibles1 = value_varibles
                value_varibles = [item for item in value_varibles if item in self.data_varibles_to_current_node.keys()]

                if len(value_varibles) == 0:
                    for index, target_value in enumerate(target_varibles):
                        need_continue = False
                        if target_value in self.data_varibles_to_current_node.keys():
                            if assign_type == 1:
                                for tar in node.targets:
                                    tarvar = self.get_variable_from_node(tar)
                                    if target_value in tarvar and type(tar).__name__ == 'Subscript':
                                        self.create_black_edge(node=node,varibles=[target_value])
                                        need_continue = True
                                        break
                            else:
                                need_continue=True
                                self.create_black_edge(node=node,varibles=[target_value])
                        if not need_continue:
                            new_node = Node(self.sorted_node_id, target_value)
                            self.sorted_node_id += 1
       
                            self.data_varibles_to_current_node[target_value] = new_node
                            self.create_black_edge(node=node, graph_node = new_node)
                            self.nodes.append(new_node)
                else:
                    if isinstance(node.value, ast.Call):
                        self.should_create_white_edge = True
                        self.generic_visit(node)
                        
                        if self.white_edge_node is not None:
                            self.nodes.append(self.white_edge_node)
                            self.data_varibles_to_current_node[self.white_edge_node.varible_name] = self.white_edge_node
                            self.create_assign_edge(target_varibles, node=node, graph_node=self.white_edge_node)
                        else:
                            self.create_black_edge(node=node, targets = target_varibles)
                        self.white_edge_node = None
                        self.should_create_black_edge = False
                    elif isinstance(node.value, ast.Subscript):
                        assert len(value_varibles) == 1 or len(value_varibles) == 0 or len(target_varibles) == 1
                        self.create_black_edge(node=node, graph_node=self.data_varibles_to_current_node[value_varibles[0]], one_target_name=target_varibles[0])
                    else:
                        self.create_black_edge(node=node, targets = target_varibles)


    def visit_Assign(self, node): # create node
        self.assign_func(node,1)

    def visit_AnnAssign(self, node):
        self.assign_func(node,2)

    def visit_AugAssign(self, node):
        self.assign_func(node,3)

    ######################################Stmt need create black edge
    def visit_If(self, node):#is a stmt need create black edge
        if self.start_get_varible:
            self.generic_visit(node)
        else:
            if self.start_get_varible_flow:
                self.is_if = True
                self.generic_visit(node)
                self.expr_id['if'] += 1
                self.is_if = False
            else:
                self.create_black_edge(node=node)
                pass

    def visit_While(self, node):#is a stmt need create black edge
        if self.start_get_varible:
            self.generic_visit(node)
        else:
            if self.start_get_varible_flow:
                self.is_while = True
                self.generic_visit(node)
                self.expr_id['while'] += 1
                self.is_while = False
            else:
                self.create_black_edge(node=node)
                pass


    def visit_AsyncFor(self, node):#is a stmt need create black edge
        if self.start_get_varible:
            self.generic_visit(node)
        else:
            if self.start_get_varible_flow:
                self.is_for=True
                self.generic_visit(node)
                self.expr_id['for'] += 1
                self.is_for=False
            else:
                self.create_black_edge(node=node)
                pass    

    def visit_For(self, node):#is a stmt need create black edge

        '''
            for i in data:
                xxxxxx(i)
                x_m = data[i]
        '''
        if self.start_get_varible:
            self.generic_visit(node)
        else:
            if self.start_get_varible_flow:
                self.is_for=True
                self.generic_visit(node)
                self.expr_id['for'] += 1
                self.is_for=False
            else:
                self.create_black_edge(node=node)
                pass

    def visit_Import(self, node):#is a stmt need create gray edge
        if self.start_get_varible:
            self.generic_visit(node)
        else:
            if self.start_get_varible_flow:
                self.generic_visit(node)
            else:
                return

    def visit_ImportFrom(self, node):#is a stmt need create gray edge
        if self.start_get_varible:
            self.generic_visit(node)
        else:
            if self.start_get_varible_flow:
                self.generic_visit(node)
            else:
                return

    def visit_FunctionDef(self, node):#is a stmt need create black edge
        if self.start_get_varible:
            self.generic_visit(node)
        else:
            if self.start_get_varible_flow:
                self.generic_visit(node)
            else:
                self.create_black_edge(node=node)
                pass

    def visit_Try(self, node):#is a stmt need create black edge
        if self.start_get_varible:
            self.generic_visit(node)
        else:
            if self.start_get_varible_flow:
                self.is_try=True
                self.generic_visit(node)
                self.expr_id['try'] += 1
                self.is_try=False
            else:
                self.create_black_edge(node=node)
                pass

    def visit_With(self, node):#is a stmt need create black edge
        if self.start_get_varible:
            self.generic_visit(node)
        else:
            if self.start_get_varible_flow:
                self.is_with = True
                self.generic_visit(node)
                self.expr_id['with'] += 1
                self.is_with=False
            else:
                self.create_black_edge(node=node)
                pass

    def visit_AsyncWith(self, node):#is a stmt need create black edge
        if self.start_get_varible:
            self.generic_visit(node)
        else:
            if self.start_get_varible_flow:
                self.is_with = True
                self.generic_visit(node)
                self.expr_id['with'] += 1
                self.is_with=False
            else:
                self.create_black_edge(node=node)
                pass

    def visit_ClassDef(self, node):#is a stmt need create black edge
        if self.start_get_varible:
            self.generic_visit(node)
        else:
            if self.start_get_varible_flow:
                self.generic_visit(node)
            else:
                self.create_black_edge(node=node)
                pass
    def visit_AsyncFunctionDef(self, node):#is a stmt need create black edge
        if self.start_get_varible:
            self.generic_visit(node)
        else:
            if self.start_get_varible_flow:
                self.generic_visit(node)
            else:
                self.create_black_edge(node=node)
                pass
    def visit_Delete(self, node):#is a stmt need create black edge
        if self.start_get_varible:
            self.generic_visit(node)
        else:
            if self.start_get_varible_flow:
                self.generic_visit(node)
            else:
                self.create_black_edge(node=node)
                pass
    def visit_Raise(self, node):#is a stmt need create black edge
        if self.start_get_varible:
            self.generic_visit(node)
        else:
            if self.start_get_varible_flow:
                self.generic_visit(node)
            else:
                self.create_black_edge(node=node)
                pass

    def visit_Assert(self, node):#is a stmt need create black edge
        if self.start_get_varible:
            self.generic_visit(node)
        else:
            if self.start_get_varible_flow:
                self.generic_visit(node)
            else:
                self.create_black_edge(node=node)
                pass
    def visit_Return(self, node):#is a stmt need create black edge
        if self.start_get_varible:
            self.generic_visit(node)
        else:
            if self.start_get_varible_flow:
                self.generic_visit(node)
            else:
                self.create_black_edge(node=node)
                pass
    def visit_Pass(self, node):
        if self.start_get_varible:
            self.generic_visit(node)
        else:
            if self.start_get_varible_flow:
                self.generic_visit(node)
            else:
                self.create_black_edge(node=node)
                pass
    def visit_Global(self, node):
        if self.start_get_varible:
            self.generic_visit(node)
        else:
            if self.start_get_varible_flow:
                self.generic_visit(node)
            else:
                self.create_black_edge(node=node)
                pass
    def visit_Nonlocal(self, node):
        if self.start_get_varible:
            self.generic_visit(node)
        else:
            if self.start_get_varible_flow:
                self.generic_visit(node)
            else:
                self.create_black_edge(node=node)
                pass

def save_graph(filepath):
    
    graph = NotebookGraph()

    ### load origin HI-program
    res = graph.load_file(filepath)
    if res == 'error':
        return

    graph.build_graph()
    graph.delete_nodes()
    graph.update_edges()
    graph.get_white_edge_line()
  
    filename = filepath.split('/')[-1]
    filename = filename.split(".")[0]
    try:
        with open('haipipe/core/tmpdata/prenotebook_graph/'+filename+".pkl", 'wb') as f:
            pickle.dump(graph, f)
    except:
        pass
    return graph.is_highlighted

def profile_hipipe(notebook_id):
    if not os.path.exists('haipipe/core/tmpdata/prenotebook_graph'):
        os.mkdir('haipipe/core/tmpdata/prenotebook_graph')
    root_path = 'haipipe/core/tmpdata/prenotebook_code/'
    file_path = root_path + notebook_id + '.py'
    highlighted = save_graph(file_path)

    


    