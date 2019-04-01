# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 15:21:22 2019

@author: 赵焕
Step 1: read para from data
Step 2: build function
Step 3: solve by tensorflow
"""
test_output = open('sol1.txt','w')
test_output.close()

import numpy as np
from scipy import sparse as sp
import math
import tensorflow as tf
import time
import csv
import argparse



start_time = time.time()

tfco = tf.contrib.constrained_optimization

# soft constraint penalty parameters
penalty_block_pow_real_max = [2.0, 50.0] # MW. when converted to p.u., this is overline_sigma_p in the formulation
penalty_block_pow_real_coeff = [1000.0, 5000.0, 1000000.0] # USD/MW-h. when converted USD/p.u.-h this is lambda_p in the formulation
penalty_block_pow_imag_max = [2.0, 50.0] # MVar. when converted to p.u., this is overline_sigma_q in the formulation
penalty_block_pow_imag_coeff = [1000.0, 5000.0, 1000000.0] # USD/MVar-h. when converted USD/p.u.-h this is lambda_q in the formulation
penalty_block_pow_abs_max = [2.0, 50.0] # MVA. when converted to p.u., this is overline_sigma_s in the formulation
penalty_block_pow_abs_coeff = [1000.0, 5000.0, 1000000.0] # USD/MWA-h. when converted USD/p.u.-h this is lambda_s in the formulation

# weight on base case in objective
base_case_penalty_weight = 0.5 # dimensionless. corresponds to delta in the formulation

# tolerance on hard constraints
#hard_constr_tol = 0.0
hard_constr_tol = 1e-12


#case path case2  scenario_1
parser = argparse.ArgumentParser()

parser.add_argument('raw')
parser.add_argument('rop')
parser.add_argument('con')
parser.add_argument('inl')
parser.add_argument('tim')
parser.add_argument('sco')
parser.add_argument('net')

args = parser.parse_args()

raw, rop, con, inl = args.raw, args.rop, args.con, args.inl



#read data, using the data object in evaluation start
"""Data structures and read/write methods for input and output data file formats

Author: Jesse Holzer, jesse.holzer@pnnl.gov

Date: 2018-04-05

"""
# data.py
# module for input and output data
# including data structures
# and read and write functions


# init_defaults_in_unused_field = True # do this anyway - it is not too big
read_unused_fields = True
write_defaults_in_unused_fields = False
write_values_in_unused_fields = True

def parse_token(token, val_type, default=None):
    val = None
    if len(token) > 0:
        val = val_type(token)
    elif default is not None:
        val = val_type(default)
    else:
        try:
            print('required field missing data, token: %s, val_type: %s' % (token, val_type))
            raise Exception('empty field not allowed')
        except Exception as e:
            raise e
        #raise Exception('empty field not allowed')
    return val

def pad_row(row, new_row_len):

    try:
        if len(row) != new_row_len:
            if len(row) < new_row_len:
                print('missing field, row:')
                print(row)
                raise Exception('missing field not allowed')
            elif len(row) > new_row_len:
                row = remove_end_of_line_comment_from_row(row, '/')
                if len(row) > new_row_len:
                    print('extra field, row:')
                    print(row)
                    raise Exception('extra field not allowed')
        else:
            row = remove_end_of_line_comment_from_row(row, '/')
    except Exception as e:
        raise e
    return row
    '''
    row_len = len(row)
    row_len_diff = new_row_len - row_len
    row_new = row
    if row_len_diff > 0:
        row_new = row + row_len_diff * ['']
    return row_new
    '''

def check_row_missing_fields(row, row_len_expected):

    try:
        if len(row) < row_len_expected:
            print('missing field, row:')
            print(row)
            raise Exception('missing field not allowed')
    except Exception as e:
        raise e

def remove_end_of_line_comment_from_row(row, end_of_line_str):

    index = [r.find(end_of_line_str) for r in row]
    len_row = len(row)
    entries_with_end_of_line_strs = [i for i in range(len_row) if index[i] > -1]
    num_entries_with_end_of_line_strs = len(entries_with_end_of_line_strs)
    if num_entries_with_end_of_line_strs > 0:
        first_entry_with_end_of_line_str = min(entries_with_end_of_line_strs)
        len_row_new = first_entry_with_end_of_line_str + 1
        row_new = [row[i] for i in range(len_row_new)]
        row_new[len_row_new - 1] = remove_end_of_line_comment(row_new[len_row_new - 1], end_of_line_str)
    else:
        row_new = [r for r in row]
    return row_new

def remove_end_of_line_comment(token, end_of_line_str):
    
    token_new = token
    index = token_new.find(end_of_line_str)
    if index > -1:
        token_new = token_new[0:index]
    return token_new

class Data:
    '''In physical units, i.e. data convention, i.e. input and output data files'''

    def __init__(self):

        self.raw = Raw()
        self.rop = Rop()
        self.inl = Inl()
        self.con = Con()
        
class Raw:
    '''In physical units, i.e. data convention, i.e. input and output data files'''

    def __init__(self):

        self.case_identification = CaseIdentification()
        self.buses = {}
        self.loads = {}
        self.fixed_shunts = {}
        self.generators = {}
        self.nontransformer_branches = {}
        self.transformers = {}
        self.areas = {}
        self.switched_shunts = {}

    def set_areas_from_buses(self):
        
        area_i_set = set([b.area for b in self.buses.values()])
        def area_set_i(area, i):
            area.i = i
            return area
        self.areas = {i:area_set_i(Area(), i) for i in area_i_set}
        

    def switched_shunts_combine_blocks_steps(self):

        for r in self.switched_shunts.values():
            b_min = 0.0
            b_max = 0.0
            b1 = float(r.n1) * r.b1
            b2 = float(r.n2) * r.b2
            b3 = float(r.n3) * r.b3
            b4 = float(r.n4) * r.b4
            b5 = float(r.n5) * r.b5
            b6 = float(r.n6) * r.b6
            b7 = float(r.n7) * r.b7
            b8 = float(r.n8) * r.b8
            for b in [b1, b2, b3, b4, b5, b6, b7, b8]:
                if b > 0.0:
                    b_max += b
                elif b < 0.0:
                    b_min += b
                else:
                    break
            r.n1 = 0
            r.b1 = 0.0
            r.n2 = 0
            r.b2 = 0.0
            r.n3 = 0
            r.b3 = 0.0
            r.n4 = 0
            r.b4 = 0.0
            r.n5 = 0
            r.b5 = 0.0
            r.n6 = 0
            r.b6 = 0.0
            r.n7 = 0
            r.b7 = 0.0
            r.n8 = 0
            r.b8 = 0.0
            if b_max > 0.0:
                r.n1 = 1
                r.b1 = b_max
                if b_min < 0.0:
                    r.n2 = 1
                    r.b2 = b_min
            elif b_min < 0.0:
                r.n1 = 1
                r.b1 = b_min
        
    def set_operating_point_to_offline_solution(self):

        for r in self.buses.values():
            r.vm = 1.0
            r.va = 0.0
        for r in self.generators.values():
            r.pg = 0.0
            r.qg = 0.0
        for r in self.switched_shunts.values():
            r.binit = 0.0
        
    def read(self, file_name):

        with open(file_name, 'r') as in_file:
            lines = in_file.readlines()
        delimiter_str = ","
        quote_str = "'"
        skip_initial_space = True
        rows = csv.reader(
            lines,
            delimiter=delimiter_str,
            quotechar=quote_str,
            skipinitialspace=skip_initial_space)
        rows = [[t.strip() for t in r] for r in rows]
        self.read_from_rows(rows)
        self.set_areas_from_buses()
        
    def row_is_file_end(self, row):

        is_file_end = False
        if len(row) == 0:
            is_file_end = True
        if row[0][:1] in {'','q','Q'}:
            is_file_end = True
        return is_file_end
    
    def row_is_section_end(self, row):

        is_section_end = False
        if row[0][:1] == '0':
            is_section_end = True
        return is_section_end
        
    def read_from_rows(self, rows):

        row_num = 0
        cid_rows = rows[row_num:(row_num + 3)]
        self.case_identification.read_from_rows(rows)
        row_num += 2
        while True:
            row_num += 1
            row = rows[row_num]
            if self.row_is_file_end(row):
                return
            if self.row_is_section_end(row):
                break
            bus = Bus()
            bus.read_from_row(row)
            self.buses[bus.i] = bus
        while True:
            row_num += 1
            row = rows[row_num]
            if self.row_is_file_end(row):
                return
            if self.row_is_section_end(row):
                break
            load = Load()
            load.read_from_row(row)
            self.loads[(load.i, load.id)] = load
        while True:
            row_num += 1
            row = rows[row_num]
            if self.row_is_file_end(row):
                return
            if self.row_is_section_end(row):
                break
            fixed_shunt = FixedShunt()
            fixed_shunt.read_from_row(row)
            self.fixed_shunts[(fixed_shunt.i, fixed_shunt.id)] = fixed_shunt
        while True:
            row_num += 1
            row = rows[row_num]
            if self.row_is_file_end(row):
                return
            if self.row_is_section_end(row):
                break
            generator = Generator()
            generator.read_from_row(row)
            self.generators[(generator.i, generator.id)] = generator
        while True:
            row_num += 1
            row = rows[row_num]
            if self.row_is_file_end(row):
                return
            if self.row_is_section_end(row):
                break
            nontransformer_branch = NontransformerBranch()
            nontransformer_branch.read_from_row(row)
            self.nontransformer_branches[(
                nontransformer_branch.i,
                nontransformer_branch.j,
                nontransformer_branch.ckt)] = nontransformer_branch
        while True:
            row_num += 1
            row = rows[row_num]
            if self.row_is_file_end(row):
                return
            if self.row_is_section_end(row):
                break
            transformer = Transformer()
            num_rows = transformer.get_num_rows_from_row(row)
            rows_temp = rows[
                row_num:(row_num + num_rows)]
            transformer.read_from_rows(rows_temp)
            self.transformers[(
                transformer.i,
                transformer.j,
                #transformer.k,
                0,
                transformer.ckt)] = transformer
            row_num += (num_rows - 1)
        while True: # areas - for now just make a set of areas based on bus info
            row_num += 1
            row = rows[row_num]
            if self.row_is_file_end(row):
                return
            if self.row_is_section_end(row):
                break
            area = Area()
            area.read_from_row(row)
            self.areas[area.i] = area
        while True:
            row_num += 1
            row = rows[row_num]
            if self.row_is_file_end(row):
                return
            if self.row_is_section_end(row):
                break
        while True:
            row_num += 1
            row = rows[row_num]
            if self.row_is_file_end(row):
                return
            if self.row_is_section_end(row):
                break
        while True:
            row_num += 1
            row = rows[row_num]
            if self.row_is_file_end(row):
                return
            if self.row_is_section_end(row):
                break
        while True:
            row_num += 1
            row = rows[row_num]
            if self.row_is_file_end(row):
                return
            if self.row_is_section_end(row):
                break
        while True:
            row_num += 1
            row = rows[row_num]
            if self.row_is_file_end(row):
                return
            if self.row_is_section_end(row):
                break
        while True: # zone
            row_num += 1
            row = rows[row_num]
            if self.row_is_file_end(row):
                return
            if self.row_is_section_end(row):
                break
        while True:
            row_num += 1
            row = rows[row_num]
            if self.row_is_file_end(row):
                return
            if self.row_is_section_end(row):
                break
        while True:
            row_num += 1
            row = rows[row_num]
            if self.row_is_file_end(row):
                return
            if self.row_is_section_end(row):
                break
        while True:
            row_num += 1
            row = rows[row_num]
            if self.row_is_file_end(row):
                return
            if self.row_is_section_end(row):
                break
        while True:
            row_num += 1
            row = rows[row_num]
            if self.row_is_file_end(row):
                return
            if self.row_is_section_end(row):
                break
            switched_shunt = SwitchedShunt()
            switched_shunt.read_from_row(row)
            self.switched_shunts[switched_shunt.i] = switched_shunt

class Rop:
    '''In physical units, i.e. data convention, i.e. input and output data files'''

    def __init__(self):

        #self.generator_dispatch_records = GeneratorDispatchRecord() # needs to be a dictionary
        self.generator_dispatch_records = {}
        self.active_power_dispatch_records = {}
        self.piecewise_linear_cost_functions = {}
        
    def trancostfuncfrom_phase_0(self,rawdata):
        ds=self.active_power_dispatch_records.get((4, '1'))

        for r in rawdata.generators.values():
            ds=self.active_power_dispatch_records.get((r.i,r.id))
            #update piecewise linear info
            self.active_power_dispatch_records.get((r.i,r.id)).npairs=10
            self.active_power_dispatch_records.get((r.i,r.id)).costzero=self.active_power_dispatch_records.get((r.i,r.id)).constc
            for i in range(self.active_power_dispatch_records.get((r.i,r.id)).npairs):
                # the points will be power followed by cost, ie[power0 cost0 power 1 cost1 ....]
                self.active_power_dispatch_records.get((r.i,r.id)).points.append(r.pb+i*(r.pt-r.pb)/(self.active_power_dispatch_records.get((r.i,r.id)).npairs-1))
                self.active_power_dispatch_records.get((r.i,r.id)).points.append(self.active_power_dispatch_records.get((r.i,r.id)).constc+self.active_power_dispatch_records.get((r.i,r.id)).linearc*(r.pb+i*(r.pt-r.pb)/(self.active_power_dispatch_records.get((r.i,r.id)).npairs-1))+self.active_power_dispatch_records.get((r.i,r.id)).quadraticc*pow(r.pb+i*(r.pt-r.pb)/(self.active_power_dispatch_records.get((r.i,r.id)).npairs-1),2))
            
        

    def read_from_phase_0(self, file_name):
        
        '''takes the generator.csv file as input'''
        with open(file_name, 'r') as in_file:
            lines = in_file.readlines()
        del lines[0]
        delimiter_str = ","
        quote_str = "'"
        skip_initial_space = True
        rows = csv.reader(
            lines,
            delimiter=delimiter_str,
            quotechar=quote_str,
            skipinitialspace=skip_initial_space)
        #[[t.strip() for t in r] for r in rows]
        for r in rows:
            indi=1
            gen_dispatch = QuadraticCostFunctions()
            gen_dispatch.read_from_csv(r)
            
            gen_dispatch.read_from_csv_quadraticinfo(r)
            while (indi<4):
                r2=rows.next()
                gen_dispatch.read_from_csv_quadraticinfo(r2)
                indi=indi+1
                #print([gen_dispatch.bus,gen_dispatch.genid, gen_dispatch.constc,gen_dispatch.linearc,gen_dispatch.quadraticc])
            self.active_power_dispatch_records[gen_dispatch.bus,gen_dispatch.genid] = gen_dispatch
        #print([gen_dispatch.bus,gen_dispatch.genid, gen_dispatch.constc])
        #ds=self.active_power_dispatch_records.get((4, '1'))

        
    def read(self, file_name):

        with open(file_name, 'r') as in_file:
            lines = in_file.readlines()
        delimiter_str = ","
        quote_str = "'"
        skip_initial_space = True
        rows = csv.reader(
            lines,
            delimiter=delimiter_str,
            quotechar=quote_str,
            skipinitialspace=skip_initial_space)
        rows = [[t.strip() for t in r] for r in rows]
        self.read_from_rows(rows)
        
    def row_is_file_end(self, row):

        is_file_end = False
        if len(row) == 0:
            is_file_end = True
        if row[0][:1] in {'','q','Q'}:
            is_file_end = True
        return is_file_end
    
    def row_is_section_end(self, row):

        is_section_end = False
        if row[0][:1] == '0':
            is_section_end = True
        return is_section_end
        
    def read_from_rows(self, rows):

        row_num = -1
        while True:
            row_num += 1
            row = rows[row_num]
            if self.row_is_file_end(row):
                return
            if self.row_is_section_end(row):
                break
        while True:
            row_num += 1
            row = rows[row_num]
            if self.row_is_file_end(row):
                return
            if self.row_is_section_end(row):
                break
        while True:
            row_num += 1
            row = rows[row_num]
            if self.row_is_file_end(row):
                return
            if self.row_is_section_end(row):
                break
        while True:
            row_num += 1
            row = rows[row_num]
            if self.row_is_file_end(row):
                return
            if self.row_is_section_end(row):
                break
        while True:
            row_num += 1
            row = rows[row_num]
            if self.row_is_file_end(row):
                return
            if self.row_is_section_end(row):
                break
        while True:
            row_num += 1
            row = rows[row_num]
            if self.row_is_file_end(row):
                return
            if self.row_is_section_end(row):
                break
            generator_dispatch_record = GeneratorDispatchRecord()
            generator_dispatch_record.read_from_row(row)
            self.generator_dispatch_records[(
                generator_dispatch_record.bus,
                generator_dispatch_record.genid)] = generator_dispatch_record
        while True:
            row_num += 1
            row = rows[row_num]
            if self.row_is_file_end(row):
                return
            if self.row_is_section_end(row):
                break
            active_power_dispatch_record = ActivePowerDispatchRecord()
            active_power_dispatch_record.read_from_row(row)
            self.active_power_dispatch_records[
                active_power_dispatch_record.tbl] = (
                    active_power_dispatch_record)
        while True:
            row_num += 1
            row = rows[row_num]
            if self.row_is_file_end(row):
                return
            if self.row_is_section_end(row):
                break
        while True:
            row_num += 1
            row = rows[row_num]
            if self.row_is_file_end(row):
                return
            if self.row_is_section_end(row):
                break
        while True:
            row_num += 1
            row = rows[row_num]
            if self.row_is_file_end(row):
                return
            if self.row_is_section_end(row):
                break
        while True:
            row_num += 1
            row = rows[row_num]
            if self.row_is_file_end(row):
                return
            if self.row_is_section_end(row):
                break
            #piecewise_linear_cost_function = PiecewiseLinearCostFunction(ActivePowerDispatchRecord)
            piecewise_linear_cost_function = PiecewiseLinearCostFunction()
            num_rows = piecewise_linear_cost_function.get_num_rows_from_row(row)
            rows_temp = rows[
                row_num:(row_num + num_rows)]
            piecewise_linear_cost_function.read_from_rows(rows_temp)
            self.piecewise_linear_cost_functions[
                piecewise_linear_cost_function.ltbl] = (
                piecewise_linear_cost_function)
            row_num += (num_rows - 1)

class Inl:
    '''In physical units, i.e. data convention, i.e. input and output data files'''

    def __init__(self):

        self.generator_inl_records = {}

    # TODO
    def read_from_phase_0(self, file_name):
        '''takes the generator.csv file as input'''


    def read(self, file_name):

        with open(file_name, 'r') as in_file:
            lines = in_file.readlines()
        delimiter_str = ","
        quote_str = "'"
        skip_initial_space = True
        rows = csv.reader(
            lines,
            delimiter=delimiter_str,
            quotechar=quote_str,
            skipinitialspace=skip_initial_space)
        rows = [[t.strip() for t in r] for r in rows]
        self.read_from_rows(rows)
        
    def row_is_file_end(self, row):

        is_file_end = False
        if len(row) == 0:
            is_file_end = True
        if row[0][:1] in {'','q','Q'}:
            is_file_end = True
        return is_file_end
    
    def row_is_section_end(self, row):

        is_section_end = False
        if row[0][:1] == '0':
            is_section_end = True
        return is_section_end
        
    def read_from_rows(self, rows):

        row_num = -1
        while True:
            row_num += 1
            row = rows[row_num]
            if self.row_is_file_end(row):
                return
            if self.row_is_section_end(row):
                break
            generator_inl_record = GeneratorInlRecord()
            generator_inl_record.read_from_row(row)
            self.generator_inl_records[(
                generator_inl_record.i,
                generator_inl_record.id)] = generator_inl_record
        
class Con:
    '''In physical units, i.e. data convention, i.e. input and output data files'''

    def __init__(self):

        self.contingencies = {}

    def read_from_phase_0(self, file_name):
        '''takes the contingency.csv file as input'''
        with open(file_name, 'r') as in_file:
            lines = in_file.readlines()
        delimiter_str = " "
        quote_str = "'"
        skip_initial_space = True
        del lines[0]
        rows = csv.reader(
            lines,
            delimiter=delimiter_str,
            quotechar=quote_str,
            skipinitialspace=skip_initial_space)
        rows = [[t.strip() for t in r] for r in rows]
        quote_str = "'"
        contingency = Contingency()
        #there is no contingency label for continency.csv
        for r in rows:
            tmprow=r[0].split(',')
            if tmprow[1].upper()=='B' or tmprow[1].upper()=='T':
                contingency.label ="LINE-"+tmprow[2]+"-"+tmprow[3]+"-"+tmprow[4]
                branch_out_event = BranchOutEvent()
                branch_out_event.read_from_csv(tmprow)
                contingency.branch_out_events.append(branch_out_event)
                self.contingencies[contingency.label] = branch_out_event
            elif tmprow[1].upper()=='G':
                contingency.label = "GEN-"+tmprow[2]+"-"+tmprow[3]
                generator_out_event = GeneratorOutEvent()
                generator_out_event.read_from_csv(tmprow)
                contingency.generator_out_events.append(generator_out_event)
                self.contingency.generator_out_event.read_from_csv(tmprow)


    def read(self, file_name):

        with open(file_name, 'r') as in_file:
            lines = in_file.readlines()
        try:
            for l in lines:
                if l.find("'") > -1 or l.find('"') > -1:
                    print('no quotes allowed, line:')
                    print(l)
                    raise Exception('no quotes allowed in CON')
        except Exception as e:
            raise e
        delimiter_str = " "
        #quote_str = "'"
        skip_initial_space = True
        rows = csv.reader(
            lines,
            delimiter=delimiter_str,
            #quotechar=quote_str,
            skipinitialspace=skip_initial_space,
            quoting=csv.QUOTE_NONE) # QUOTE_NONE
        rows = [[t.strip() for t in r] for r in rows]
        self.read_from_rows(rows)
        
    def row_is_file_end(self, row):

        is_file_end = False
        if len(row) == 0:
            is_file_end = True
        if row[0][:1] in {'','q','Q'}:
            is_file_end = True
        return is_file_end
    
    #def row_is_section_end(self, row):
    #
    #    is_section_end = False
    #    if row[0][:1] == '0':
    #        is_section_end = True
    #    return is_section_end

    def is_contingency_start(self, row):

        return (row[0].upper() == 'CONTINGENCY')

    def is_end(self, row):

        return (row[0].upper() == 'END')

    def is_branch_out_event(self, row):

        #return (
        #    row[0].upper() in {'DISCONNECT', 'OPEN', 'TRIP'} and
        #    row[1].upper() in {'BRANCH', 'LINE'})
        return (row[0] == 'OPEN' and row[1] == 'BRANCH')

    def is_three_winding(self, row):

        #print(row)
        if len(row) < 9:
            return False
        elif row[8].upper() == 'TO':
            return True
        else:
            return False

    def is_generator_out_event(self, row):

        #return(
        #    row[0].upper() == 'REMOVE' and
        #    row[1].upper() in {'UNIT', 'MACHINE'})
        return(row[0] == 'REMOVE' and row[1] == 'UNIT')
        
    def read_from_rows(self, rows):

        row_num = -1
        in_contingency = False
        while True:
            row_num += 1
            #if row_num >= len(rows): # in case the data provider failed to put an end file line
            #    return
            row = rows[row_num]
            if self.row_is_file_end(row):
                return
            #if self.row_is_section_end(row):
            #    break
            elif self.is_contingency_start(row):
                in_contingency = True
                contingency = Contingency()
                contingency.label = row[1]
            elif self.is_end(row):
                if in_contingency:
                    self.contingencies[contingency.label] = contingency
                    in_contingency = False
                else:
                    break
            elif self.is_branch_out_event(row):
                branch_out_event = BranchOutEvent()
                if self.is_three_winding(row):
                    branch_out_event.read_three_winding_from_row(row)
                else:
                    branch_out_event.read_from_row(row)
                contingency.branch_out_events.append(branch_out_event)
            elif self.is_generator_out_event(row):
                generator_out_event = GeneratorOutEvent()
                generator_out_event.read_from_row(row)
                contingency.generator_out_events.append(generator_out_event)
            else:
                try:
                    print('format error in CON file row:')
                    print(row)
                    raise Exception('format error in CON file')
                except Exception as e:
                    raise e

class CaseIdentification:

    def __init__(self):

        self.ic = 0
        self.sbase = 100.0
        self.rev = 33
        self.xfrrat = 0
        self.nxfrat = 1
        self.basfrq = 60.0
        self.record_2 = 'Grid Optimization Competition'
        self.record_3 = 'RAW file. Other required input data files include ROP, INL, CON'

    def read_record_1_from_row(self, row):

        row = pad_row(row, 6)
        #row[5] = remove_end_of_line_comment(row[5], '/')
        self.sbase = parse_token(row[1], float, default=None)
        if read_unused_fields:
            self.ic = parse_token(row[0], int, 0)
            self.rev = parse_token(row[2], int, 33)
            self.xfrrat = parse_token(row[3], int, 0)
            self.nxfrat = parse_token(row[4], int, 1)
            self.basfrq = parse_token(row[5], float, 60.0) # need to remove end of line comment

    def read_from_rows(self, rows):

        self.read_record_1_from_row(rows[0])
        #self.record_2 = '' # not preserving these at this point
        #self.record_3 = '' # do that later

class Bus:

    def __init__(self):

        self.i = None # no default allowed - we want this to throw an error
        self.name = 12*' '
        self.baskv = 0.0
        self.ide = 1
        self.area = 1
        self.zone = 1
        self.owner = 1
        self.vm = 1.0
        self.va = 0.0
        self.nvhi = 1.1
        self.nvlo = 0.9
        self.evhi = 1.1
        self.evlo = 0.9

    def read_from_row(self, row):

        row = pad_row(row, 13)
        self.i = parse_token(row[0], int, default=None)
        self.area = parse_token(row[4], int, default=None)
        self.vm = parse_token(row[7], float, default=None)
        self.va = parse_token(row[8], float, default=None)
        self.nvhi = parse_token(row[9], float, default=None)
        self.nvlo = parse_token(row[10], float, default=None)
        self.evhi = parse_token(row[11], float, default=None)
        self.evlo = parse_token(row[12], float, default=None)
        if read_unused_fields:
            self.name = parse_token(row[1], str, 12*' ')
            self.baskv = parse_token(row[2], float, 0.0)
            self.ide = parse_token(row[3], int, 1)
            self.zone = parse_token(row[5], int, 1)
            self.owner = parse_token(row[6], int, 1)
    
class Load:

    def __init__(self):

        self.i = None # no default allowed - should be an error
        self.id = '1'
        self.status = 1
        self.area = 1 # default is area of bus self.i, but this is not available yet
        self.zone = 1
        self.pl = 0.0
        self.ql = 0.0
        self.ip = 0.0
        self.iq = 0.0
        self.yp = 0.0
        self.yq = 0.0
        self.owner = 1
        self.scale = 1
        self.intrpt = 0

    def read_from_row(self, row):

        row = pad_row(row, 14)
        self.i = parse_token(row[0], int, default=None)
        self.id = parse_token(row[1], str, default=None)
        self.status = parse_token(row[2], int, default=None)
        self.pl = parse_token(row[5], float, default=None)
        self.ql = parse_token(row[6], float, default=None)
        if read_unused_fields:
            self.area = parse_token(row[3], int, 1)
            self.zone = parse_token(row[4], int, 1)
            self.ip = parse_token(row[7], float, 0.0)
            self.iq = parse_token(row[8], float, 0.0)
            self.yp = parse_token(row[9], float, 0.0)
            self.yq = parse_token(row[10], float, 0.0)
            self.owner = parse_token(row[11], int, 1)
            self.scale = parse_token(row[12], int, 1)
            self.intrpt = parse_token(row[13], int, 0)

class FixedShunt:

    def __init__(self):

        self.i = None # no default allowed
        self.id = '1'
        self.status = 1
        self.gl = 0.0
        self.bl = 0.0

    def read_from_row(self, row):

        row = pad_row(row, 5)
        self.i = parse_token(row[0], int, default=None)
        self.id = parse_token(row[1], str, default=None)
        self.status = parse_token(row[2], int, default=None)
        self.gl = parse_token(row[3], float, default=None)
        self.bl = parse_token(row[4], float, default=None)
        if read_unused_fields:
            pass

class Generator:

    def __init__(self):
        self.i = None # no default allowed
        self.id = '1'
        self.pg = 0.0
        self.qg = 0.0
        self.qt = 9999.0
        self.qb = -9999.0
        self.vs = 1.0
        self.ireg = 0
        self.mbase = 100.0 # need to take default value for this from larger Raw class
        self.zr = 0.0
        self.zx = 1.0
        self.rt = 0.0
        self.xt = 0.0
        self.gtap = 1.0
        self.stat = 1
        self.rmpct = 100.0
        self.pt = 9999.0
        self.pb = -9999.0
        self.o1 = 1
        self.f1 = 1.0
        self.o2 = 0
        self.f2 = 1.0
        self.o3 = 0
        self.f3 = 1.0
        self.o4 = 0
        self.f4 = 1.0
        self.wmod = 0
        self.wpf = 1.0

    def read_from_row(self, row):

        row = pad_row(row, 28)
        self.i = parse_token(row[0], int, default=None)
        self.id = parse_token(row[1], str, default=None)
        self.pg = parse_token(row[2], float, default=None)
        self.qg = parse_token(row[3], float, default=None)
        self.qt = parse_token(row[4], float, default=None)
        self.qb = parse_token(row[5], float, default=None)
        self.stat = parse_token(row[14], int, default=None)
        self.pt = parse_token(row[16], float, default=None)
        self.pb = parse_token(row[17], float, default=None)
        if read_unused_fields:
            self.vs = parse_token(row[6], float, 1.0)
            self.ireg = parse_token(row[7], int, 0)
            self.mbase = parse_token(row[8], float, 100.0)
            self.zr = parse_token(row[9], float, 0.0)
            self.zx = parse_token(row[10], float, 1.0)
            self.rt = parse_token(row[11], float, 0.0)
            self.xt = parse_token(row[12], float, 0.0)
            self.gtap = parse_token(row[13], float, 1.0)
            self.rmpct = parse_token(row[15], float, 100.0)
            self.o1 = parse_token(row[18], int, 1)
            self.f1 = parse_token(row[19], float, 1.0)
            self.o2 = parse_token(row[20], int, 0)
            self.f2 = parse_token(row[21], float, 1.0)
            self.o3 = parse_token(row[22], int, 0)
            self.f3 = parse_token(row[23], float, 1.0)
            self.o4 = parse_token(row[24], int, 0)
            self.f4 = parse_token(row[25], float, 1.0)
            self.wmod = parse_token(row[26], int, 0)
            self.wpf = parse_token(row[27], float, 1.0)

class NontransformerBranch:

    def __init__(self):

        self.i = None # no default
        self.j = None # no default
        self.ckt = '1'
        self.r = None # no default
        self.x = None # no default
        self.b = 0.0
        self.ratea = 0.0
        self.rateb = 0.0
        self.ratec = 0.0
        self.gi = 0.0
        self.bi = 0.0
        self.gj = 0.0
        self.bj = 0.0
        self.st = 1
        self.met = 1
        self.len = 0.0
        self.o1 = 1
        self.f1 = 1.0
        self.o2 = 0
        self.f2 = 1.0
        self.o3 = 0
        self.f3 = 1.0
        self.o4 = 0
        self.f4 = 1.0

    def read_from_row(self, row):

        row = pad_row(row, 24)
        self.i = parse_token(row[0], int, default=None)
        self.j = parse_token(row[1], int, default=None)
        self.ckt = parse_token(row[2], str, default=None)
        self.r = parse_token(row[3], float, default=None)
        self.x = parse_token(row[4], float, default=None)
        self.b = parse_token(row[5], float, default=None)
        self.ratea = parse_token(row[6], float, default=None)
        self.ratec = parse_token(row[8], float, default=None)
        self.st = parse_token(row[13], int, default=None)
        if read_unused_fields:
            self.rateb = parse_token(row[7], float, 0.0)
            self.gi = parse_token(row[9], float, 0.0)
            self.bi = parse_token(row[10], float, 0.0)
            self.gj = parse_token(row[11], float, 0.0)
            self.bj = parse_token(row[12], float, 0.0)
            self.met = parse_token(row[14], int, 1)
            self.len = parse_token(row[15], float, 0.0)
            self.o1 = parse_token(row[16], int, 1)
            self.f1 = parse_token(row[17], float, 1.0)
            self.o2 = parse_token(row[18], int, 0)
            self.f2 = parse_token(row[19], float, 1.0)
            self.o3 = parse_token(row[20], int, 0)
            self.f3 = parse_token(row[21], float, 1.0)
            self.o4 = parse_token(row[22], int, 0)
            self.f4 = parse_token(row[23], float, 1.0)

class Transformer:

    def __init__(self):

        self.i = None # no default
        self.j = None # no default
        self.k = 0
        self.ckt = '1'
        self.cw = 1
        self.cz = 1
        self.cm = 1
        self.mag1 = 0.0
        self.mag2 = 0.0
        self.nmetr = 2
        self.name = 12*' '
        self.stat = 1
        self.o1 = 1
        self.f1 = 1.0
        self.o2 = 0
        self.f2 = 1.0
        self.o3 = 0
        self.f3 = 1.0
        self.o4 = 0
        self.f4 = 1.0
        self.vecgrp = 12*' '
        self.r12 = 0.0
        self.x12 = None # no default allowed
        self.sbase12 = 100.0
        self.windv1 = 1.0
        self.nomv1 = 0.0
        self.ang1 = 0.0
        self.rata1 = 0.0
        self.ratb1 = 0.0
        self.ratc1 = 0.0
        self.cod1 = 0
        self.cont1 = 0
        self.rma1 = 1.1
        self.rmi1 = 0.9
        self.vma1 = 1.1
        self.vmi1 = 0.9
        self.ntp1 = 33
        self.tab1 = 0
        self.cr1 = 0.0
        self.cx1 = 0.0
        self.cnxa1 = 0.0
        self.windv2 = 1.0
        self.nomv2 = 0.0

    @property
    def num_windings(self):

        num_windings = 0
        if self.k is None:
            num_windings = 0
        elif self.k == 0:
            num_windings = 2
        else:
            num_windings = 3
        return num_windings
    
    def get_num_rows_from_row(self, row):

        num_rows = 0
        k = parse_token(row[2], int, 0)
        if k == 0:
            num_rows = 4
        else:
            num_rows = 5
        return num_rows

    def read_from_rows(self, rows):

        full_rows = self.pad_rows(rows)
        row = self.flatten_rows(full_rows)
        try:
            self.read_from_row(row)
        except Exception as e:
            print("row:")
            print(row)
            raise e
        
    def pad_rows(self, rows):

        return rows
        '''
        rows_new = rows
        if len(rows_new) == 4:
            rows_new.append([])
        rows_len = [len(r) for r in rows_new]
        rows_len_new = [21, 11, 17, 17, 17]
        rows_len_increase = [rows_len_new[i] - rows_len[i] for i in range(5)]
        # check no negatives in increase
        rows_new = [rows_new[i] + rows_len_increase[i]*[''] for i in range(5)]
        return rows_new
        '''

    def flatten_rows(self, rows):

        row = [t for r in rows for t in r]
        return row
    
    def read_from_row(self, row):       
        # just 2-winding, 4-row
        try:
            if len(row) != 43:
                if len(row) < 43:
                    raise Exception('missing field not allowed')
                elif len(row) > 43:
                    row = remove_end_of_line_comment_from_row(row, '/')
                    if len(row) > new_row_len:
                        raise Exception('extra field not allowed')
        except Exception as e:
            raise e
        self.i = parse_token(row[0], int, default=None)
        self.j = parse_token(row[1], int, default=None)
        self.ckt = parse_token(row[3], str, default=None)
        self.mag1 = parse_token(row[7], float, default=None)
        self.mag2 = parse_token(row[8], float, default=None)
        self.stat = parse_token(row[11], int, default=None)
        self.r12 = parse_token(row[21], float, default=None)
        self.x12 = parse_token(row[22], float, default=None)
        self.windv1 = parse_token(row[24], float, default=None)
        self.ang1 = parse_token(row[26], float, default=None)
        self.rata1 = parse_token(row[27], float, default=None)
        self.ratc1 = parse_token(row[29], float, default=None)
        self.windv2 = parse_token(row[41], float, default=None)
        if read_unused_fields:
            self.k = parse_token(row[2], int, 0)
            self.cw = parse_token(row[4], int, 1)
            self.cz = parse_token(row[5], int, 1)
            self.cm = parse_token(row[6], int, 1)
            self.nmetr = parse_token(row[9], int, 2)
            self.name = parse_token(row[10], str, 12*' ')
            self.o1 = parse_token(row[12], int, 1)
            self.f1 = parse_token(row[13], float, 1.0)
            self.o2 = parse_token(row[14], int, 0)
            self.f2 = parse_token(row[15], float, 1.0)
            self.o3 = parse_token(row[16], int, 0)
            self.f3 = parse_token(row[17], float, 1.0)
            self.o4 = parse_token(row[18], int, 0)
            self.f4 = parse_token(row[19], float, 1.0)
            self.vecgrp = parse_token(row[20], str, 12*' ')
            self.sbase12 = parse_token(row[23], float, 0.0)
            self.nomv1 = parse_token(row[25], float, 0.0)
            self.ratb1 = parse_token(row[28], float, 0.0)
            self.cod1 = parse_token(row[30], int, 0)
            self.cont1 = parse_token(row[31], int, 0)
            self.rma1 = parse_token(row[32], float, 1.1)
            self.rmi1 = parse_token(row[33], float, 0.9)
            self.vma1 = parse_token(row[34], float, 1.1)
            self.vmi1 = parse_token(row[35], float, 0.9)
            self.ntp1 = parse_token(row[36], int, 33)
            self.tab1 = parse_token(row[37], int, 0)
            self.cr1 = parse_token(row[38], float, 0.0)
            self.cx1 = parse_token(row[39], float, 0.0)
            self.cnxa1 = parse_token(row[40], float, 0.0)
            self.nomv2 = parse_token(row[42], float, 0.0)

class Area:

    def __init__(self):

        self.i = None # no default
        self.isw = 0
        self.pdes = 0.0
        self.ptol = 10.0
        self.arname = 12*' '

    def read_from_row(self, row):

        row = pad_row(row, 5)
        self.i = parse_token(row[0], int, default=None)
        if read_unused_fields:
            self.isw = parse_token(row[1], int, 0)
            self.pdes = parse_token(row[2], float, 0.0)
            self.ptol = parse_token(row[3], float, 10.0)
            self.arname = parse_token(row[4], str, 12*' ')

class Zone:

    def __init__(self):

        self.i = None # no default
        self.zoname = 12*' '
        
    def read_from_row(self, row):

        row = pad_row(row, 2)
        self.i = parse_token(row[0], int, default=None)
        if read_unused_fields:
            self.zoname = parse_token(row[1], str, 12*' ')

class SwitchedShunt:

    def __init__(self):

        self.i = None # no default
        self.modsw = 1
        self.adjm = 0
        self.stat = 1
        self.vswhi = 1.0
        self.vswlo = 1.0
        self.swrem = 0
        self.rmpct = 100.0
        self.rmidnt = 12*' '
        self.binit = 0.0
        self.n1 = 0
        self.b1 = 0.0
        self.n2 = 0
        self.b2 = 0.0
        self.n3 = 0
        self.b3 = 0.0
        self.n4 = 0
        self.b4 = 0.0
        self.n5 = 0
        self.b5 = 0.0
        self.n6 = 0
        self.b6 = 0.0
        self.n7 = 0
        self.b7 = 0.0
        self.n8 = 0
        self.b8 = 0.0

    def read_from_row(self, row):

        row = pad_row(row, 26)
        self.i = parse_token(row[0], int, default=None)
        self.stat = parse_token(row[3], int, default=None)
        self.binit = parse_token(row[9], float, default=None)
        self.n1 = parse_token(row[10], int, default=None)
        self.b1 = parse_token(row[11], float, default=None)
        self.n2 = parse_token(row[12], int, default=None)
        self.b2 = parse_token(row[13], float, default=None)
        self.n3 = parse_token(row[14], int, default=None)
        self.b3 = parse_token(row[15], float, default=None)
        self.n4 = parse_token(row[16], int, default=None)
        self.b4 = parse_token(row[17], float, default=None)
        self.n5 = parse_token(row[18], int, default=None)
        self.b5 = parse_token(row[19], float, default=None)
        self.n6 = parse_token(row[20], int, default=None)
        self.b6 = parse_token(row[21], float, default=None)
        self.n7 = parse_token(row[22], int, default=None)
        self.b7 = parse_token(row[23], float, default=None)
        self.n8 = parse_token(row[24], int, default=None)
        self.b8 = parse_token(row[25], float, default=None)
        if read_unused_fields:
            self.modsw = parse_token(row[1], int, 1)
            self.adjm = parse_token(row[2], int, 0)
            self.vswhi = parse_token(row[4], float, 1.0)
            self.vswlo = parse_token(row[5], float, 1.0)
            self.swrem = parse_token(row[6], int, 0)
            self.rmpct = parse_token(row[7], float, 100.0)
            self.rmidnt = parse_token(row[8], str, 12*' ')
        
class GeneratorDispatchRecord:

    def __init__(self):

        self.bus = None
        self.genid = None
        #self.disp = None
        self.dsptbl = None

    def read_from_row(self, row):

        row = pad_row(row, 4)
        self.bus = parse_token(row[0], int, default=None)
        self.genid = parse_token(row[1], str, default=None).strip()
        #self.disp = parse_token(row[2], float, 1.0)
        self.dsptbl = parse_token(row[3], int, default=None)

    def read_from_csv(self, row):
        self.bus = parse_token(row[0], int, default=None)
        self.genid = parse_token(row[1], str, default=None).strip()
        
class ActivePowerDispatchRecord:

    def __init__(self):

        self.tbl = None
        #self.pmax = None
        #self.pmin = None
        #self.fuelcost = None
        #self.ctyp = None
        #self.status = None
        self.ctbl = None

    def read_from_row(self, row):

        row = pad_row(row, 7)
        self.tbl = parse_token(row[0], int, default=None)
        #self.pmax = parse_token(row[1], float, 9999.0)
        #self.pmin = parse_token(row[2], float, -9999.0)
        #self.fuelcost = parse_token(row[3], float, 1.0)
        #self.ctyp = parse_token(row[4], int, 1)
        #self.status = parse_token(row[5], int, 1)
        self.ctbl = parse_token(row[6], int, default=None)

class PiecewiseLinearCostFunction():

    def __init__(self):

        self.ltbl = None
        #self.label = None
        #self.costzero = None
        self.npairs = None
        self.points = []

    def read_from_row(self, row):

        self.ltbl = parse_token(row[0], int, default=None)
        #self.label = parse_token(row[1], str, '')
        self.npairs = parse_token(row[2], int, default=None)
        for i in range(self.npairs):
            point = Point()
            point.read_from_row(
                row[(3 + 2*i):(5 + 2*i)])
            self.points.append(point)

    def get_num_rows_from_row(self, row):

        num_rows = parse_token(row[2], int, 0) + 1
        return num_rows

    def flatten_rows(self, rows):

        row = [t for r in rows for t in r]
        return row

    def read_from_rows(self, rows):

        self.read_from_row(self.flatten_rows(rows))
    
class  QuadraticCostFunctions(GeneratorDispatchRecord,PiecewiseLinearCostFunction):
    def __init__(self):
        GeneratorDispatchRecord.__init__(self)
        PiecewiseLinearCostFunction.__init__(self)
        self.constc = None
        self.linearc = None
        self.quadraticc = None
        self.powerfactor = None

    def read_from_csv_quadraticinfo(self, row):
        if parse_token(row[2], int, '')==0:
            self.constc = parse_token(row[3], float, 0.0)
        elif parse_token(row[2], int, '')==1:
            self.linearc =  parse_token(row[3], float, 0.0)
        elif parse_token(row[2], int, '')==2:    
            self.quadraticc =  parse_token(row[3], float, 0.0)
        elif parse_token(row[2], int, '')==9: 
            self.powerfactor =  parse_token(row[3], float, 0.0)

class GeneratorInlRecord:

    def __init__(self):

        self.i = None
        self.id = None
        #self.h = None
        #self.pmax = None
        #self.pmin = None
        self.r = None
        #self.d = None

    def read_from_row(self, row):

        row = pad_row(row, 7)
        self.i = parse_token(row[0], int, '')
        self.id = parse_token(row[1], str, '1')
        #self.h = parse_token(row[2], float, 4.0)
        #self.pmax = parse_token(row[3], float, 1.0)
        #self.pmin = parse_token(row[4], float, 0.0)
        self.r = parse_token(row[5], float, default=None)
        #self.d = parse_token(row[6], float, 0.0)
        
class Contingency:

    def __init__(self):

        self.label = ''
        self.branch_out_events = []
        self.generator_out_events = []

class Point:

    def __init__(self):

        self.x = None
        self.y = None

    def read_from_row(self, row):

        row = pad_row(row, 2)
        self.x = parse_token(row[0], float, default=None)
        self.y = parse_token(row[1], float, default=None)

class BranchOutEvent:

    def __init__(self):

        self.i = None
        self.j = None
        self.ckt = None

    def read_from_row(self, row):

        check_row_missing_fields(row, 10)
        self.i = parse_token(row[4], int, default=None)
        self.j = parse_token(row[7], int, default=None)
        self.ckt = parse_token(row[9], str, default=None)

    def read_from_csv(self, row):

        self.i = parse_token(row[2], int, '')
        self.j = parse_token(row[3], int, '')
        self.ckt = parse_token(row[4], str, '1')

    '''
    def read_three_winding_from_row(self, row):

        row = pad_row(row, 13)
        self.i = parse_token(row[4], int, '')
        self.j = parse_token(row[7], int, '')
        self.k = parse_token(row[10], int, '')
        self.ckt = parse_token(row[12], str, '1')
    '''

class GeneratorOutEvent:

    def __init__(self):

        self.i = None
        self.id = None

    def read_from_csv(self, row):

        self.i = parse_token(row[2], int, '')
        self.id = parse_token(row[3], str, '')

    def read_from_row(self, row):

        self.i = parse_token(row[5], int, default=None)
        self.id = parse_token(row[2], str, default=None)
    

'''
=================================
my part start
=================================
'''

p = Data()
p.raw.read(raw)
p.rop.read(rop)
p.con.read(con)
p.inl.read(inl)

'''
set all data start
'''

#set_data_scalars
base_mva = p.raw.case_identification.sbase


#set_data_bus_params
buses = list(p.raw.buses.values())
num_bus = len(buses)
bus_i = [r.i for r in buses]
bus_map = {bus_i[i]:i for i in range(len(bus_i))}
bus_volt_mag_max = np.array([r.nvhi for r in buses]) #cons
bus_volt_mag_min = np.array([r.nvlo for r in buses]) #cons
ctg_bus_volt_mag_max = np.array([r.evhi for r in buses]) #cons
ctg_bus_volt_mag_min = np.array([r.evlo for r in buses]) #cons
#transpose
bus_volt_mag_max = np.transpose([bus_volt_mag_max])
bus_volt_mag_min = np.transpose([bus_volt_mag_min])
ctg_bus_volt_mag_max = np.transpose([ctg_bus_volt_mag_max])
ctg_bus_volt_mag_min = np.transpose([ctg_bus_volt_mag_min])

areas = [r.area for r in buses]
num_area = len(areas)
area_i = [i for i in areas]
area_map = dict(zip(area_i, range(num_area)))

bus_area = [area_map[r.area] for r in buses]


#set_data_load_params
loads = list(p.raw.loads.values())
num_load = len(loads)
load_i = [r.i for r in loads]
load_id = [r.id for r in loads]
load_bus = [bus_map[load_i[i]] for i in range(num_load)]
load_map = {(load_i[i], load_id[i]):i for i in range(num_load)}
load_status = np.array([r.status for r in loads])
load_const_pow_real = np.array([r.pl / base_mva for r in loads]) * load_status
load_const_pow_imag = np.array([r.ql / base_mva for r in loads]) * load_status
bus_load_matrix = sp.csc_matrix(
    ([1.0 for i in range(num_load)],
     (load_bus, list(range(num_load)))),
    (num_bus, num_load)).astype(np.float32).toarray()
bus_load_const_pow_real = bus_load_matrix.dot(load_const_pow_real)
bus_load_const_pow_imag = bus_load_matrix.dot(load_const_pow_imag)
#transpose
bus_load_const_pow_real = np.transpose([bus_load_const_pow_real])
bus_load_const_pow_imag = np.transpose([bus_load_const_pow_imag])

#set_data_fxsh_params fix shunt
fxshs = list(p.raw.fixed_shunts.values())
num_fxsh = len(fxshs)
fxsh_i = [r.i for r in fxshs]
fxsh_id = [r.id for r in fxshs]
fxsh_bus = [bus_map[fxsh_i[i]] for i in range(num_fxsh)]
fxsh_map = {(fxsh_i[i], fxsh_id[i]):i for i in range(num_fxsh)}
fxsh_status = np.array([r.status for r in fxshs])
fxsh_adm_real = np.array([r.gl / base_mva for r in fxshs]) * fxsh_status
fxsh_adm_imag = np.array([r.bl / base_mva for r in fxshs]) * fxsh_status
bus_fxsh_matrix = sp.csc_matrix(
    ([1.0 for i in range(num_fxsh)],
     (fxsh_bus, list(range(num_fxsh)))),
    (num_bus, num_fxsh)).astype(np.float32).toarray()
bus_fxsh_adm_real = bus_fxsh_matrix.dot(fxsh_adm_real)
bus_fxsh_adm_real = np.transpose([bus_fxsh_adm_real])
bus_fxsh_adm_imag = bus_fxsh_matrix.dot(fxsh_adm_imag)
bus_fxsh_adm_imag = np.transpose([bus_fxsh_adm_imag])


#set_data_gen_params
gens = list(p.raw.generators.values())
gen_key = [(r.i, r.id) for r in gens]
num_gen = len(gens)
gen_i = [r.i for r in gens]
gen_id = [r.id for r in gens]
gen_bus = [bus_map[gen_i[i]] for i in range(num_gen)]
gen_map = {(gen_i[i], gen_id[i]):i for i in range(num_gen)}
gen_status = np.array([r.stat for r in gens])
gen_pow_imag_max = np.array([r.qt / base_mva for r in gens]) * gen_status #cons
gen_pow_imag_min = np.array([r.qb / base_mva for r in gens]) * gen_status #cons
gen_pow_real_max = np.array([r.pt / base_mva for r in gens]) * gen_status #cons
gen_pow_real_min = np.array([r.pb / base_mva for r in gens]) * gen_status #cons
gen_part_fact = {(r.i, r.id) : r.r for r in p.inl.generator_inl_records.values()}
gen_part_fact = np.array([gen_part_fact[(r.i, r.id)] for r in gens]) * gen_status
bus_gen_matrix = sp.csc_matrix(
    ([1.0 for i in range(num_gen)],
     (gen_bus, list(range(num_gen)))),
    (num_bus, num_gen)).astype(np.float32).toarray()
#transpose
gen_pow_imag_max = np.transpose([gen_pow_imag_max])
gen_pow_imag_min = np.transpose([gen_pow_imag_min])
gen_pow_real_max = np.transpose([gen_pow_real_max])
gen_pow_real_min = np.transpose([gen_pow_real_min])


    
gen_area = [bus_area[r] for r in gen_bus]
area_gens = [set() for a in range(num_area)]
for i in range(num_gen):
    area_gens[gen_area[i]].add(i)
gen_out_of_service = [
    i for i in range(num_gen)
    if gen_status[i] == 0.0]
print('num gen in service: %u, out of service: %u' % (num_gen - len(gen_out_of_service), len(gen_out_of_service)))


#set_data_line_params
lines = list(p.raw.nontransformer_branches.values())
line_key = [(r.i, r.j, r.ckt) for r in lines]
num_line = len(lines)
line_i = [r.i for r in lines]
line_j = [r.j for r in lines]
line_ckt = [r.ckt for r in lines]
line_orig_bus = [bus_map[line_i[i]] for i in range(num_line)]
line_dest_bus = [bus_map[line_j[i]] for i in range(num_line)]
line_map = {(line_i[i], line_j[i], line_ckt[i]):i for i in range(num_line)}
line_status = np.array([r.st for r in lines])
line_adm_real = np.array([r.r / (r.r**2.0 + r.x**2.0) for r in lines]) * line_status
line_adm_imag = np.array([-r.x / (r.r**2.0 + r.x**2.0) for r in lines]) * line_status
line_adm_ch_imag = np.array([r.b for r in lines]) * line_status
line_adm_total_imag = line_adm_imag + 0.5 * line_adm_ch_imag
line_curr_mag_max = np.array([r.ratea / base_mva for r in lines]) # todo - normalize by bus base kv???
ctg_line_curr_mag_max = np.array([r.ratec / base_mva for r in lines]) # todo - normalize by bus base kv???
bus_line_orig_matrix = sp.csc_matrix(
    ([1.0 for i in range(num_line)],
     (line_orig_bus, list(range(num_line)))),
    (num_bus, num_line)).astype(np.float32).toarray()
bus_line_dest_matrix = sp.csc_matrix(
    ([1.0 for i in range(num_line)],
     (line_dest_bus, list(range(num_line)))),
    (num_bus, num_line)).astype(np.float32).toarray()    
line_adm_real = np.transpose([line_adm_real.astype(np.float32)])
line_adm_imag = np.transpose([line_adm_imag.astype(np.float32)])
line_adm_total_imag = np.transpose([line_adm_total_imag])
line_curr_mag_max = np.transpose([line_curr_mag_max])


#set_data_xfmr_params transformers
xfmrs = list(p.raw.transformers.values())
xfmr_key = [(r.i, r.j, r.ckt) for r in xfmrs]
num_xfmr = len(xfmrs)
xfmr_i = [r.i for r in xfmrs]
xfmr_j = [r.j for r in xfmrs]
xfmr_ckt = [r.ckt for r in xfmrs]
xfmr_orig_bus = [bus_map[xfmr_i[i]] for i in range(num_xfmr)]
xfmr_dest_bus = [bus_map[xfmr_j[i]] for i in range(num_xfmr)]
xfmr_map = {(xfmr_i[i], xfmr_j[i], xfmr_ckt[i]):i for i in range(num_xfmr)}
xfmr_status = np.array([r.stat for r in xfmrs])
xfmr_adm_real = np.array([r.r12 / (r.r12**2.0 + r.x12**2.0) for r in xfmrs]) * xfmr_status
xfmr_adm_real = np.transpose([xfmr_adm_real])
xfmr_adm_imag = np.array([-r.x12 / (r.r12**2.0 + r.x12**2.0) for r in xfmrs]) * xfmr_status
xfmr_adm_imag = np.transpose([xfmr_adm_imag])
xfmr_adm_mag_real = np.array([r.mag1 for r in xfmrs]) * xfmr_status # todo normalize?
xfmr_adm_mag_real = np.transpose([xfmr_adm_mag_real])
xfmr_adm_mag_imag = np.array([r.mag2 for r in xfmrs]) * xfmr_status # todo normalize?
xfmr_adm_mag_imag = np.transpose([xfmr_adm_mag_imag])
xfmr_tap_mag = np.array([(r.windv1 / r.windv2) if r.stat else 1.0 for r in xfmrs]) # note status field is used here
xfmr_tap_mag = np.transpose([xfmr_tap_mag])
xfmr_tap_ang = np.array([r.ang1 * math.pi / 180.0 for r in xfmrs]) * xfmr_status
xfmr_tap_ang = np.transpose([xfmr_tap_ang])
xfmr_pow_mag_max = np.array([r.rata1 / base_mva for r in xfmrs]) # todo check normalization
xfmr_pow_mag_max = np.transpose([xfmr_pow_mag_max])
ctg_xfmr_pow_mag_max = np.array([r.ratc1 / base_mva for r in xfmrs]) # todo check normalization
bus_xfmr_orig_matrix = sp.csc_matrix(
    ([1.0 for i in range(num_xfmr)],
     (xfmr_orig_bus, list(range(num_xfmr)))),
    (num_bus, num_xfmr)).astype(np.float32).toarray()
bus_xfmr_dest_matrix = sp.csc_matrix(
    ([1.0 for i in range(num_xfmr)],
     (xfmr_dest_bus, list(range(num_xfmr)))),
    (num_bus, num_xfmr)).astype(np.float32).toarray()


#set_data_swsh_params
swshs = list(p.raw.switched_shunts.values())
num_swsh = len(swshs)
swsh_i = [r.i for r in swshs]
swsh_bus = [bus_map[swsh_i[i]] for i in range(num_swsh)]
swsh_map = {swsh_i[i]:i for i in range(num_swsh)}
swsh_status = np.array([r.stat for r in swshs])
swsh_adm_imag_max = np.array([
    (max(0.0, r.n1 * r.b1) +
     max(0.0, r.n2 * r.b2) +
     max(0.0, r.n3 * r.b3) +
     max(0.0, r.n4 * r.b4) +
     max(0.0, r.n5 * r.b5) +
     max(0.0, r.n6 * r.b6) +
     max(0.0, r.n7 * r.b7) +
     max(0.0, r.n8 * r.b8)) / base_mva
    for r in swshs]) * swsh_status 
swsh_adm_imag_min = np.array([
    (min(0.0, r.n1 * r.b1) +
     min(0.0, r.n2 * r.b2) +
     min(0.0, r.n3 * r.b3) +
     min(0.0, r.n4 * r.b4) +
     min(0.0, r.n5 * r.b5) +
     min(0.0, r.n6 * r.b6) +
     min(0.0, r.n7 * r.b7) +
     min(0.0, r.n8 * r.b8)) / base_mva
    for r in swshs]) * swsh_status 
bus_swsh_matrix = sp.csc_matrix(
    ([1.0 for i in range(num_swsh)],
     (swsh_bus, list(range(num_swsh)))),
    (num_bus, num_swsh)).astype(np.float32).toarray()
bus_swsh_adm_imag_max = bus_swsh_matrix.dot(swsh_adm_imag_max)#con
bus_swsh_adm_imag_min = bus_swsh_matrix.dot(swsh_adm_imag_min)#con
bus_swsh_adm_imag_max = np.transpose([bus_swsh_adm_imag_max])
bus_swsh_adm_imag_min = np.transpose([bus_swsh_adm_imag_min])


#set_data_gen_cost_params
gen_num_pl = [0 for i in range(num_gen)]
gen_pl_x = [None for i in range(num_gen)]
gen_pl_y = [None for i in range(num_gen)]
for r in p.rop.generator_dispatch_records.values():
    r_bus = r.bus
    r_genid = r.genid
    gen = gen_map[(r_bus, r_genid)]
    r_dsptbl = r.dsptbl
    s = p.rop.active_power_dispatch_records[r_dsptbl]
    r_ctbl = s.ctbl
    t = p.rop.piecewise_linear_cost_functions[r_ctbl]
    r_npairs = t.npairs
    gen_num_pl[gen] = r_npairs
    gen_pl_x[gen] = np.zeros(r_npairs)
    gen_pl_y[gen] = np.zeros(r_npairs)
    for i in range(r_npairs):
        gen_pl_x[gen][i] = t.points[i].x / base_mva
        gen_pl_y[gen][i] = t.points[i].y
    # from here on is checking assumptions and cleaning data - this should be done in a separate module to release clean datasets only
    assert(r_npairs > 1)
    for i in range(r_npairs - 1):
        assert (gen_pl_x[gen][i + 1] - gen_pl_x[gen][i]) >= 0.0
    i_to_keep = [0]
    for i in range(r_npairs - 1):
        if gen_pl_x[gen][i + 1] > gen_pl_x[gen][i]:
            i_to_keep.append(i + 1)
    gen_num_pl[gen] = len(i_to_keep)
    gen_pl_x[gen] = [gen_pl_x[gen][i] for i in i_to_keep]
    gen_pl_y[gen] = [gen_pl_y[gen][i] for i in i_to_keep]
    # check cost function convexity - this should be done in a separate module
    #'''
    if gen_num_pl[gen] > 2:
        d1 = [
            ((gen_pl_y[gen][i + 1] - gen_pl_y[gen][i]) /
             (gen_pl_x[gen][i + 1] - gen_pl_x[gen][i]))
            for i in range(gen_num_pl[gen] - 1)]
        d2 = [(d1[i + 1] - d1[i]) for i in range(gen_num_pl[gen] - 2)]
        for i in range(len(d2)):
            if d2[i] < 0.0:
                print('cost convexity error')
                print('gen i: %s' % r_bus)
                print('gen id: %s' % r_genid)
                print('num pairs: %s' % r_npairs)
                print('pairs:')
                print([(t.points[i].x, t.points[i].y) for i in range(r_npairs)])
                #print(gen_num_pl[gen])
                print('x points:')
                print(gen_pl_x[gen])
                print('y points:')
                print(gen_pl_y[gen])
                print('i: %s' % i)
                print('slopes:')
                print(d1)


#set_data_ctg_params
ctgs = p.con.contingencies.values()
num_ctg = len(ctgs)
ctg_label = [r.label for r in ctgs]
ctg_map = dict(zip(ctg_label, range(num_ctg)))
line_keys = set(line_key)
xfmr_keys = set(xfmr_key)
ctg_gen_keys_out = {
    r.label:set([(e.i, e.id) for e in r.generator_out_events])
    for r in ctgs}
ctg_branch_keys_out = {
    r.label:set([(e.i, e.j, e.ckt) for e in r.branch_out_events])
    for r in ctgs}
ctg_line_keys_out = {k:(v & line_keys) for k,v in ctg_branch_keys_out.items()}
ctg_xfmr_keys_out = {k:(v & xfmr_keys) for k,v in ctg_branch_keys_out.items()}
ctg_areas_affected = {
    k.label:(
        set([bus_area[bus_map[r[0]]] for r in ctg_gen_keys_out[k.label]]) |
        set([bus_area[bus_map[r[0]]] for r in ctg_branch_keys_out[k.label]]) |
        set([bus_area[bus_map[r[1]]] for r in ctg_branch_keys_out[k.label]]))
    for k in ctgs}
ctg_gens_out = [
    [gen_map[k] for k in ctg_gen_keys_out[ctg_label[i]]]
    for i in range(num_ctg)]
ctg_lines_out = [
    [line_map[k] for k in ctg_line_keys_out[ctg_label[i]]]
    for i in range(num_ctg)]
ctg_xfmrs_out = [
    [xfmr_map[k] for k in ctg_xfmr_keys_out[ctg_label[i]]]
    for i in range(num_ctg)]
ctg_areas_affected = [
    ctg_areas_affected[ctg_label[i]]
    for i in range(num_ctg)]

'''
set all data end
'''


'''
process data function 180-194
'''
penalty_block_pow_real_max = np.array(penalty_block_pow_real_max) / base_mva #bound
penalty_block_pow_real_coeff = np.array(penalty_block_pow_real_coeff) * base_mva #cost
penalty_block_pow_imag_max = np.array(penalty_block_pow_imag_max) / base_mva #bound
penalty_block_pow_imag_coeff = np.array(penalty_block_pow_imag_coeff) * base_mva #cost
penalty_block_pow_abs_max = np.array(penalty_block_pow_abs_max) / base_mva #bound
penalty_block_pow_abs_coeff = np.array(penalty_block_pow_abs_coeff) * base_mva #cost








def eval_piecewise_linear_penalty(residual, penalty_block_max, penalty_block_coeff):
    '''residual, penaltyblock_max, penalty_block_coeff are 1-dimensional numpy arrays'''

    r = residual
    num_block = len(penalty_block_coeff)
    num_block_bounded = len(penalty_block_max)
    assert(num_block_bounded + 1 == num_block)
    num_resid = r.get_shape().as_list()[0]
    abs_resid = np.abs(r)
    #penalty_block_max_extended = np.concatenate((penalty_block_max, np.inf))
    remaining_resid = abs_resid
    penalty = tf.Variable(tf.zeros([num_resid,1]))
    for i in range(num_block):
        #block_min = penalty_block_cumul_min[i]
        #block_max = penalty_block_cumul_max[i]
        block_coeff = penalty_block_coeff[i]
        if i < num_block - 1:
            block_max = penalty_block_max[i]
            penalized_resid = tf.minimum(tf.constant(block_max,dtype=tf.float32), remaining_resid)
            penalty = penalty + block_coeff * penalized_resid
            remaining_resid = remaining_resid - penalized_resid
        else:
            penalty = penalty + block_coeff * remaining_resid
    return penalty



#do calculation
class MyProblem(tfco.ConstrainedMinimizationProblem):

  def __init__(self):
    '''
    initial all para of object function
    
    '''
    #x. to initla good start point
    self.bus_volt_mag = tf.Variable(tf.ones([num_bus,1]),dtype=tf.float32)
    self.bus_volt_ang = tf.Variable(tf.zeros([num_bus,1]),dtype=tf.float32)
    self.bus_swsh_adm_imag = tf.Variable(tf.zeros([num_bus,1]),dtype=tf.float32) #shunt susceptance
    self.gen_pow_real = tf.Variable((gen_pow_real_max + gen_pow_real_min)/2,dtype=tf.float32) #generate real power pg
    self.gen_pow_imag = tf.Variable(tf.zeros([num_gen,1]),dtype=tf.float32) #generate imag power pg
    self.obj = tf.Variable(0.)
    self.test = tf.Variable(0.)

    '''
    end of initialize object function
    
    '''


  @property
  def objective(self):
    '''
    self.cost = tf.Variable(0) #total cost sum of cg
    self.penalty = tf.Variable(0) #pena c sig c sig k
    self.base_penalty = tf.Variable(0)
    self.ctg_penalty = tf.Variable(0)   
    self.infeas_all = 1 # starts out infeasible ??
    '''
    
    #function 2
    pl_x = np.array(gen_pl_x).astype(np.float32)
    pl_y = np.array(gen_pl_y).astype(np.float32)
    
    #suppose the slope increase and ignore the insecure situation less and larger        
    slope = (pl_y[...,1:] - pl_y[...,:-1]) / (pl_x[...,1:] - pl_x[...,:-1])
    x_change = tf.maximum(tf.zeros([num_gen,1]), tf.minimum(pl_x[...,1:] - pl_x[...,:-1], self.gen_pow_real - pl_x[...,:-1]))
    y_value = pl_y[...,0]
    gen_cost = tf.constant(np.sum(y_value)) + tf.reduce_sum(slope * x_change)
           
    #if gen_status[k] == 0.0:#todo 
        #continue
    cost = gen_cost
    
    
    #line part
    line_orig_volt_mag = tf.gather(self.bus_volt_mag,line_orig_bus)
    line_dest_volt_mag = tf.gather(self.bus_volt_mag,line_dest_bus)
    line_volt_ang_diff = tf.gather(self.bus_volt_ang,line_orig_bus) - tf.gather(self.bus_volt_ang,line_dest_bus)
    line_cos_volt_ang_diff = tf.cos(line_volt_ang_diff)
    line_sin_volt_ang_diff = tf.sin(line_volt_ang_diff)
    line_orig_dest_volt_mag_prod = line_orig_volt_mag * line_dest_volt_mag
    line_orig_volt_mag_sq = line_orig_volt_mag ** 2.0
    line_dest_volt_mag_sq = line_dest_volt_mag ** 2.0
    
    #function 38
    line_pow_orig_real = ( # line_status not needed as we have already done it on the parameter level
        line_adm_real * line_orig_volt_mag_sq + # ** 2.0 +
        ( - line_adm_real * line_cos_volt_ang_diff
          - line_adm_imag * line_sin_volt_ang_diff) *
        line_orig_dest_volt_mag_prod)
    #function 39
    line_pow_orig_imag = (
        - line_adm_total_imag * line_orig_volt_mag_sq + # ** 2.0 +
        (   line_adm_imag * line_cos_volt_ang_diff
          - line_adm_real * line_sin_volt_ang_diff) *
        line_orig_dest_volt_mag_prod)
    #function 40
    line_pow_dest_real = (
        line_adm_real * line_dest_volt_mag_sq + # ** 2.0 +
        ( - line_adm_real * line_cos_volt_ang_diff
          + line_adm_imag * line_sin_volt_ang_diff) *
        line_orig_dest_volt_mag_prod)
    #function 41
    line_pow_dest_imag = (
        - line_adm_total_imag * line_dest_volt_mag_sq + # ** 2.0 +
        (   line_adm_imag * line_cos_volt_ang_diff
          + line_adm_real * line_sin_volt_ang_diff) *
        line_orig_dest_volt_mag_prod)
    #function 52-53
    line_curr_orig_mag_max_viol = tf.maximum(
        0.0,
        (line_pow_orig_real**2.0 + line_pow_orig_imag**2.0 + hard_constr_tol)**0.5 -
        line_curr_mag_max * line_orig_volt_mag)
    #function 54
    line_curr_dest_mag_max_viol = tf.maximum(
        0.0,
        (line_pow_dest_real**2.0 + line_pow_dest_imag**2.0 + hard_constr_tol)**0.5 -
        line_curr_mag_max * line_dest_volt_mag)
    
    #trnas aprt
    xfmr_orig_volt_mag = tf.gather(self.bus_volt_mag,xfmr_orig_bus)
    xfmr_dest_volt_mag = tf.gather(self.bus_volt_mag,xfmr_dest_bus)
    xfmr_volt_ang_diff = tf.gather(self.bus_volt_ang,xfmr_orig_bus) - tf.gather(self.bus_volt_ang,xfmr_dest_bus) - xfmr_tap_ang
    xfmr_cos_volt_ang_diff = tf.cos(xfmr_volt_ang_diff)
    xfmr_sin_volt_ang_diff = tf.sin(xfmr_volt_ang_diff)
    #xfmr_orig_dest_volt_mag_prod = xfmr_orig_volt_mag * xfmr_dest_volt_mag
    xfmr_orig_volt_mag_sq = xfmr_orig_volt_mag ** 2.0
    xfmr_dest_volt_mag_sq = xfmr_dest_volt_mag ** 2.0
    #function 42
    xfmr_pow_orig_real = (
        (xfmr_adm_real / xfmr_tap_mag**2.0 + xfmr_adm_mag_real) * xfmr_orig_volt_mag_sq +
        ( - xfmr_adm_real / xfmr_tap_mag * xfmr_cos_volt_ang_diff
          - xfmr_adm_imag / xfmr_tap_mag * xfmr_sin_volt_ang_diff) *
            xfmr_orig_volt_mag * xfmr_dest_volt_mag)
    #function 43
    xfmr_pow_orig_imag = (
        - (xfmr_adm_imag / xfmr_tap_mag**2.0 + xfmr_adm_mag_imag) * xfmr_orig_volt_mag_sq +
        (   xfmr_adm_imag / xfmr_tap_mag * xfmr_cos_volt_ang_diff
          - xfmr_adm_real / xfmr_tap_mag * xfmr_sin_volt_ang_diff) *
            xfmr_orig_volt_mag * xfmr_dest_volt_mag)
    #function 44
    xfmr_pow_dest_real = (
        xfmr_adm_real * xfmr_dest_volt_mag_sq +
        ( - xfmr_adm_real / xfmr_tap_mag * xfmr_cos_volt_ang_diff
          + xfmr_adm_imag / xfmr_tap_mag * xfmr_sin_volt_ang_diff) *
            xfmr_orig_volt_mag * xfmr_dest_volt_mag)
    #function 45
    xfmr_pow_dest_imag = (
        - xfmr_adm_imag * xfmr_dest_volt_mag_sq +
        (   xfmr_adm_imag / xfmr_tap_mag * xfmr_cos_volt_ang_diff
          + xfmr_adm_real / xfmr_tap_mag * xfmr_sin_volt_ang_diff) *
            xfmr_orig_volt_mag * xfmr_dest_volt_mag)
    #function 55-56    
    xfmr_pow_orig_mag_max_viol = tf.maximum(
        0.0,
        (xfmr_pow_orig_real**2.0 + xfmr_pow_orig_imag**2.0 + hard_constr_tol)**0.5 -
        xfmr_pow_mag_max)
    #function 57
    xfmr_pow_dest_mag_max_viol = tf.maximum(
        0.0,
        (xfmr_pow_dest_real**2.0 + xfmr_pow_dest_imag**2.0 + hard_constr_tol)**0.5 -
        xfmr_pow_mag_max)
    
    #power part
    bus_load_pow_real = bus_load_const_pow_real
    bus_load_pow_imag = bus_load_const_pow_imag
    #function 46 bFS v^2
    bus_fxsh_pow_real = bus_fxsh_adm_real * (self.bus_volt_mag ** 2.0)
    #function 49 bFS v^2
    bus_fxsh_pow_imag = - bus_fxsh_adm_imag * (self.bus_volt_mag ** 2.0)
    #function 49 bCS v^2
    bus_swsh_pow_imag = -self.bus_swsh_adm_imag * self.bus_volt_mag**2.0
    
    #function 46
    bus_pow_balance_real_viol = (
        tf.matmul(tf.constant(bus_gen_matrix),self.gen_pow_real) -
        bus_load_pow_real -
        bus_fxsh_pow_real -
        tf.matmul(tf.constant(bus_line_orig_matrix),line_pow_orig_real) -
        tf.matmul(tf.constant(bus_line_dest_matrix),line_pow_dest_real) -
        tf.matmul(tf.constant(bus_xfmr_orig_matrix),xfmr_pow_orig_real) -
        tf.matmul(tf.constant(bus_xfmr_dest_matrix),xfmr_pow_dest_real))
    #function 49
    bus_pow_balance_imag_viol = (
        tf.matmul(tf.constant(bus_gen_matrix),self.gen_pow_imag) -
        bus_load_pow_imag -
        bus_fxsh_pow_imag -
        bus_swsh_pow_imag -
        tf.matmul(tf.constant(bus_line_orig_matrix),line_pow_orig_imag) -
        tf.matmul(tf.constant(bus_line_dest_matrix),line_pow_dest_imag) -
        tf.matmul(tf.constant(bus_xfmr_orig_matrix),xfmr_pow_orig_imag) -
        tf.matmul(tf.constant(bus_xfmr_dest_matrix),xfmr_pow_dest_imag))


    #function 6-19
    base_penalty = base_case_penalty_weight * (
        tf.reduce_sum(
            eval_piecewise_linear_penalty(
                tf.maximum(
                    line_curr_orig_mag_max_viol,
                    line_curr_dest_mag_max_viol),
                penalty_block_pow_abs_max,
                penalty_block_pow_abs_coeff)) +
        tf.reduce_sum(
            eval_piecewise_linear_penalty(
                tf.maximum(
                    xfmr_pow_orig_mag_max_viol,
                    xfmr_pow_dest_mag_max_viol),
                penalty_block_pow_abs_max,
                penalty_block_pow_abs_coeff)) +
        tf.reduce_sum(
            eval_piecewise_linear_penalty(
                bus_pow_balance_real_viol,
                penalty_block_pow_real_max,

                penalty_block_pow_real_coeff)) +
        tf.reduce_sum(
            eval_piecewise_linear_penalty(
                bus_pow_balance_imag_viol,
                penalty_block_pow_imag_max,
                penalty_block_pow_imag_coeff)))


    
    #ctg part
    #todo change

    ctg_penalty = tf.Variable(0.)
    for i in range(len(ctgs)):
        ctg_current = ctg_map[ctg_label[i]]
        
        #genpart out of service to set the minmax as zeros
        gens_out_of_service = set(gen_out_of_service) | set(ctg_gens_out[ctg_current])
        ctg_gen_out_of_service = sorted(list(gens_out_of_service))

        #todo check none situation
        ctg_power_multi_vector = np.ones([num_gen,1])
        for index in ctg_gen_out_of_service:
            ctg_power_multi_vector[index] = 0.
        ctg_gen_pow_real = self.gen_pow_real * ctg_power_multi_vector
        ctg_gen_pow_imag = self.gen_pow_imag * ctg_power_multi_vector

        #line part
        ctg_line_orig_volt_mag = tf.gather(self.bus_volt_mag,line_orig_bus)
        ctg_line_dest_volt_mag = tf.gather(self.bus_volt_mag,line_dest_bus)
        ctg_line_volt_ang_diff = tf.gather(self.bus_volt_ang,line_orig_bus) - tf.gather(self.bus_volt_ang,line_dest_bus)
        ctg_line_cos_volt_ang_diff = tf.cos(ctg_line_volt_ang_diff)
        ctg_line_sin_volt_ang_diff = tf.sin(ctg_line_volt_ang_diff)
        ctg_line_orig_dest_volt_mag_prod = ctg_line_orig_volt_mag * ctg_line_dest_volt_mag
        ctg_line_orig_volt_mag_sq = ctg_line_orig_volt_mag ** 2.0
        ctg_line_dest_volt_mag_sq = ctg_line_dest_volt_mag ** 2.0
        
        #function 64
        ctg_line_pow_orig_real = ( # line_status not needed as we have already done it on the parameter level
            line_adm_real * ctg_line_orig_volt_mag_sq + # ** 2.0 +
            ( - line_adm_real * ctg_line_cos_volt_ang_diff
              - line_adm_imag * ctg_line_sin_volt_ang_diff) *
            ctg_line_orig_dest_volt_mag_prod)
        #function 65
        ctg_line_pow_orig_imag = (
            - line_adm_total_imag * ctg_line_orig_volt_mag_sq + # ** 2.0 +
            (   line_adm_imag * ctg_line_cos_volt_ang_diff
              - line_adm_real * ctg_line_sin_volt_ang_diff) *
            ctg_line_orig_dest_volt_mag_prod)
        #function 66
        ctg_line_pow_dest_real = (
            line_adm_real * ctg_line_dest_volt_mag_sq + # ** 2.0 +
            ( - line_adm_real * ctg_line_cos_volt_ang_diff
              + line_adm_imag * ctg_line_sin_volt_ang_diff) *
            ctg_line_orig_dest_volt_mag_prod)
        #function 67
        ctg_line_pow_dest_imag = (
            - line_adm_total_imag * ctg_line_dest_volt_mag_sq + # ** 2.0 +
            (   line_adm_imag * ctg_line_cos_volt_ang_diff
              + line_adm_real * ctg_line_sin_volt_ang_diff) *
            ctg_line_orig_dest_volt_mag_prod)
        #todo check gradient 
        ctg_line_multi_vector = np.ones([num_line,1])
        for index in ctg_lines_out[ctg_current]:
            ctg_line_multi_vector[index] = 0.
        ctg_line_pow_orig_real = ctg_line_pow_orig_real * ctg_line_multi_vector
        ctg_line_pow_orig_imag = ctg_line_pow_orig_imag * ctg_line_multi_vector
        ctg_line_pow_dest_real = ctg_line_pow_dest_real * ctg_line_multi_vector
        ctg_line_pow_dest_imag = ctg_line_pow_dest_imag * ctg_line_multi_vector
    
        
        
        #function 78
        ctg_line_curr_orig_mag_max_viol = tf.maximum(
            0.0,
            (ctg_line_pow_orig_real**2.0 + ctg_line_pow_orig_imag**2.0 + hard_constr_tol)**0.5 -
            ctg_line_curr_mag_max * ctg_line_orig_volt_mag)
        #function 80
        ctg_line_curr_dest_mag_max_viol = tf.maximum(
            0.0,
            (ctg_line_pow_dest_real**2.0 + ctg_line_pow_dest_imag**2.0 + hard_constr_tol)**0.5 -
            ctg_line_curr_mag_max * ctg_line_dest_volt_mag)
    
        
        
        #trnas aprt
        ctg_xfmr_orig_volt_mag = tf.gather(self.bus_volt_mag,xfmr_orig_bus)
        ctg_xfmr_dest_volt_mag = tf.gather(self.bus_volt_mag,xfmr_dest_bus)
        ctg_xfmr_volt_ang_diff = tf.gather(self.bus_volt_ang,xfmr_orig_bus) - tf.gather(self.bus_volt_ang,xfmr_dest_bus) - xfmr_tap_ang
        ctg_xfmr_cos_volt_ang_diff = tf.cos(ctg_xfmr_volt_ang_diff)
        ctg_xfmr_sin_volt_ang_diff = tf.sin(ctg_xfmr_volt_ang_diff)
        #xfmr_orig_dest_volt_mag_prod = xfmr_orig_volt_mag * xfmr_dest_volt_mag
        ctg_xfmr_orig_volt_mag_sq = ctg_xfmr_orig_volt_mag ** 2.0
        ctg_xfmr_dest_volt_mag_sq = ctg_xfmr_dest_volt_mag ** 2.0
        #function 68
        ctg_xfmr_pow_orig_real = (
            (xfmr_adm_real / xfmr_tap_mag**2.0 + xfmr_adm_mag_real) * ctg_xfmr_orig_volt_mag_sq +
            ( - xfmr_adm_real / xfmr_tap_mag * ctg_xfmr_cos_volt_ang_diff
              - xfmr_adm_imag / xfmr_tap_mag * ctg_xfmr_sin_volt_ang_diff) *
                ctg_xfmr_orig_volt_mag * ctg_xfmr_dest_volt_mag)
        #function 69
        ctg_xfmr_pow_orig_imag = (
            - (xfmr_adm_imag / xfmr_tap_mag**2.0 + xfmr_adm_mag_imag) * ctg_xfmr_orig_volt_mag_sq +
            (   xfmr_adm_imag / xfmr_tap_mag * ctg_xfmr_cos_volt_ang_diff
              - xfmr_adm_real / xfmr_tap_mag * ctg_xfmr_sin_volt_ang_diff) *
                ctg_xfmr_orig_volt_mag * ctg_xfmr_dest_volt_mag)
        #function 70
        ctg_xfmr_pow_dest_real = (
            xfmr_adm_real * ctg_xfmr_dest_volt_mag_sq +
            ( - xfmr_adm_real / xfmr_tap_mag * ctg_xfmr_cos_volt_ang_diff
              + xfmr_adm_imag / xfmr_tap_mag * ctg_xfmr_sin_volt_ang_diff) *
                ctg_xfmr_orig_volt_mag * ctg_xfmr_dest_volt_mag)
        #function 71
        ctg_xfmr_pow_dest_imag = (
            - xfmr_adm_imag * ctg_xfmr_dest_volt_mag_sq +
            (   xfmr_adm_imag / xfmr_tap_mag * ctg_xfmr_cos_volt_ang_diff
              + xfmr_adm_real / xfmr_tap_mag * ctg_xfmr_sin_volt_ang_diff) *
                ctg_xfmr_orig_volt_mag * ctg_xfmr_dest_volt_mag)
            
        #todo check none situation
        ctg_xfmr_multi_vector = np.ones([num_xfmr,1])
        for index in ctg_xfmrs_out[ctg_current]:
            ctg_xfmr_multi_vector[index] = 0.
        ctg_xfmr_pow_orig_real = ctg_xfmr_pow_orig_real * ctg_xfmr_multi_vector
        ctg_xfmr_pow_orig_imag = ctg_xfmr_pow_orig_imag * ctg_xfmr_multi_vector
        ctg_xfmr_pow_dest_real = ctg_xfmr_pow_dest_real * ctg_xfmr_multi_vector
        ctg_xfmr_pow_dest_imag = ctg_xfmr_pow_dest_imag * ctg_xfmr_multi_vector  
        
        #function 81    
        ctg_xfmr_pow_orig_mag_max_viol = tf.maximum(
            0.0,
            (ctg_xfmr_pow_orig_real**2.0 + ctg_xfmr_pow_orig_imag**2.0 + hard_constr_tol)**0.5 -
            ctg_xfmr_pow_mag_max)
        #function 83
        ctg_xfmr_pow_dest_mag_max_viol = tf.maximum(
            0.0,
            (ctg_xfmr_pow_dest_real**2.0 + ctg_xfmr_pow_dest_imag**2.0 + hard_constr_tol)**0.5 -
            ctg_xfmr_pow_mag_max)
    
               
        #power part
        ctg_bus_load_pow_real = bus_load_const_pow_real
        ctg_bus_load_pow_imag = bus_load_const_pow_imag
        #function 72 bFS v^2
        ctg_bus_fxsh_pow_real = bus_fxsh_adm_real * (self.bus_volt_mag ** 2.0)
        #function 75 bFS v^2
        ctg_bus_fxsh_pow_imag = - bus_fxsh_adm_imag * (self.bus_volt_mag ** 2.0)
        #function 75 bCS v^2
        ctg_bus_swsh_pow_imag = -self.bus_swsh_adm_imag * self.bus_volt_mag**2.0
        
        
        #function 72,75
        ctg_bus_pow_balance_real_viol = tf.abs(
            tf.matmul(tf.constant(bus_gen_matrix),ctg_gen_pow_real) -
            ctg_bus_load_pow_real -
            ctg_bus_fxsh_pow_real -
            tf.matmul(tf.constant(bus_line_orig_matrix),ctg_line_pow_orig_real) -
            tf.matmul(tf.constant(bus_line_dest_matrix),ctg_line_pow_dest_real) -
            tf.matmul(tf.constant(bus_xfmr_orig_matrix),ctg_xfmr_pow_orig_real) -
            tf.matmul(tf.constant(bus_xfmr_dest_matrix),ctg_xfmr_pow_dest_real))
        ctg_bus_pow_balance_imag_viol = tf.abs(
            tf.matmul(tf.constant(bus_gen_matrix),ctg_gen_pow_imag) -
            ctg_bus_load_pow_imag -
            ctg_bus_fxsh_pow_imag -
            ctg_bus_swsh_pow_imag -
            tf.matmul(tf.constant(bus_line_orig_matrix),ctg_line_pow_orig_imag) -
            tf.matmul(tf.constant(bus_line_dest_matrix),ctg_line_pow_dest_imag) -
            tf.matmul(tf.constant(bus_xfmr_orig_matrix),ctg_xfmr_pow_orig_imag) -
            tf.matmul(tf.constant(bus_xfmr_dest_matrix),ctg_xfmr_pow_dest_imag))
        
        
        #function 6-19
        ctg_penalty_now =  (
                tf.reduce_sum(
                    eval_piecewise_linear_penalty(
                        tf.maximum(
                            ctg_line_curr_orig_mag_max_viol,
                            ctg_line_curr_dest_mag_max_viol),
                        penalty_block_pow_abs_max,
                        penalty_block_pow_abs_coeff)) +
                tf.reduce_sum(
                    eval_piecewise_linear_penalty(
                        tf.maximum(
                            ctg_xfmr_pow_orig_mag_max_viol,
                            ctg_xfmr_pow_dest_mag_max_viol),
                        penalty_block_pow_abs_max,
                        penalty_block_pow_abs_coeff)) +
                tf.reduce_sum(
                    eval_piecewise_linear_penalty(
                        ctg_bus_pow_balance_real_viol,
                        penalty_block_pow_real_max,
                        penalty_block_pow_real_coeff)) +
                tf.reduce_sum(
                    eval_piecewise_linear_penalty(
                        ctg_bus_pow_balance_imag_viol,
                        penalty_block_pow_imag_max,
                        penalty_block_pow_imag_coeff)))
                
        ctg_penalty = ctg_penalty + ctg_penalty_now
    
    
    #function 1
    penalty = base_case_penalty_weight * base_penalty + (1-base_case_penalty_weight)/max(1.0, float(num_ctg)) * ctg_penalty 
    #todo need add other penalty
    self.test = penalty 
    self.obj = cost + penalty
    
    return self.obj





  @property
  def constraints(self):

    # The constraint is (recall >= self._recall_lower_bound), which we convert
    # to (self._recall_lower_bound - recall <= 0) because
    # ConstrainedMinimizationProblems must always provide their constraints in
    # the form (tensor <= 0).    
    # The result of this function should be a tensor, with each element being
    # a quantity that is constrained to be nonpositive. We only have one
    # constraint, so we return a one-element tensor.
    
    constraint_1 = self.bus_volt_mag - bus_volt_mag_max    
    constraint_2 = -self.bus_volt_mag + bus_volt_mag_min
    constraint_3 = self.bus_swsh_adm_imag - bus_swsh_adm_imag_max
    constraint_4 = -self.bus_swsh_adm_imag + bus_swsh_adm_imag_min
    constraint_5 = self.gen_pow_real - gen_pow_real_max
    constraint_6 = -self.gen_pow_real + gen_pow_real_min
    constraint_7 = self.gen_pow_imag - gen_pow_imag_max
    constraint_8 = -self.gen_pow_imag + gen_pow_imag_min
    constraints = tf.concat([constraint_1,constraint_2,constraint_3,constraint_4,constraint_5,constraint_6,constraint_7,constraint_8], axis = 0)    
    
    return constraints



  @property
  def proxy_constraints(self):
    # Use 1 - hinge since we're SUBTRACTING recall in the constraint function,
    # and we want the proxy constraint function to be convex.
    #true_positives = self._labels * tf.minimum(1.0, self._predictions)
    #true_positive_count = tf.reduce_sum(true_positives)
    #recall = true_positive_count / self._positive_count
    # Please see the corresponding comment in the constraints property.
    constraint_1 = self.bus_volt_mag - bus_volt_mag_max    
    constraint_2 = -self.bus_volt_mag + bus_volt_mag_min
    constraint_3 = self.bus_swsh_adm_imag - bus_swsh_adm_imag_max
    constraint_4 = -self.bus_swsh_adm_imag + bus_swsh_adm_imag_min
    constraint_5 = self.gen_pow_real - gen_pow_real_max
    constraint_6 = -self.gen_pow_real + gen_pow_real_min
    constraint_7 = self.gen_pow_imag - gen_pow_imag_max
    constraint_8 = -self.gen_pow_imag + gen_pow_imag_min
    constraints = tf.concat([constraint_1,constraint_2,constraint_3,constraint_4,constraint_5,constraint_6,constraint_7,constraint_8], axis = 0)    
    
    return constraints




#main start
problem = MyProblem()
itertime = 20000
print('now')
with tf.Session() as session:
    '''
    https://git.codingcafe.org/Mirrors/tensorflow/tensorflow/commit/ff15c81e2b92ef8fb47bb15790cffd18377a4ef2?expanded=1
    AdditiveExternalRegretOptimizer
    AdditiveSwapRegretOptimizer
    MultiplicativeSwapRegretOptimizer
    '''
    optimizer = tfco.AdditiveExternalRegretOptimizer(
            optimizer=tf.train.AdagradOptimizer(learning_rate = 0.01))
    
    train_op = optimizer.minimize(problem)
    session.run(tf.global_variables_initializer())
    
    
    # 2 methods 1st:iter time ;  2nd:runing time
    while True:
    #for i in range(itertime):
        session.run(train_op)
        '''
        if i % 5000 == 0:
            print(i)
            print('volt_mag ')
            print(session.run(tf.transpose(problem.bus_volt_mag)))
            print('volt_ang ')
            print(session.run(tf.transpose(problem.bus_volt_ang)))
            print('gen_real ')
            print(session.run(tf.transpose(problem.gen_pow_real)))
            print('gen_img ')
            print(session.run(tf.transpose(problem.gen_pow_imag)))
            print('swsh ')
            print(session.run(tf.transpose(problem.bus_swsh_adm_imag)))
            print('base_pen')
            print(session.run(problem.test))
            print('obj')
            print(session.run(problem.obj))
        '''
        if time.time() - start_time > 60 * 5 - 30 : #5min #change to str write
        #if i == itertime-1:
            
            #write sol1
            volt_mag = list(session.run(tf.transpose(problem.bus_volt_mag))[0])
            volt_ang = list(session.run(tf.transpose(problem.bus_volt_ang))[0])
            gen_real = list(session.run(tf.transpose(problem.gen_pow_real))[0])
            gen_img = list(session.run(tf.transpose(problem.gen_pow_imag))[0])
            swsh = list(session.run(tf.transpose(problem.bus_swsh_adm_imag))[0])
            
            sol1_content = ''            
            sol1_content += '--bus section\n'
            sol1_content += 'i, v, theta, b\n'
            part1 = list(zip(volt_mag,volt_ang,swsh))
            for i in range(len(part1)):
                sol1_content += str(i+1) + ','
                sol1_content += str(part1[i]) + '\n'
            
            sol1_content += '--generator section\n'
            sol1_content += 'i, uid, p, q\n'
            part2 = list(zip(gen_real,gen_img))
            for i in range(len(part2)):
                sol1_content += '{},'.format(gen_key[i])
                sol1_content += str(part2[i])+'\n'
            #remove ()
            sol1_content = sol1_content.replace('(','').replace(')','')

            file_w = open('solution1.txt','w')
            file_w.write(sol1_content)
            file_w.close()
            
            break
            


        






