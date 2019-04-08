# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 15:21:22 2019

@author: Zhao Huan
Step 1: read para from data
Step 2: build function
Step 3: solve by tensorflow
"""
import time
start_time = time.time()

import numpy as np
from scipy import sparse as sp
import math
import tensorflow as tf
import argparse
import data
import sys,os



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



args = sys.argv

con_name = args[1]
inl_name = args[2]
raw_name = args[3]
rop_name = args[4]
time_limit = args[5]
score_method = args[6]
network_model = args[7]
sol1_name = 'solution1.txt'
sol2_name = 'solution2.txt'

raw = os.path.normpath(raw_name)
rop = os.path.normpath(rop_name)
con = os.path.normpath(con_name)
inl = os.path.normpath(inl_name)



#read data, using the data object in evaluation start   
'''
=================================
my part start
=================================
'''

p = data.Data()
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
bus_map = {}
for i in range(len(bus_i)):
    bus_map[bus_i[i]]= i 
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
load_map = {}
for i in range(num_load):
    load_map[(load_i[i], load_id[i])] = i
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
fxsh_map = {}
for i in range(num_fxsh):
    fxsh_map[(fxsh_i[i], fxsh_id[i])] = i 
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
gen_map = {}
for i in range(num_gen):
    gen_map[(gen_i[i], gen_id[i])] = i 
gen_status = np.array([r.stat for r in gens])
gen_pow_imag_max = np.array([r.qt / base_mva for r in gens]) * gen_status #cons
gen_pow_imag_min = np.array([r.qb / base_mva for r in gens]) * gen_status #cons
gen_pow_real_max = np.array([r.pt / base_mva for r in gens]) * gen_status #cons
gen_pow_real_min = np.array([r.pb / base_mva for r in gens]) * gen_status #cons
gen_part_fact = {}
for r in p.inl.generator_inl_records.values():
    gen_part_fact[(r.i, r.id)] = r.r 
    
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
line_map = {}
for i in range(num_line):
    line_map[(line_i[i], line_j[i], line_ckt[i])] = i 
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
ctg_line_curr_mag_max = np.transpose([ctg_line_curr_mag_max])


#set_data_xfmr_params transformers
xfmrs = list(p.raw.transformers.values())
xfmr_key = [(r.i, r.j, r.ckt) for r in xfmrs]
num_xfmr = len(xfmrs)
xfmr_i = [r.i for r in xfmrs]
xfmr_j = [r.j for r in xfmrs]
xfmr_ckt = [r.ckt for r in xfmrs]
xfmr_orig_bus = [bus_map[xfmr_i[i]] for i in range(num_xfmr)]
xfmr_dest_bus = [bus_map[xfmr_j[i]] for i in range(num_xfmr)]
xfmr_map = {}
for i in range(num_xfmr):
    xfmr_map[(xfmr_i[i], xfmr_j[i], xfmr_ckt[i])] = i 
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
ctg_xfmr_pow_mag_max = np.transpose([ctg_xfmr_pow_mag_max])
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
swsh_map = {}
for i in range(num_swsh):
    swsh_map[swsh_i[i]] = i 
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
max_gen_num_pl = max(gen_num_pl)
#fill to a matrix
for i in range(num_gen):
    while len(gen_pl_x[i]) < max_gen_num_pl:
        gen_pl_x[i].insert(len(gen_pl_x[i])-1,(gen_pl_x[i][-1]+gen_pl_x[i][-2])/2)
        gen_pl_y[i].insert(len(gen_pl_y[i])-1,(gen_pl_y[i][-1]+gen_pl_y[i][-2])/2)


#set_data_ctg_params
ctgs = p.con.contingencies.values()
num_ctg = len(ctgs)
ctg_label = [r.label for r in ctgs]
ctg_map = dict(zip(ctg_label, range(num_ctg)))
line_keys = set(line_key)
xfmr_keys = set(xfmr_key)
ctg_gen_keys_out = {}
for r in ctgs:
    ctg_gen_keys_out[r.label] = set([(e.i, e.id) for e in r.generator_out_events])
    
ctg_branch_keys_out = {}
for r in ctgs:
    ctg_branch_keys_out[r.label] = set([(e.i, e.j, e.ckt) for e in r.branch_out_events])
    
ctg_line_keys_out = {}
for k,v in ctg_branch_keys_out.items():
    ctg_line_keys_out[k] = (v & line_keys) 
    
ctg_xfmr_keys_out = {}
for k,v in ctg_branch_keys_out.items():
    ctg_xfmr_keys_out[k] = (v & xfmr_keys) 
    
ctg_areas_affected = {}
for k in ctgs:
    ctg_areas_affected[k.label] = (
        set([bus_area[bus_map[r[0]]] for r in ctg_gen_keys_out[k.label]]) |
        set([bus_area[bus_map[r[0]]] for r in ctg_branch_keys_out[k.label]]) |
        set([bus_area[bus_map[r[1]]] for r in ctg_branch_keys_out[k.label]]))
    
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
process data function 180-194
'''
penalty_block_pow_real_max = np.array(penalty_block_pow_real_max) / base_mva #bound
penalty_block_pow_real_coeff = np.array(penalty_block_pow_real_coeff) * base_mva #cost
penalty_block_pow_imag_max = np.array(penalty_block_pow_imag_max) / base_mva #bound
penalty_block_pow_imag_coeff = np.array(penalty_block_pow_imag_coeff) * base_mva #cost
penalty_block_pow_abs_max = np.array(penalty_block_pow_abs_max) / base_mva #bound
penalty_block_pow_abs_coeff = np.array(penalty_block_pow_abs_coeff) * base_mva #cost


'''
set all data end
'''


print('finish read data')



#start tensorflow variables


def eval_piecewise_linear_penalty(residual, penalty_block_max, penalty_block_coeff):
    '''residual, penaltyblock_max, penalty_block_coeff are 1-dimensional numpy arrays'''

    r = tf.abs(residual)
    penalty1 = tf.constant(penalty_block_coeff[0],dtype=tf.float32) * tf.minimum(tf.constant(penalty_block_max[0],dtype=tf.float32 ),r - tf.constant(0.))
    penalty2 = tf.constant(penalty_block_coeff[1],dtype=tf.float32) * tf.minimum(tf.maximum(tf.constant(0.),r - tf.constant(penalty_block_max[0],dtype=tf.float32)), tf.constant(penalty_block_max[1]-penalty_block_max[0],dtype=tf.float32))
    penalty3 = tf.constant(penalty_block_coeff[2],dtype=tf.float32) * tf.maximum(tf.constant(0.),r - tf.constant(penalty_block_max[1],dtype=tf.float32))
    penalty = penalty1 + penalty2 + penalty3
    return penalty





#x. to initla good start point
bus_volt_mag = tf.Variable(tf.ones([num_bus,1]),dtype=tf.float32)
bus_volt_ang = tf.Variable(tf.zeros([num_bus,1]),dtype=tf.float32) * (math.pi/180)
bus_swsh_adm_imag = tf.Variable(tf.zeros([num_bus,1]),dtype=tf.float32) #shunt susceptance
gen_pow_real = tf.Variable((gen_pow_real_max + gen_pow_real_min)/2,dtype=tf.float32) #generate real power pg
gen_pow_imag = tf.Variable(tf.zeros([num_gen,1]),dtype=tf.float32) #generate imag power pg



#function 2
pl_x = np.array(gen_pl_x).astype(np.float32)
pl_y = np.array(gen_pl_y).astype(np.float32)

#suppose the slope increase and ignore the insecure situation less and larger        
slope =  tf.constant((pl_y[...,1:] - pl_y[...,:-1]) / (pl_x[...,1:] - pl_x[...,:-1]))
max_output = tf.constant(pl_x[...,1:] - pl_x[...,:-1])
x_change = tf.maximum(tf.constant(np.zeros([num_gen,1]),tf.float32), tf.minimum(max_output, gen_pow_real - tf.constant(pl_x[...,:-1])))
y_value = pl_y[...,0]
gen_cost = tf.constant(np.sum(y_value)) + tf.reduce_sum(slope * x_change)
       
#if gen_status[k] == 0.0:#todo 
    #continue
cost = gen_cost
 
#line part
constant_line_orig_bus = tf.constant(line_orig_bus)
constant_line_dest_bus = tf.constant(line_dest_bus)
line_orig_volt_mag = tf.gather(bus_volt_mag, constant_line_orig_bus)
line_dest_volt_mag = tf.gather(bus_volt_mag, constant_line_dest_bus)
line_volt_ang_diff = tf.gather(bus_volt_ang, constant_line_orig_bus) - tf.gather(bus_volt_ang,constant_line_dest_bus)
line_cos_volt_ang_diff = tf.cos(line_volt_ang_diff)
line_sin_volt_ang_diff = tf.sin(line_volt_ang_diff)
line_orig_dest_volt_mag_prod = line_orig_volt_mag * line_dest_volt_mag
line_orig_volt_mag_sq = line_orig_volt_mag ** 2.0
line_dest_volt_mag_sq = line_dest_volt_mag ** 2.0

#function 38
constant_line_adm_real = tf.constant(line_adm_real)
constant_line_adm_imag = tf.constant(line_adm_imag)
constant_line_adm_total_imag = tf.constant(line_adm_total_imag, dtype = tf.float32)
line_pow_orig_real = ( # line_status not needed as we have already done it on the parameter level
    constant_line_adm_real * line_orig_volt_mag_sq + # ** 2.0 +
    ( - constant_line_adm_real * line_cos_volt_ang_diff
      - constant_line_adm_imag * line_sin_volt_ang_diff) *
    line_orig_dest_volt_mag_prod)
#function 39
line_pow_orig_imag = (
    - constant_line_adm_total_imag * line_orig_volt_mag_sq + # ** 2.0 +
    (   constant_line_adm_imag * line_cos_volt_ang_diff
      - constant_line_adm_real * line_sin_volt_ang_diff) *
    line_orig_dest_volt_mag_prod)
#function 40
line_pow_dest_real = (
    constant_line_adm_real * line_dest_volt_mag_sq + # ** 2.0 +
    ( - constant_line_adm_real * line_cos_volt_ang_diff
      + constant_line_adm_imag * line_sin_volt_ang_diff) *
    line_orig_dest_volt_mag_prod)
#function 41
line_pow_dest_imag = (
    - constant_line_adm_total_imag * line_dest_volt_mag_sq + # ** 2.0 +
    (   constant_line_adm_imag * line_cos_volt_ang_diff
      + constant_line_adm_real * line_sin_volt_ang_diff) *
    line_orig_dest_volt_mag_prod)
#function 52-53
constant_line_curr_mag_max = tf.constant(line_curr_mag_max, dtype = tf.float32)
line_curr_orig_mag_max_viol = tf.maximum(
    tf.constant(0.0),
    (line_pow_orig_real**2.0 + line_pow_orig_imag**2.0 + hard_constr_tol)**0.5 -
    constant_line_curr_mag_max * line_orig_volt_mag)
#function 54
line_curr_dest_mag_max_viol = tf.maximum(
    tf.constant(0.0),
    (line_pow_dest_real**2.0 + line_pow_dest_imag**2.0 + hard_constr_tol)**0.5 -
    constant_line_curr_mag_max * line_dest_volt_mag)
 

#trnas aprt
constant_xfmr_orig_bus = tf.constant(xfmr_orig_bus)
constant_xfmr_dest_bus = tf.constant(xfmr_dest_bus)
xfmr_orig_volt_mag = tf.gather(bus_volt_mag, constant_xfmr_orig_bus)
xfmr_dest_volt_mag = tf.gather(bus_volt_mag, constant_xfmr_dest_bus)
xfmr_volt_ang_diff = tf.gather(bus_volt_ang, constant_xfmr_orig_bus) - tf.gather(bus_volt_ang, constant_xfmr_dest_bus) - tf.constant(xfmr_tap_ang, dtype = tf.float32)
xfmr_cos_volt_ang_diff = tf.cos(xfmr_volt_ang_diff)
xfmr_sin_volt_ang_diff = tf.sin(xfmr_volt_ang_diff)
#xfmr_orig_dest_volt_mag_prod = xfmr_orig_volt_mag * xfmr_dest_volt_mag
xfmr_orig_volt_mag_sq = xfmr_orig_volt_mag ** 2.0
xfmr_dest_volt_mag_sq = xfmr_dest_volt_mag ** 2.0
#function 42
constant_xfmr_adm_real = tf.constant(xfmr_adm_real, dtype = tf.float32)
constant_xfmr_adm_imag = tf.constant(xfmr_adm_imag, dtype = tf.float32)
constant_xfmr_tap_mag = tf.constant(xfmr_tap_mag, dtype = tf.float32)
constant_xfmr_adm_mag_real = tf.constant(xfmr_adm_mag_real, dtype = tf.float32)
constant_xfmr_adm_mag_imag = tf.constant(xfmr_adm_mag_imag, dtype = tf.float32)
xfmr_pow_orig_real = (
    (constant_xfmr_adm_real / constant_xfmr_tap_mag**2.0 + constant_xfmr_adm_mag_real) * xfmr_orig_volt_mag_sq +
    ( - constant_xfmr_adm_real / constant_xfmr_tap_mag * xfmr_cos_volt_ang_diff
      - constant_xfmr_adm_imag / constant_xfmr_tap_mag * xfmr_sin_volt_ang_diff) *
        xfmr_orig_volt_mag * xfmr_dest_volt_mag)
#function 43
xfmr_pow_orig_imag = (
    - (constant_xfmr_adm_imag / constant_xfmr_tap_mag**2.0 + constant_xfmr_adm_mag_imag) * xfmr_orig_volt_mag_sq +
    (   constant_xfmr_adm_imag / constant_xfmr_tap_mag * xfmr_cos_volt_ang_diff
      - constant_xfmr_adm_real / constant_xfmr_tap_mag * xfmr_sin_volt_ang_diff) *
        xfmr_orig_volt_mag * xfmr_dest_volt_mag)
#function 44
xfmr_pow_dest_real = (
    constant_xfmr_adm_real * xfmr_dest_volt_mag_sq +
    ( - constant_xfmr_adm_real / constant_xfmr_tap_mag * xfmr_cos_volt_ang_diff
      + constant_xfmr_adm_imag / constant_xfmr_tap_mag * xfmr_sin_volt_ang_diff) *
        xfmr_orig_volt_mag * xfmr_dest_volt_mag)
#function 45
xfmr_pow_dest_imag = (
    - constant_xfmr_adm_imag * xfmr_dest_volt_mag_sq +
    (   constant_xfmr_adm_imag / constant_xfmr_tap_mag * xfmr_cos_volt_ang_diff
      + constant_xfmr_adm_real / constant_xfmr_tap_mag * xfmr_sin_volt_ang_diff) *
        xfmr_orig_volt_mag * xfmr_dest_volt_mag)

#function 55-56    
constant_xfmr_pow_mag_max = tf.constant(xfmr_pow_mag_max, dtype = tf.float32)
xfmr_pow_orig_mag_max_viol = tf.maximum(
    tf.constant(0.0),
    (xfmr_pow_orig_real**2.0 + xfmr_pow_orig_imag**2.0 + hard_constr_tol)**0.5 -
    constant_xfmr_pow_mag_max)
#function 57
xfmr_pow_dest_mag_max_viol = tf.maximum(
    tf.constant(0.0),
    (xfmr_pow_dest_real**2.0 + xfmr_pow_dest_imag**2.0 + hard_constr_tol)**0.5 -
    constant_xfmr_pow_mag_max)

#power part
bus_load_pow_real = tf.constant(bus_load_const_pow_real, dtype = tf.float32)
bus_load_pow_imag = tf.constant(bus_load_const_pow_imag, dtype = tf.float32)
#function 46 bFS v^2
constant_bus_fxsh_adm_real = tf.constant(bus_fxsh_adm_real, dtype = tf.float32)
bus_fxsh_pow_real = constant_bus_fxsh_adm_real  * (bus_volt_mag ** 2.0)
#function 49 bFS v^2
constant_bus_fxsh_adm_imag = tf.constant(bus_fxsh_adm_imag, dtype = tf.float32)
bus_fxsh_pow_imag = - constant_bus_fxsh_adm_imag * (bus_volt_mag ** 2.0)
#function 49 bCS v^2
bus_swsh_pow_imag = -bus_swsh_adm_imag * bus_volt_mag**2.0

#function 46
bus_pow_balance_real_viol = tf.abs((
    tf.matmul(tf.constant(bus_gen_matrix), gen_pow_real) -
    bus_load_pow_real -
    bus_fxsh_pow_real -
    tf.matmul(tf.constant(bus_line_orig_matrix),line_pow_orig_real) -
    tf.matmul(tf.constant(bus_line_dest_matrix),line_pow_dest_real) -
    tf.matmul(tf.constant(bus_xfmr_orig_matrix),xfmr_pow_orig_real) -
    tf.matmul(tf.constant(bus_xfmr_dest_matrix),xfmr_pow_dest_real)))
#function 49
bus_pow_balance_imag_viol = tf.abs((
    tf.matmul(tf.constant(bus_gen_matrix),gen_pow_imag) -
    bus_load_pow_imag -
    bus_fxsh_pow_imag -
    bus_swsh_pow_imag -
    tf.matmul(tf.constant(bus_line_orig_matrix),line_pow_orig_imag) -
    tf.matmul(tf.constant(bus_line_dest_matrix),line_pow_dest_imag) -
    tf.matmul(tf.constant(bus_xfmr_orig_matrix),xfmr_pow_orig_imag) -
    tf.matmul(tf.constant(bus_xfmr_dest_matrix),xfmr_pow_dest_imag)))


#function 6-19
base_penalty = (
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
i = 0
for i in range(len(ctgs)):
    ctg_current = ctg_map[ctg_label[i]]
    
    #genpart out of service to set the minmax as zeros
    gens_out_of_service = set(gen_out_of_service) | set(ctg_gens_out[ctg_current])
    ctg_gen_out_of_service = sorted(list(gens_out_of_service))

    #todo check none situation
    ctg_power_multi_vector = np.ones(num_gen)
    for index in ctg_gen_out_of_service:
        ctg_power_multi_vector[index] = 0.
    if i == 0:
        ctg_power_multi_matrix = np.transpose([ctg_power_multi_vector])
    else:
        ctg_power_multi_matrix = np.column_stack((ctg_power_multi_matrix,ctg_power_multi_vector))

    #todo check gradient 
    ctg_line_multi_vector = np.ones(num_line)
    for index in ctg_lines_out[ctg_current]:
        ctg_line_multi_vector[index] = 0.
    if i == 0:
        ctg_line_multi_matrix =  np.transpose([ctg_line_multi_vector])
    else:
        ctg_line_multi_matrix = np.column_stack((ctg_line_multi_matrix,ctg_line_multi_vector))
    
    #todo check none situation
    ctg_xfmr_multi_vector = np.ones(num_xfmr)
    for index in ctg_xfmrs_out[ctg_current]:
        ctg_xfmr_multi_vector[index] = 0.
    if i == 0:
        ctg_xfmr_multi_matrix = np.transpose([ctg_xfmr_multi_vector])
    else:
        ctg_xfmr_multi_matrix = np.column_stack((ctg_xfmr_multi_matrix,ctg_xfmr_multi_vector))
    
    i += 0
        
        
        
#ctg_gen p,q        
ctg_gen_pow_real = gen_pow_real * tf.constant(ctg_power_multi_matrix, dtype = tf.float32)
ctg_gen_pow_imag = gen_pow_imag * tf.constant(ctg_power_multi_matrix, dtype = tf.float32)


#line part
ctg_line_pow_orig_real = line_pow_orig_real * tf.constant(ctg_line_multi_matrix, dtype = tf.float32)
ctg_line_pow_orig_imag = line_pow_orig_imag * tf.constant(ctg_line_multi_matrix, dtype = tf.float32)
ctg_line_pow_dest_real = line_pow_dest_real * tf.constant(ctg_line_multi_matrix, dtype = tf.float32)
ctg_line_pow_dest_imag = line_pow_dest_imag * tf.constant(ctg_line_multi_matrix, dtype = tf.float32)

#function 78
constant_ctg_line_curr_mag_max = tf.constant(ctg_line_curr_mag_max, dtype = tf.float32)
ctg_line_curr_orig_mag_max_viol = tf.maximum(
    tf.constant(0.0),
    (ctg_line_pow_orig_real**2.0 + ctg_line_pow_orig_imag**2.0 + hard_constr_tol)**0.5 -
     constant_ctg_line_curr_mag_max * line_orig_volt_mag)
#function 80
ctg_line_curr_dest_mag_max_viol = tf.maximum(
    tf.constant(0.0),
    (ctg_line_pow_dest_real**2.0 + ctg_line_pow_dest_imag**2.0 + hard_constr_tol)**0.5 -
    constant_ctg_line_curr_mag_max * line_dest_volt_mag)



#trnas aprt
ctg_xfmr_pow_orig_real = xfmr_pow_orig_real * tf.constant(ctg_xfmr_multi_matrix, dtype = tf.float32)
ctg_xfmr_pow_orig_imag = xfmr_pow_orig_imag * tf.constant(ctg_xfmr_multi_matrix, dtype = tf.float32)
ctg_xfmr_pow_dest_real = xfmr_pow_dest_real * tf.constant(ctg_xfmr_multi_matrix, dtype = tf.float32)
ctg_xfmr_pow_dest_imag = xfmr_pow_dest_imag * tf.constant(ctg_xfmr_multi_matrix, dtype = tf.float32)  

#function 81 
constant_ctg_xfmr_pow_mag_max = tf.constant(ctg_xfmr_pow_mag_max, dtype = tf.float32)
ctg_xfmr_pow_orig_mag_max_viol = tf.maximum(
    tf.constant(0.0),
    (ctg_xfmr_pow_orig_real**2.0 + ctg_xfmr_pow_orig_imag**2.0 + hard_constr_tol)**0.5 -
    constant_ctg_xfmr_pow_mag_max)
#function 83
ctg_xfmr_pow_dest_mag_max_viol = tf.maximum(
    tf.constant(0.0),
    (ctg_xfmr_pow_dest_real**2.0 + ctg_xfmr_pow_dest_imag**2.0 + hard_constr_tol)**0.5 -
    constant_ctg_xfmr_pow_mag_max)

       
#power part
#function 72,75
ctg_bus_pow_balance_real_viol = tf.abs(
    tf.matmul(tf.constant(bus_gen_matrix),ctg_gen_pow_real) -
    bus_load_pow_real -
    bus_fxsh_pow_real -
    tf.matmul(tf.constant(bus_line_orig_matrix),ctg_line_pow_orig_real) -
    tf.matmul(tf.constant(bus_line_dest_matrix),ctg_line_pow_dest_real) -
    tf.matmul(tf.constant(bus_xfmr_orig_matrix),ctg_xfmr_pow_orig_real) -
    tf.matmul(tf.constant(bus_xfmr_dest_matrix),ctg_xfmr_pow_dest_real))
ctg_bus_pow_balance_imag_viol = tf.abs(
    tf.matmul(tf.constant(bus_gen_matrix),ctg_gen_pow_imag) -
    bus_load_pow_imag -
    bus_fxsh_pow_imag -
    bus_swsh_pow_imag -
    tf.matmul(tf.constant(bus_line_orig_matrix),ctg_line_pow_orig_imag) -
    tf.matmul(tf.constant(bus_line_dest_matrix),ctg_line_pow_dest_imag) -
    tf.matmul(tf.constant(bus_xfmr_orig_matrix),ctg_xfmr_pow_orig_imag) -
    tf.matmul(tf.constant(bus_xfmr_dest_matrix),ctg_xfmr_pow_dest_imag))


#function 6-19
ctg_penalty =  (
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
            
#ctg_penalty = 




#obj_cons
con1 = tf.reduce_sum(tf.maximum(tf.constant(0.), bus_volt_mag - tf.constant(bus_volt_mag_max, dtype = tf.float32)))  
con2 = tf.reduce_sum(tf.maximum(tf.constant(0.), -bus_volt_mag + tf.constant(bus_volt_mag_min, dtype = tf.float32)))
con3 = tf.reduce_sum(tf.maximum(tf.constant(0.), bus_swsh_adm_imag - tf.constant(bus_swsh_adm_imag_max, dtype = tf.float32)))
con4 = tf.reduce_sum(tf.maximum(tf.constant(0.), -bus_swsh_adm_imag + tf.constant(bus_swsh_adm_imag_min, dtype = tf.float32)))
con5 = tf.reduce_sum(tf.maximum(tf.constant(0.), gen_pow_real - tf.constant(gen_pow_real_max, dtype = tf.float32)))
con6 = tf.reduce_sum(tf.maximum(tf.constant(0.), -gen_pow_real + tf.constant(gen_pow_real_min, dtype = tf.float32)))
con7 = tf.reduce_sum(tf.maximum(tf.constant(0.), gen_pow_imag - tf.constant(gen_pow_imag_max, dtype = tf.float32)))
con8 = tf.reduce_sum(tf.maximum(tf.constant(0.), -gen_pow_imag + tf.constant(gen_pow_imag_min, dtype = tf.float32)))
obj_cons = con1 + con2 + con3 + con4 + con5 + con6 + con7 + con8


#sig_cons
val = (tf.reduce_sum(line_curr_orig_mag_max_viol) + tf.reduce_sum(line_curr_dest_mag_max_viol)
                + tf.reduce_sum(xfmr_pow_orig_mag_max_viol) + tf.reduce_sum(xfmr_pow_dest_mag_max_viol)
                + tf.reduce_sum(bus_pow_balance_real_viol) + tf.reduce_sum(bus_pow_balance_imag_viol))


constant_base_case_penalty_weight = tf.constant(base_case_penalty_weight)
constant_ctg_case_penalty_weight = tf.constant(1-base_case_penalty_weight)
constant_num_k = tf.constant(max(1.0, float(num_ctg)))

#function 1
penalty =  constant_base_case_penalty_weight * base_penalty + constant_ctg_case_penalty_weight/constant_num_k * ctg_penalty
obj = cost + penalty
print('finish graphing')

#do calculation
class MyProblem(tfco.ConstrainedMinimizationProblem):

  def __init__(self, obj, obj_cons, val):
    '''
    initial all para of object function
    
    '''
    self.obj_ = obj
    self.cons_ = tf.constant(100000000 * base_mva) * (obj_cons + val)
    self.test = tf.Variable(0.)
     

  @property
  def objective(self):
    #todo need add other penalty

    #100* num_bus * 20
    #self.test = 

    return self.obj_ + self.cons_


  @property
  def constraints(self):

    return self.cons_


  @property
  def proxy_constraints(self):

    return self.cons_
    




#main start
problem = MyProblem(obj, obj_cons, val)
itertime = 10000
print('now')
with tf.Session() as session:
    '''
    https://git.codingcafe.org/Mirrors/tensorflow/tensorflow/commit/ff15c81e2b92ef8fb47bb15790cffd18377a4ef2?expanded=1
    AdditiveExternalRegretOptimizer
    AdditiveSwapRegretOptimizer
    MultiplicativeSwapRegretOptimizer
    '''
    #initial
    nowtmie1 = time.time()
    print('initial optimizer')
    optimizer = tfco.AdditiveExternalRegretOptimizer(
            optimizer=tf.train.AdagradOptimizer(learning_rate = 0.01))
    nowtmie2 = time.time()
    print(nowtmie2-nowtmie1)    
    print('initial train')
    train_op = optimizer.minimize(problem)#
    nowtmie3 = time.time()
    print(nowtmie3-nowtmie2) 
    print('initial variables')
    session.run(tf.global_variables_initializer())
    nowtmie4 = time.time()
    print(nowtmie4-nowtmie3) 
    print('start training')
    # 2 methods 1st:iter time ;  2nd:runing time
    i = 1
    while True:

        session.run(train_op)
        if i % 5000 == 0:
            print('iter:'+str(i))
        i += 1
        if time.time() - start_time > 600 - 30 : #5min #change to str write
            print('iter:'+str(i))
            volt_mag = list(session.run(tf.transpose(bus_volt_mag))[0])
            volt_ang = list(session.run(tf.transpose(bus_volt_ang))[0] / (math.pi/180))
            gen_real = list(session.run(tf.transpose(gen_pow_real))[0] * base_mva)
            gen_img = list(session.run(tf.transpose(gen_pow_imag))[0] * base_mva)
            swsh = list(session.run(tf.transpose(bus_swsh_adm_imag))[0] * base_mva)
            
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

        






