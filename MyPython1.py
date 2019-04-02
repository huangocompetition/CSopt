"""
Created on Fri Mar  8 15:21:22 2019
@author: Zhao Huan
Step 1: read para from data
Step 2: build function
Step 3: solve by tensorflow
"""

import numpy as np
from scipy import sparse as sp
import math
import tensorflow as tf
import time
import csv
import argparse
import data

start_time = time.time()

tfco = tf.contrib.constrained_optimization

#soft constraint penalty parameters
penalty_block_pow_real_max = [2.0, 50.0] # MW. when converted to p.u., this is overline_sigma_p in the formulation
penalty_block_pow_real_coeff = [1000.0, 5000.0, 1000000.0] # USD/MW-h. when converted USD/p.u.-h this is lambda_p in the formulation
penalty_block_pow_imag_max = [2.0, 50.0] # MVar. when converted to p.u., this is overline_sigma_q in the formulation
penalty_block_pow_imag_coeff = [1000.0, 5000.0, 1000000.0] # USD/MVar-h. when converted USD/p.u.-h this is lambda_q in the formulation
penalty_block_pow_abs_max = [2.0, 50.0] # MVA. when converted to p.u., this is overline_sigma_s in the formulation
penalty_block_pow_abs_coeff = [1000.0, 5000.0, 1000000.0] # USD/MWA-h. when converted USD/p.u.-h this is lambda_s in the formulation

#weight on base case in objective
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

raw = args.raw
rop = args.rop
con = args.con
inl = args.inl



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
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config = config) as session:
#with tf.Session() as session:
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
            


        






