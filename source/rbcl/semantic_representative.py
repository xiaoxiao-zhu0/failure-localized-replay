"""Frozen-semantic prototype representative memory for causal ER-ACE."""
from __future__ import annotations
import copy
from pathlib import Path
import hashlib
import math
import random
import time
from collections import Counter, deque
from statistics import NormalDist
import torch
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18
from avalanche.models.utils import avalanche_forward
from avalanche.training.utils import get_last_fc_layer
from .causal_er_ace import CausalERACE
from .calibration_audit import (
 empty_dual_head_audit,
 summarize_dual_head_audit,
 update_dual_head_audit,
)
from .semantic_anchor import _CIFAR_STATS, _IMAGENET_MEAN, _IMAGENET_STD
from .tirp import constrained_probabilities, expected_uniform_class_coverage, sample_without_replacement

class _TIRPPolicy(torch.nn.Module):
 def __init__(self):
  super().__init__(); self.net=torch.nn.Sequential(torch.nn.Linear(35,32),torch.nn.Tanh(),torch.nn.Linear(32,1))
 def forward(self,x): return self.net(x).squeeze(-1)

class SemanticRepresentativeERACE(CausalERACE):
 def __init__(self,*args,dataset_family:str,with_reservoir:bool,semantic_capacity:int|None=None,reservoir_capacity:int|None=None,coverage_first:bool=False,**kwargs):
  super().__init__(*args,memory_policy='hybrid' if with_reservoir else 'class_balanced',**kwargs)
  self.dataset_family=dataset_family; self.with_reservoir=with_reservoir; self.semantic_capacity=int(semantic_capacity if semantic_capacity is not None else (self.mem_size//2 if with_reservoir else self.mem_size)); self.reservoir_capacity=int(reservoir_capacity if reservoir_capacity is not None else self.mem_size-self.semantic_capacity)
  if self.semantic_capacity+self.reservoir_capacity!=self.mem_size: raise ValueError('semantic/reservoir capacities must sum to mem_size')
  if coverage_first and not with_reservoir: raise ValueError('coverage_first requires hybrid semantic/reservoir memory')
  self.coverage_first=bool(coverage_first)
  self.encoder=resnet18(weights=ResNet18_Weights.DEFAULT); self.encoder.fc=torch.nn.Identity(); self.encoder.to(self.device).eval()
  for p in self.encoder.parameters(): p.requires_grad_(False)
  self.proto_sum={}; self.proto_count={}; self.proto_features={}; self._memory_arrival={}
 @staticmethod
 def _memory_key(item,label):
  payload=item.detach().cpu().contiguous().numpy().tobytes()
  return f'{int(label)}:{hashlib.sha1(payload).hexdigest()}'
 def _update_memory(self,x,y,task_ids):
  super()._update_memory(x,y,task_ids)
  clock=max(self._seen_samples,self._hybrid_seen)
  present={self._memory_key(item,label) for item,label in zip(self._memory_x,self._memory_y)}
  self._memory_arrival={key:self._memory_arrival.get(key,clock) for key in present}
 def _active_semantic_capacity(self):
  if not self.coverage_first: return self.semantic_capacity
  return min(self.mem_size,max(self.semantic_capacity,len(self._class_memory)))
 def _active_reservoir_capacity(self):
  return self.mem_size-self._active_semantic_capacity()
 def _class_capacities(self):
  labels=sorted(self._class_memory); base,rem=divmod(self._active_semantic_capacity(),len(labels)) if labels else (0,0); return {y:base+int(i<rem) for i,y in enumerate(labels)}
 def _update_hybrid_reservoir(self,x,y,task_ids):
  cap=self._active_reservoir_capacity()
  if len(self._hybrid_memory_x)>cap:
   keep=sorted(self._rng.sample(range(len(self._hybrid_memory_x)),cap)) if cap else []
   self._hybrid_memory_x=[self._hybrid_memory_x[i] for i in keep]; self._hybrid_memory_y=[self._hybrid_memory_y[i] for i in keep]; self._hybrid_memory_tid=[self._hybrid_memory_tid[i] for i in keep]
  for sx,sy,st in zip(x,y,task_ids):
   self._hybrid_seen+=1; item=sx.detach().cpu().clone(); label=int(sy); tid=int(st)
   if cap<=0: continue
   if len(self._hybrid_memory_x)<cap: self._hybrid_memory_x.append(item); self._hybrid_memory_y.append(label); self._hybrid_memory_tid.append(tid); continue
   j=self._rng.randrange(self._hybrid_seen)
   if j<cap: self._hybrid_memory_x[j]=item; self._hybrid_memory_y[j]=label; self._hybrid_memory_tid[j]=tid
 @torch.no_grad()
 def _features(self,x):
  mean,std=_CIFAR_STATS[self.dataset_family]; mean=torch.tensor(mean,device=self.device).view(1,3,1,1); std=torch.tensor(std,device=self.device).view(1,3,1,1)
  im=torch.tensor(_IMAGENET_MEAN,device=self.device).view(1,3,1,1); iss=torch.tensor(_IMAGENET_STD,device=self.device).view(1,3,1,1)
  x=(x.to(self.device)*std+mean).clamp(0,1); x=F.interpolate(x,size=(224,224),mode='bilinear',align_corners=False)
  return self.encoder((x-im)/iss).detach().cpu()
 def _rebalance_semantic(self):
  caps=self._class_capacities()
  for y,items in self._class_memory.items():
   cap=caps[y]
   if len(items)>cap:
    proto=self.proto_sum[y]/self.proto_count[y]
    keep=torch.argsort(torch.cdist(self.proto_features[y],proto.unsqueeze(0)).squeeze(1))[:cap].tolist()
    self._class_memory[y]=[items[i] for i in keep]; self.proto_features[y]=self.proto_features[y][keep]
 def _update_class_balanced(self,x,y,task_ids):
  zs=self._features(x)
  for sx,sy,st,z in zip(x,y,task_ids,zs):
   label=int(sy); self._seen_samples+=1; self._class_seen[label]=self._class_seen.get(label,0)+1
   if label not in self._class_memory:
    self._class_memory[label]=[]; self.proto_features[label]=torch.empty((0,z.numel())); self.proto_sum[label]=torch.zeros_like(z); self.proto_count[label]=0; self._rebalance_semantic()
   self.proto_sum[label]+=z; self.proto_count[label]+=1; cap=self._class_capacities()[label]
   if cap<=0: continue
   item=(sx.detach().cpu().clone(),int(st)); items=self._class_memory[label]; feats=self.proto_features[label]
   if len(items)<cap: items.append(item); self.proto_features[label]=torch.cat([feats,z.unsqueeze(0)]); continue
   proto=self.proto_sum[label]/self.proto_count[label]; old=torch.norm(feats-proto,dim=1); candidate=torch.norm(z-proto)
   worst=int(old.argmax())
   if candidate<old[worst]: items[worst]=item; feats[worst]=z; self.proto_features[label]=feats
  self._refresh_flat_memory()
 def rbcl_summary(self):
  mean,std=_CIFAR_STATS[self.dataset_family]
  r=super().rbcl_summary(); r['semantic_representative_memory']={'enabled':True,'dataset_family':self.dataset_family,'input_normalization_mean':list(mean),'input_normalization_std':list(std),'frozen_imagenet_resnet18':True,'with_reservoir':self.with_reservoir,'semantic_capacity':self.semantic_capacity,'reservoir_capacity':self.reservoir_capacity,'coverage_first':self.coverage_first,'active_semantic_capacity':self._active_semantic_capacity(),'active_reservoir_capacity':self._active_reservoir_capacity(),'coverage_floor_is_one_per_seen_class_when_feasible':self.coverage_first,'policy_uses_task_loss_gradient_or_validation':False}; return r


class ClassReplayVulnerabilityAuditERACE(SemanticRepresentativeERACE):
 """Audit lagged class-level replay vulnerability without changing training."""

 def __init__(self,*args,replay_vulnerability_ema_decay:float=.99,**kwargs):
  super().__init__(*args,**kwargs)
  if not 0.0<=replay_vulnerability_ema_decay<1.0: raise ValueError('replay_vulnerability_ema_decay must be in [0,1)')
  self.replay_vulnerability_ema_decay=float(replay_vulnerability_ema_decay)
  self._replay_vulnerability_classes={}; self._replay_vulnerability_audit_calls=0; self._replay_vulnerability_sample_count=0
  self._replay_vulnerability_predictive={key:self._empty_pair_stats() for key in ('ce','margin','negative_margin_fraction')}
  self._replay_vulnerability_history=[]
 @staticmethod
 def _empty_pair_stats():
  return {'count':0,'x_sum':0.0,'y_sum':0.0,'x_sq_sum':0.0,'y_sq_sum':0.0,'xy_sum':0.0}
 @staticmethod
 def _update_pair_stats(stats,x,y):
  x=float(x); y=float(y); stats['count']+=1; stats['x_sum']+=x; stats['y_sum']+=y; stats['x_sq_sum']+=x*x; stats['y_sq_sum']+=y*y; stats['xy_sum']+=x*y
 @staticmethod
 def _pair_pearson(stats):
  n=stats['count']
  if n<2: return 0.0
  first=max(0.0,n*stats['x_sq_sum']-stats['x_sum']**2); second=max(0.0,n*stats['y_sq_sum']-stats['y_sum']**2); denominator=math.sqrt(first*second)
  return (n*stats['xy_sum']-stats['x_sum']*stats['y_sum'])/denominator if denominator>1e-12 else 0.0
 @staticmethod
 def _empty_class_vulnerability():
  return {'replay_count':0,'batch_update_count':0,'ce_sum':0.0,'margin_sum':0.0,'negative_margin_count':0,'ema_ce':None,'ema_margin':None,'ema_negative_margin_fraction':None}
 @classmethod
 def _snapshot_class_vulnerability(cls,state):
  count=max(1,state['replay_count'])
  return {'replay_count':state['replay_count'],'batch_update_count':state['batch_update_count'],'mean_replay_ce':state['ce_sum']/count,'mean_replay_margin':state['margin_sum']/count,'negative_margin_fraction':state['negative_margin_count']/count,'lagged_ema_ce':state['ema_ce'],'lagged_ema_margin':state['ema_margin'],'lagged_ema_negative_margin_fraction':state['ema_negative_margin_fraction']}
 def _vulnerability_snapshot(self):
  return {'audit_calls':self._replay_vulnerability_audit_calls,'replay_sample_count':self._replay_vulnerability_sample_count,'predictive_pair_count':self._replay_vulnerability_predictive['ce']['count'],'lagged_ema_vs_next_observation_pearson':{key:self._pair_pearson(stats) for key,stats in self._replay_vulnerability_predictive.items()},'classes':{str(label):self._snapshot_class_vulnerability(state) for label,state in sorted(self._replay_vulnerability_classes.items())}}
 def _after_training_exp(self,**kwargs):
  self._replay_vulnerability_history.append(self._vulnerability_snapshot()); super()._after_training_exp(**kwargs)
 def _add_auxiliary_loss(self,current_x,current_y,current_tid,replay,replay_output=None):
  super()._add_auxiliary_loss(current_x,current_y,current_tid,replay,replay_output)
  if replay is None or replay_output is None: return
  with torch.no_grad():
   device_labels=replay[1].detach(); output=replay_output.detach(); losses=F.cross_entropy(output,device_labels,reduction='none'); rows=torch.arange(output.shape[0],device=output.device); correct=output[rows,device_labels]; rivals=output.clone(); rivals[rows,device_labels]=-torch.inf; margins=correct-rivals.max(dim=1).values
   labels=device_labels.cpu(); losses=losses.cpu(); margins=margins.cpu()
   self._replay_vulnerability_audit_calls+=1; self._replay_vulnerability_sample_count+=int(labels.numel())
   for label in torch.unique(labels).tolist():
    key=int(label); mask=labels.eq(key); count=int(mask.sum()); batch_ce=float(losses[mask].mean()); batch_margin=float(margins[mask].mean()); batch_negative=float(margins[mask].le(0).float().mean()); state=self._replay_vulnerability_classes.setdefault(key,self._empty_class_vulnerability())
    if state['batch_update_count']>0:
     self._update_pair_stats(self._replay_vulnerability_predictive['ce'],state['ema_ce'],batch_ce); self._update_pair_stats(self._replay_vulnerability_predictive['margin'],state['ema_margin'],batch_margin); self._update_pair_stats(self._replay_vulnerability_predictive['negative_margin_fraction'],state['ema_negative_margin_fraction'],batch_negative)
    decay=self.replay_vulnerability_ema_decay
    state['replay_count']+=count; state['batch_update_count']+=1; state['ce_sum']+=float(losses[mask].sum()); state['margin_sum']+=float(margins[mask].sum()); state['negative_margin_count']+=int(margins[mask].le(0).sum())
    state['ema_ce']=batch_ce if state['ema_ce'] is None else decay*state['ema_ce']+(1.0-decay)*batch_ce
    state['ema_margin']=batch_margin if state['ema_margin'] is None else decay*state['ema_margin']+(1.0-decay)*batch_margin
    state['ema_negative_margin_fraction']=batch_negative if state['ema_negative_margin_fraction'] is None else decay*state['ema_negative_margin_fraction']+(1.0-decay)*batch_negative
 def rbcl_summary(self):
  r=super().rbcl_summary(); r['class_replay_vulnerability_audit']={'enabled':True,'audit_only':True,'uniform_replay_parent':'semantic_proto_hybrid_75_25','primary_signal':'lagged_ema_replay_ce','secondary_signals':['lagged_ema_replay_margin','lagged_ema_negative_margin_fraction'],'ema_decay':self.replay_vulnerability_ema_decay,'sampling_modified_by_audit':False,'training_objective_modified':False,'ema_is_read_before_current_batch_update':True,'uses_arrived_replay_labels_and_logits_only':True,'uses_task_id_gradient_validation_pss_or_future_data':False,'history':self._replay_vulnerability_history,'global':self._vulnerability_snapshot()}; return r


class SelfReferencedReplayDeteriorationAuditERACE(SemanticRepresentativeERACE):
 """Audit same-item replay deterioration without changing training."""

 def __init__(self,*args,self_ref_ema_decay:float=.99,**kwargs):
  super().__init__(*args,**kwargs)
  if not 0.0<=self_ref_ema_decay<1.0: raise ValueError('self_ref_ema_decay must be in [0,1)')
  self.self_ref_ema_decay=float(self_ref_ema_decay); self._srrd_memory_keys=[]; self._srrd_last_keys=[]; self._srrd_items={}; self._srrd_classes={}; self._srrd_history=[]; self._srrd_audit_calls=0; self._srrd_replay_samples=0
 @staticmethod
 def _empty_item_state(label):
  return {'label':int(label),'observation_count':0,'ema_ce':None,'best_ema_ce':None,'ema_margin':None,'best_ema_margin':None}
 @staticmethod
 def _empty_class_state():
  return {'item_observation_count':0,'repeat_event_count':0,'ce_deterioration_sum':0.0,'margin_deterioration_sum':0.0,'positive_ce_event_count':0,'positive_margin_event_count':0,'ema_ce_deterioration':None,'ema_margin_deterioration':None,'max_ce_deterioration':0.0,'max_margin_deterioration':0.0}
 @staticmethod
 def _ema(old,value,decay):
  return float(value) if old is None else float(decay)*float(old)+(1.0-float(decay))*float(value)
 @classmethod
 def _record_class_deterioration(cls,state,ce_deterioration,margin_deterioration,decay):
  ce=float(ce_deterioration); margin=float(margin_deterioration); state['repeat_event_count']+=1; state['ce_deterioration_sum']+=ce; state['margin_deterioration_sum']+=margin; state['positive_ce_event_count']+=int(ce>0.0); state['positive_margin_event_count']+=int(margin>0.0); state['ema_ce_deterioration']=cls._ema(state['ema_ce_deterioration'],ce,decay); state['ema_margin_deterioration']=cls._ema(state['ema_margin_deterioration'],margin,decay); state['max_ce_deterioration']=max(state['max_ce_deterioration'],ce); state['max_margin_deterioration']=max(state['max_margin_deterioration'],margin)
 @staticmethod
 def _snapshot_class_deterioration(state):
  count=max(1,state['repeat_event_count'])
  return {'item_observation_count':state['item_observation_count'],'repeat_event_count':state['repeat_event_count'],'mean_self_referenced_ce_deterioration':state['ce_deterioration_sum']/count,'mean_self_referenced_margin_deterioration':state['margin_deterioration_sum']/count,'positive_ce_deterioration_fraction':state['positive_ce_event_count']/count,'positive_margin_deterioration_fraction':state['positive_margin_event_count']/count,'ema_self_referenced_ce_deterioration':state['ema_ce_deterioration'],'ema_self_referenced_margin_deterioration':state['ema_margin_deterioration'],'max_self_referenced_ce_deterioration':state['max_ce_deterioration'],'max_self_referenced_margin_deterioration':state['max_margin_deterioration']}
 def _srrd_snapshot(self):
  repeated=sum(int(state['observation_count']>=2) for state in self._srrd_items.values()); events=sum(state['repeat_event_count'] for state in self._srrd_classes.values())
  return {'audit_calls':self._srrd_audit_calls,'replay_sample_count':self._srrd_replay_samples,'active_item_count':len(self._srrd_items),'active_repeated_item_count':repeated,'repeat_event_count':events,'classes':{str(label):self._snapshot_class_deterioration(state) for label,state in sorted(self._srrd_classes.items()) if state['repeat_event_count']>0}}
 def _refresh_srrd_memory_keys(self):
  self._srrd_memory_keys=[self._memory_key(item,label) for item,label in zip(self._memory_x,self._memory_y)]; present=set(self._srrd_memory_keys); self._srrd_items={key:state for key,state in self._srrd_items.items() if key in present}
 def _after_memory_update(self,current_x,current_y,current_tid):
  self._refresh_srrd_memory_keys(); super()._after_memory_update(current_x,current_y,current_tid)
 def _sample_memory(self):
  replay=super()._sample_memory()
  if replay is None:
   self._srrd_last_keys=[]; return None
  if len(self._srrd_memory_keys)!=len(self._memory_x): self._refresh_srrd_memory_keys()
  self._srrd_last_keys=[self._srrd_memory_keys[index] for index in self._last_replay_memory_indices]
  return replay
 def _after_training_exp(self,**kwargs):
  self._srrd_history.append(self._srrd_snapshot()); super()._after_training_exp(**kwargs)
 def _add_auxiliary_loss(self,current_x,current_y,current_tid,replay,replay_output=None):
  super()._add_auxiliary_loss(current_x,current_y,current_tid,replay,replay_output)
  if replay is None or replay_output is None or len(self._srrd_last_keys)!=int(replay[1].numel()): return
  with torch.no_grad():
   device_labels=replay[1].detach(); output=replay_output.detach(); losses=F.cross_entropy(output,device_labels,reduction='none'); rows=torch.arange(output.shape[0],device=output.device); correct=output[rows,device_labels]; rivals=output.clone(); rivals[rows,device_labels]=-torch.inf; margins=correct-rivals.max(dim=1).values
   labels=device_labels.cpu(); losses=losses.cpu(); margins=margins.cpu(); grouped={}
   for position,key in enumerate(self._srrd_last_keys): grouped.setdefault(key,[]).append(position)
   self._srrd_audit_calls+=1; self._srrd_replay_samples+=int(labels.numel())
   for key,positions in grouped.items():
    selected=torch.tensor(positions,dtype=torch.long); label=int(labels[positions[0]]); current_ce=float(losses[selected].mean()); current_margin=float(margins[selected].mean()); item=self._srrd_items.setdefault(key,self._empty_item_state(label)); class_state=self._srrd_classes.setdefault(label,self._empty_class_state()); class_state['item_observation_count']+=1
    if item['observation_count']>0:
     ce_deterioration=max(0.0,current_ce-float(item['best_ema_ce'])); margin_deterioration=max(0.0,float(item['best_ema_margin'])-current_margin); self._record_class_deterioration(class_state,ce_deterioration,margin_deterioration,self.self_ref_ema_decay)
    item['observation_count']+=1; item['ema_ce']=self._ema(item['ema_ce'],current_ce,self.self_ref_ema_decay); item['ema_margin']=self._ema(item['ema_margin'],current_margin,self.self_ref_ema_decay); item['best_ema_ce']=item['ema_ce'] if item['best_ema_ce'] is None else min(float(item['best_ema_ce']),float(item['ema_ce'])); item['best_ema_margin']=item['ema_margin'] if item['best_ema_margin'] is None else max(float(item['best_ema_margin']),float(item['ema_margin']))
 def rbcl_summary(self):
  r=super().rbcl_summary(); r['self_referenced_replay_deterioration_audit']={'enabled':True,'audit_only':True,'uniform_replay_parent':'semantic_proto_hybrid_75_25','primary_signal':'class_ema_same_item_ce_deterioration_from_historical_best','secondary_signal':'class_ema_same_item_margin_deterioration_from_historical_best','ema_decay':self.self_ref_ema_decay,'same_sample_identity_uses_memory_content_hash':True,'evicted_raw_samples_are_not_retained':True,'sampling_modified_by_audit':False,'training_objective_modified':False,'uses_absolute_replay_difficulty_as_primary_signal':False,'uses_historical_logits_or_margins_as_loss_targets':False,'uses_task_id_gradient_validation_pss_or_future_data':False,'history':self._srrd_history,'global':self._srrd_snapshot()}; return r


class PersistentSelfReferencedReplayDeteriorationAuditERACE(SelfReferencedReplayDeteriorationAuditERACE):
 """Audit how consistently replayed items deteriorate, without intervention."""

 @staticmethod
 def _persistent_signal_from_snapshot(snapshot):
  return float(snapshot['positive_ce_deterioration_fraction'])
 def rbcl_summary(self):
  r=super().rbcl_summary(); audit=r.pop('self_referenced_replay_deterioration_audit'); audit.update({'primary_signal':'class_positive_same_item_ce_deterioration_event_fraction','secondary_signal':'class_lifetime_mean_same_item_ce_deterioration','event_consistency_not_recent_magnitude_is_primary':True,'d92_gate_thresholds_reused_without_relaxation':True}); r['persistent_self_referenced_replay_deterioration_audit']=audit; return r


class PersistentSRRDDebtSwapERACE(PersistentSelfReferencedReplayDeteriorationAuditERACE):
 """Apply bounded replay swaps from lagged persistent deterioration debt."""

 def __init__(self,*args,max_swaps_per_batch:int=4,force_neutral_signal:bool=False,**kwargs):
  super().__init__(*args,**kwargs)
  if max_swaps_per_batch<0: raise ValueError('max_swaps_per_batch must be nonnegative')
  self.max_swaps_per_batch=int(max_swaps_per_batch); self.force_neutral_signal=bool(force_neutral_signal); self._persistent_debt={}; self._persistent_scores={}; self._persistent_replay_calls=0; self._persistent_swap_count=0; self._persistent_batches_with_swaps=0; self._persistent_max_abs_debt=0.0; self._persistent_target_score_sum=0.0; self._persistent_source_score_sum=0.0; self._persistent_last_swap_records=[]
 @staticmethod
 def _risk_target_deltas(labels,class_scores,count):
  counts={}
  for label in labels: counts[int(label)]=counts.get(int(label),0)+1
  total=max(1,len(labels)); weighted={label:(amount/total)*max(float(class_scores.get(label,1.0)),1e-12) for label,amount in counts.items()}; normalizer=sum(weighted.values())
  return {label:count*(weighted[label]/max(normalizer,1e-12)-counts[label]/total) for label in counts}
 def _class_persistence_scores(self):
  labels=sorted(set(int(label) for label in self._memory_y))
  if self.force_neutral_signal: return {label:1.0 for label in labels}
  observed={}
  for label in labels:
   state=self._srrd_classes.get(label)
   if state is not None and state['repeat_event_count']>0: observed[label]=state['positive_ce_event_count']/state['repeat_event_count']
  neutral=sum(observed.values())/len(observed) if observed else 1.0
  return {label:max(float(observed.get(label,neutral)),1e-12) for label in labels}
 def _update_persistent_debt(self,count):
  for label,delta in self._risk_target_deltas(self._memory_y,self._persistent_scores,count).items(): self._persistent_debt[label]=max(-float(count),min(float(count),self._persistent_debt.get(label,0.0)+delta))
 def _select_persistent_debt_pair(self,positive,negative):
  return max(positive,key=lambda label:(self._persistent_debt[label],-label)),min(negative,key=lambda label:(self._persistent_debt[label],label))
 def _apply_persistent_debt_swaps(self,indices):
  swaps=0; labels=self._memory_y; self._persistent_last_swap_records=[]
  while swaps<self.max_swaps_per_batch:
   selected=set(indices); positive=[label for label,debt in self._persistent_debt.items() if debt>=.5 and any(index not in selected and labels[index]==label for index in range(len(labels)))]; negative=[label for label,debt in self._persistent_debt.items() if debt<=-.5 and any(labels[index]==label for index in indices)]
   if not positive or not negative: break
   pair=self._select_persistent_debt_pair(positive,negative)
   if pair is None: break
   target_label,source_label=pair; target_candidates=[index for index in range(len(labels)) if index not in selected and labels[index]==target_label]; source_positions=[position for position,index in enumerate(indices) if labels[index]==source_label]
   target_index=self._audit_rng.choice(target_candidates); source_position=self._audit_rng.choice(source_positions); indices[source_position]=target_index; self._persistent_last_swap_records.append({'target_position':source_position,'target_memory_index':target_index,'target_label':target_label,'source_label':source_label,'target_score':float(self._persistent_scores[target_label]),'source_score':float(self._persistent_scores[source_label])}); self._persistent_debt[target_label]-=1.0; self._persistent_debt[source_label]+=1.0; swaps+=1; self._persistent_target_score_sum+=self._persistent_scores[target_label]; self._persistent_source_score_sum+=self._persistent_scores[source_label]
  return swaps
 def _sample_memory(self):
  if not self._memory_x:
   self._srrd_last_keys=[]; return None
  count=min(self.batch_size_mem,len(self._memory_x)); indices=self._rng.sample(range(len(self._memory_x)),count); self._persistent_scores=self._class_persistence_scores(); self._update_persistent_debt(count)
  swaps=self._apply_persistent_debt_swaps(indices); self._persistent_replay_calls+=1; self._persistent_swap_count+=swaps; self._persistent_batches_with_swaps+=int(swaps>0); self._persistent_max_abs_debt=max(self._persistent_max_abs_debt,max((abs(value) for value in self._persistent_debt.values()),default=0.0)); self._last_replay_memory_indices=list(indices); self._record_memory_trace_replay(indices); self._replay_examples+=count
  if len(self._srrd_memory_keys)!=len(self._memory_x): self._refresh_srrd_memory_keys()
  self._srrd_last_keys=[self._srrd_memory_keys[index] for index in indices]; x=torch.stack([self._memory_x[index] for index in indices]).to(self.device); y=torch.tensor([self._memory_y[index] for index in indices],dtype=torch.long,device=self.device); tids=torch.tensor([self._memory_tid[index] for index in indices],dtype=torch.long,device=self.device); return x,y,tids
 def rbcl_summary(self):
  r=super().rbcl_summary(); calls=max(1,self._persistent_replay_calls); swaps=max(1,self._persistent_swap_count); audit=r['persistent_self_referenced_replay_deterioration_audit']; enabled=not self.force_neutral_signal and self.max_swaps_per_batch>0; audit['audit_only']=not enabled; audit['sampling_modified_by_audit']=enabled; audit['primary_signal_controls_sampling']=enabled; r['persistent_srrd_replay']={'enabled':enabled,'force_neutral_signal':self.force_neutral_signal,'parent_uniform_draw_is_exact':True,'main_sampler_rng_receives_no_extra_draws':True,'independent_within_class_rng':True,'signal_is_read_before_current_batch_update':True,'uses_current_or_future_loss_for_selection':False,'uses_task_id_gradient_validation_pss_or_future_data':False,'debt_activation_abs':.5,'max_swaps_per_batch':self.max_swaps_per_batch,'replay_calls':self._persistent_replay_calls,'swap_count':self._persistent_swap_count,'mean_swaps_per_batch':self._persistent_swap_count/calls,'batches_with_swaps':self._persistent_batches_with_swaps,'max_abs_debt':self._persistent_max_abs_debt,'mean_target_persistence':self._persistent_target_score_sum/swaps,'mean_source_persistence':self._persistent_source_score_sum/swaps,'neutral_signal_is_exact_noop':True}; return r


class SelectivePersistentSRRDSwapERACE(PersistentSRRDDebtSwapERACE):
 """Use one confidence-gated swap from the current lagged SRRD residual."""

 def __init__(self,*args,confidence_z:float=1.96,force_neutral_signal:bool=False,**kwargs):
  if confidence_z<=0: raise ValueError('confidence_z must be positive')
  super().__init__(*args,max_swaps_per_batch=1,force_neutral_signal=force_neutral_signal,**kwargs)
  self.confidence_z=float(confidence_z); self._selective_confidence_rejections=0
 @staticmethod
 def _wilson_interval(positive_count,total_count,z=1.96):
  if total_count<=0: return 0.0,1.0
  p=float(positive_count)/float(total_count); z2=float(z)*float(z); denominator=1.0+z2/total_count; center=(p+z2/(2.0*total_count))/denominator; radius=(float(z)/denominator)*math.sqrt(p*(1.0-p)/total_count+z2/(4.0*total_count*total_count)); return center-radius,center+radius
 def _update_persistent_debt(self,count):
  self._persistent_debt={label:max(-float(count),min(float(count),delta)) for label,delta in self._risk_target_deltas(self._memory_y,self._persistent_scores,count).items()}
 def _select_persistent_debt_pair(self,positive,negative):
  candidates=[]
  for target_label in positive:
   target=self._srrd_classes.get(target_label)
   if target is None or target['repeat_event_count']<=0: continue
   target_low,_=self._wilson_interval(target['positive_ce_event_count'],target['repeat_event_count'],self.confidence_z)
   for source_label in negative:
    source=self._srrd_classes.get(source_label)
    if source is None or source['repeat_event_count']<=0: continue
    _,source_high=self._wilson_interval(source['positive_ce_event_count'],source['repeat_event_count'],self.confidence_z)
    gap=target_low-source_high
    if gap>0: candidates.append((gap,self._persistent_debt[target_label],-self._persistent_debt[source_label],-target_label,-source_label,target_label,source_label))
  if not candidates:
   self._selective_confidence_rejections+=1; return None
  best=max(candidates); return best[-2],best[-1]
 def rbcl_summary(self):
  r=super().rbcl_summary(); replay=r['persistent_srrd_replay']; replay.update({'controller_mode':'confidence_gated_instantaneous_residual','debt_carries_across_batches':False,'confidence_interval':'wilson_95_non_overlap','confidence_z':self.confidence_z,'minimum_intervention_units_per_batch':0,'maximum_intervention_units_per_batch':1,'supports_abstention':True,'batches_without_swaps':self._persistent_replay_calls-self._persistent_batches_with_swaps,'confidence_rejected_batches':self._selective_confidence_rejections}); return r


class ReplayFeatureDualHeadCalibrationERACE(SelectivePersistentSRRDSwapERACE):
 """Decouple representation learning from a memory-only deployment head.

 The parent model and its classifier remain the exact Layer-2 training path.
 A separate linear head is updated only from replay features that Layer 2 has
 already computed.  Evaluation uses that head, while a zero learning-rate
 scale is an exact Layer-2 no-op.  This follows the classifier-decoupling
 principle of Online Bias Correction without a second replay draw or a second
 backbone forward.
 """

 def __init__(
  self,
  *args,
  calibration_lr_scale:float=1.0,
  calibration_label_smoothing:float=0.1,
  calibration_replay_detailed_audit:bool=True,
  **kwargs,
 ):
  if calibration_lr_scale<0.0:
   raise ValueError('calibration_lr_scale must be nonnegative')
  if not 0.0<=calibration_label_smoothing<1.0:
   raise ValueError('calibration_label_smoothing must be in [0,1)')
  super().__init__(*args,**kwargs)
  self.calibration_lr_scale=float(calibration_lr_scale)
  self.calibration_label_smoothing=float(calibration_label_smoothing)
  self.calibration_replay_detailed_audit=bool(
   calibration_replay_detailed_audit
  )
  _,training_head=get_last_fc_layer(self.model)
  self._calibration_head=copy.deepcopy(training_head).to(self.device)
  self._calibration_head.eval()
  self._calibration_capture_enabled=False
  self._calibration_captured_features=[]
  self._calibration_pending_batch=None
  self._calibration_replay_calls=0
  self._calibration_updates=0
  self._calibration_samples=0
  self._calibration_class_observations=0
  self._calibration_nonfinite_skips=0
  self._calibration_loss_sum=0.0
  self._calibration_min_loss=None
  self._calibration_max_loss=0.0
  self._calibration_head_update_host_seconds=0.0
  self._calibration_replay_head_audit=empty_dual_head_audit()
  self._calibration_train_phase_index=-1
  self._calibration_class_phase={}
  self._calibration_eval_current=None
  self._calibration_eval_history=[]
  self._calibration_eval_forward_calls=0

  def capture_head_input(_module,inputs):
   if self._calibration_capture_enabled:
    self._calibration_captured_features.append(inputs[0].detach())

  self._calibration_feature_hook=(
   training_head.register_forward_pre_hook(capture_head_input)
   if self.calibration_lr_scale>0.0 else None
  )

 @staticmethod
 def _class_balanced_smoothed_loss(logits,labels,label_smoothing):
  losses=F.cross_entropy(
   logits,
   labels,
   reduction='none',
   label_smoothing=float(label_smoothing),
  )
  classes=torch.unique(labels,sorted=True)
  if int(classes.numel())==0:
   raise ValueError('calibration labels must be nonempty')
  return torch.stack([losses[labels==label].mean() for label in classes]).mean()

 @staticmethod
 def _head_learning_rate(optimizer,parameters):
  parameter_ids={id(parameter) for parameter in parameters}; learning_rates=[]
  for group in optimizer.param_groups:
   if any(id(parameter) in parameter_ids for parameter in group['params']):
    learning_rates.append(float(group['lr']))
  if not learning_rates:
   raise RuntimeError('classifier head parameters are absent from optimizer')
  return min(learning_rates)

 @staticmethod
 def _module_state_hash(module):
  digest=hashlib.sha256()
  for name,value in sorted(module.state_dict().items()):
   digest.update(name.encode('utf-8'))
   digest.update(value.detach().cpu().contiguous().numpy().tobytes())
  return digest.hexdigest()

 def _before_training_exp(self,**kwargs):
  if self.calibration_lr_scale>0.0:
   self._calibration_train_phase_index+=1
  super()._before_training_exp(**kwargs)

 def _before_forward(self,**kwargs):
  if self.calibration_lr_scale<=0.0:
   return super()._before_forward(**kwargs)
  if self.is_training:
   for label in self.mb_y.detach().unique(sorted=True).cpu().tolist():
    self._calibration_class_phase.setdefault(
     int(label),self._calibration_train_phase_index
    )
  self._calibration_captured_features=[]
  self._calibration_capture_enabled=(
   self.is_training and self.calibration_lr_scale>0.0
  )
  super()._before_forward(**kwargs)

 def _before_eval(self,**kwargs):
  self._calibration_eval_current=(
   empty_dual_head_audit()
   if self.calibration_lr_scale>0.0 and self._calibration_updates>0
   else None
  )
  super()._before_eval(**kwargs)

 def _deployment_audit_state(self):
  return {
   'calibration_updates':self._calibration_updates,
   'calibration_samples':self._calibration_samples,
  }

 def _after_eval(self,**kwargs):
  if self._calibration_eval_current is not None:
   observed=sorted(int(label) for label in self._observed_classes)
   phases={label:self._calibration_class_phase.get(label,-1) for label in observed}
   newest=max(phases.values(),default=-1)
   self._calibration_eval_history.append({
    'evaluation_index':len(self._calibration_eval_history),
    'training_phase_index':self._calibration_train_phase_index,
    'observed_classes':observed,
    'old_classes':[label for label in observed if phases[label]<newest],
    'new_classes':[label for label in observed if phases[label]==newest],
    'deployment_state':self._deployment_audit_state(),
    'dual_head_metrics':summarize_dual_head_audit(
     self._calibration_eval_current
    ),
   })
  self._calibration_eval_current=None
  super()._after_eval(**kwargs)

 def _record_calibration_evaluation(self,training_logits,deployment_logits):
  if self._calibration_eval_current is None:
   return
  labels=self.mb_y.detach()
  observed=sorted(int(label) for label in self._observed_classes)
  if not observed:
   return
  observed_tensor=torch.tensor(observed,dtype=torch.long,device=labels.device)
  mask=labels[:,None].eq(observed_tensor[None,:]).any(dim=1)
  if not bool(mask.any()):
   return
  phases={label:self._calibration_class_phase.get(label,-1) for label in observed}
  newest=max(phases.values(),default=-1)
  old=[label for label in observed if phases[label]<newest]
  new=[label for label in observed if phases[label]==newest]
  update_dual_head_audit(
   self._calibration_eval_current,
   training_logits[mask].detach(),
   deployment_logits[mask].detach(),
   labels[mask],
   old_classes=old,
   new_classes=new,
  )
  self._calibration_eval_forward_calls+=1

 def forward(self):
  if (
   self.is_training
   or self.calibration_lr_scale<=0.0
   or self._calibration_updates<=0
  ):
   return super().forward()
  _,training_head=get_last_fc_layer(self.model); captured={}

  def capture(_module,inputs):
   captured['features']=inputs[0].detach()

  handle=training_head.register_forward_pre_hook(capture)
  try:
   training_logits=super().forward()
  finally:
   handle.remove()
  if 'features' not in captured:
   raise RuntimeError('classifier head input was not observed during evaluation')
  self._calibration_head.eval()
  calibration_logits=self._calibration_head(captured['features'])
  deployment_logits=self._deployment_logits(
   training_logits,calibration_logits
  )
  self._record_calibration_evaluation(training_logits,deployment_logits)
  return deployment_logits

 def _deployment_logits(self,training_logits,calibration_logits):
  return calibration_logits

 def _add_auxiliary_loss(
  self,current_x,current_y,current_tid,replay,replay_output=None
 ):
  super()._add_auxiliary_loss(
   current_x,current_y,current_tid,replay,replay_output
  )
  self._calibration_pending_batch=None
  if self.calibration_lr_scale<=0.0 or replay is None:
   self._calibration_capture_enabled=False
   self._calibration_captured_features=[]
   return
  replay_count=int(replay[1].numel())
  captured=list(self._calibration_captured_features)
  candidates=[features for features in captured if int(features.shape[0])==replay_count]
  self._calibration_capture_enabled=False
  self._calibration_captured_features=[]
  if not candidates:
   raise RuntimeError('replay classifier features were not captured')
  if not captured:
   raise RuntimeError('current classifier features were not captured')
  self._calibration_pending_batch=(
   captured[0].detach(),
   current_y.detach(),
   self.mb_output.detach(),
   candidates[-1].detach(),
   replay[1].detach(),
   replay_output.detach(),
  )

 def _after_calibration_head_update(
  self,current_features,current_labels,current_training_logits,
  replay_features,replay_labels,replay_training_logits,
 ):
  """Extension point for causal deployment-head arbitration."""

 def _after_anchor_update(self,current_x,current_y,current_tid):
  super()._after_anchor_update(current_x,current_y,current_tid)
  pending=self._calibration_pending_batch
  self._calibration_pending_batch=None
  if self.calibration_lr_scale<=0.0 or pending is None:
   return
  (
   current_features,current_labels,current_training_logits,
   features,labels,training_logits,
  )=pending
  self._calibration_replay_calls+=1
  self._calibration_head.train()
  parameters=tuple(
   parameter for parameter in self._calibration_head.parameters()
   if parameter.requires_grad
  )
  host_started=time.perf_counter()
  logits=self._calibration_head(features)
  loss=self._class_balanced_smoothed_loss(
   logits,labels,self.calibration_label_smoothing
  )
  if not bool(torch.isfinite(loss)):
   self._calibration_head_update_host_seconds+=(
    time.perf_counter()-host_started
   )
   self._calibration_nonfinite_skips+=1
   self._calibration_head.eval()
   return
  gradients=torch.autograd.grad(loss,parameters)
  _,training_head=get_last_fc_layer(self.model)
  training_parameters=tuple(
   parameter for parameter in training_head.parameters()
   if parameter.requires_grad
  )
  step=self.calibration_lr_scale*self._head_learning_rate(
   self.optimizer,training_parameters
  )
  with torch.no_grad():
   for parameter,gradient in zip(parameters,gradients):
    parameter.add_(gradient,alpha=-step)
  self._calibration_head_update_host_seconds+=(
   time.perf_counter()-host_started
  )
  if self.calibration_replay_detailed_audit:
   update_dual_head_audit(
    self._calibration_replay_head_audit,
    training_logits,
    logits.detach(),
    labels,
    old_classes=torch.unique(labels,sorted=True).cpu().tolist(),
   )
  value=float(loss.detach().cpu())
  self._calibration_updates+=1
  self._calibration_samples+=int(labels.numel())
  self._calibration_class_observations+=int(torch.unique(labels).numel())
  self._calibration_loss_sum+=value
  self._calibration_min_loss=(
   value if self._calibration_min_loss is None
   else min(self._calibration_min_loss,value)
  )
  self._calibration_max_loss=max(self._calibration_max_loss,value)
  self._calibration_head.eval()
  self._after_calibration_head_update(
   current_features,current_labels,current_training_logits,
   features,labels,training_logits,
  )

 def rbcl_summary(self):
  r=super().rbcl_summary(); updates=max(1,self._calibration_updates)
  training_head=get_last_fc_layer(self.model)[1]
  r['replay_feature_dual_head_calibration']={
   'enabled':self.calibration_lr_scale>0.0,
   'parent_layer2':'persistent_srrd_selective_swap_1',
   'function':'memory-only deployment-head calibration with replay-feature reuse',
   'source_principles':[
    'online_bias_correction',
    'decoupled_representation_and_classifier_learning',
   ],
   'calibration_lr_scale':self.calibration_lr_scale,
   'label_smoothing':self.calibration_label_smoothing,
   'replay_calls':self._calibration_replay_calls,
   'calibration_updates':self._calibration_updates,
   'calibration_samples':self._calibration_samples,
   'calibration_class_observations':self._calibration_class_observations,
    'mean_classes_per_update':self._calibration_class_observations/updates,
    'mean_calibration_loss':self._calibration_loss_sum/updates,
    'minimum_calibration_loss':self._calibration_min_loss,
    'maximum_calibration_loss':self._calibration_max_loss,
    'head_update_host_dispatch_seconds':self._calibration_head_update_host_seconds,
    'mean_head_update_host_dispatch_seconds':(
      self._calibration_head_update_host_seconds/updates
    ),
    'replay_head_comparison':summarize_dual_head_audit(
      self._calibration_replay_head_audit
    ),
    'replay_head_comparison_detailed':self.calibration_replay_detailed_audit,
    'evaluation_bias_audit':{
     'enabled':self.calibration_lr_scale>0.0,
     'evaluation_forward_calls':self._calibration_eval_forward_calls,
     'history':self._calibration_eval_history,
     'metrics':[
      'accuracy','nll','brier_score','ece_15bin',
      'prediction_disagreement_rate','mean_absolute_logit_delta',
      'new_to_old_prediction_ratio','mean_new_minus_old_max_logit',
      'per_class',
     ],
     'uses_eval_labels_for_audit_only':True,
     'uses_training_phase_boundary_for_old_new_audit_only':True,
     'audit_does_not_control_training_sampling_or_prediction':True,
     'future_unobserved_target_classes_are_excluded':True,
    },
   'nonfinite_skips':self._calibration_nonfinite_skips,
   'extra_head_parameters':sum(
    int(parameter.numel()) for parameter in self._calibration_head.parameters()
   ),
   'training_head_hash':self._module_state_hash(training_head),
   'deployment_head_hash':self._module_state_hash(self._calibration_head),
   'main_training_head_is_surrogate':True,
   'deployment_uses_calibration_head':(
    self.calibration_lr_scale>0.0 and self._calibration_updates>0
   ),
   'calibration_uses_replay_only':True,
   'calibration_loss_is_class_balanced_over_present_classes':True,
   'reuses_layer2_replay_features':True,
   'additional_replay_draws':0,
    'additional_backbone_forwards':0,
    'additional_training_calibration_head_forwards':self._calibration_replay_calls,
    'additional_evaluation_calibration_head_forwards':self._calibration_eval_forward_calls,
    'training_feature_hook_registered':self._calibration_feature_hook is not None,
    'disabled_path_registers_no_training_feature_hook':(
     self.calibration_lr_scale>0.0 or self._calibration_feature_hook is None
    ),
    'run_level_peak_cuda_memory_is_recorded_in_run_metadata':True,
    'evaluation_audit_adds_metric_only_tensor_operations':True,
   'main_model_parameters_are_not_modified_by_calibration':True,
   'main_optimizer_state_is_not_modified_by_calibration':True,
   'memory_and_replay_identities_are_unchanged':True,
   'uses_task_id_boundary_validation_pss_or_future_data':False,
   'zero_lr_scale_is_exact_layer2_noop':True,
  }
  return r


class FixedAlphaDualHeadCalibrationERACE(
 ReplayFeatureDualHeadCalibrationERACE
):
 """Use the replay-trained calibration head with a preregistered fixed blend."""

 def __init__(self,*args,deployment_alpha:float=0.5,**kwargs):
  if not 0.0<=deployment_alpha<=1.0:
   raise ValueError('deployment_alpha must be in [0,1]')
  self.deployment_alpha=float(deployment_alpha)
  super().__init__(*args,**kwargs)

 def _deployment_logits(self,training_logits,calibration_logits):
  return torch.lerp(
   training_logits,calibration_logits,self.deployment_alpha
  )

 def rbcl_summary(self):
  r=super().rbcl_summary()
  calibration=r['replay_feature_dual_head_calibration']
  calibration['deployment_uses_calibration_head']=(
   calibration['enabled'] and self._calibration_updates>0
   and self.deployment_alpha>0.0
  )
  calibration['deployment_uses_fixed_alpha_blend']=(
   calibration['enabled'] and self._calibration_updates>0
  )
  r['fixed_alpha_head_arbitration']={
   'enabled':calibration['enabled'],
   'deployment_rule':'z_train + alpha * (z_calibration - z_train)',
   'deployment_alpha':self.deployment_alpha,
   'alpha_source':'preregistered constant, not selected from results',
   'fixed_or_tuned_alpha':True,
   'uses_validation_pss_future_data_or_task_boundary':False,
   'additional_replay_draws':0,
   'additional_backbone_forwards':0,
  }
  return r


class CanonicalOBCDualHeadERACE(ReplayFeatureDualHeadCalibrationERACE):
 """Faithful Algorithm-1 OBC wrapper around the frozen Layer-2 parent.

 Unlike the feature-reuse control, this class performs a second independent
 uniform memory draw after the base train-and-populate stages, forwards that
 batch through the frozen/eval-mode feature extractor, and updates only the
 deployment classifier with label-smoothed cross entropy.
 """

 def __init__(self,*args,obc_seed:int=190100,**kwargs):
  super().__init__(*args,**kwargs)
  self._obc_rng=random.Random(int(obc_seed))
  self._obc_replay_draws=0
  self._obc_backbone_forwards=0
  self._obc_replay_samples=0

 def _before_forward(self,**kwargs):
  self._calibration_capture_enabled=False
  self._calibration_captured_features=[]
  return super(ReplayFeatureDualHeadCalibrationERACE,self)._before_forward(
   **kwargs
  )

 def _add_auxiliary_loss(
  self,current_x,current_y,current_tid,replay,replay_output=None
 ):
  self._calibration_pending_batch=None
  return super(ReplayFeatureDualHeadCalibrationERACE,self)._add_auxiliary_loss(
   current_x,current_y,current_tid,replay,replay_output
  )

 def _after_anchor_update(self,current_x,current_y,current_tid):
  self._calibration_pending_batch=None
  return super(ReplayFeatureDualHeadCalibrationERACE,self)._after_anchor_update(
   current_x,current_y,current_tid
  )

 def _after_memory_population_stage(self,current_x,current_y,current_tid):
  super()._after_memory_population_stage(current_x,current_y,current_tid)
  if self.calibration_lr_scale<=0.0 or not self._memory_x:
   return
  count=min(self.batch_size_mem,len(self._memory_x))
  indices=self._obc_rng.sample(range(len(self._memory_x)),count)
  replay_x=torch.stack([self._memory_x[index] for index in indices]).to(
   self.device
  )
  replay_y=torch.tensor(
   [self._memory_y[index] for index in indices],
   dtype=torch.long,device=self.device,
  )
  replay_tid=torch.tensor(
   [self._memory_tid[index] for index in indices],
   dtype=torch.long,device=self.device,
  )
  _,training_head=get_last_fc_layer(self.model)
  captured={}

  def capture(_module,inputs):
   captured['features']=inputs[0].detach()

  handle=training_head.register_forward_pre_hook(capture)
  was_training=self.model.training
  self.model.eval()
  try:
   with torch.no_grad():
    avalanche_forward(self.model,replay_x,replay_tid)
  finally:
   handle.remove()
   self.model.train(was_training)
  if 'features' not in captured:
   raise RuntimeError('OBC classifier input was not observed')

  self._obc_replay_draws+=1
  self._obc_backbone_forwards+=1
  self._obc_replay_samples+=count
  self._calibration_replay_calls+=1
  self._calibration_head.train()
  parameters=tuple(
   parameter for parameter in self._calibration_head.parameters()
   if parameter.requires_grad
  )
  host_started=time.perf_counter()
  logits=self._calibration_head(captured['features'])
  loss=F.cross_entropy(
   logits,replay_y,label_smoothing=self.calibration_label_smoothing
  )
  if not bool(torch.isfinite(loss)):
   self._calibration_head_update_host_seconds+=(
    time.perf_counter()-host_started
   )
   self._calibration_nonfinite_skips+=1
   self._calibration_head.eval()
   return
  gradients=torch.autograd.grad(loss,parameters)
  training_parameters=tuple(
   parameter for parameter in training_head.parameters()
   if parameter.requires_grad
  )
  step=self.calibration_lr_scale*self._head_learning_rate(
   self.optimizer,training_parameters
  )
  with torch.no_grad():
   for parameter,gradient in zip(parameters,gradients):
    parameter.add_(gradient,alpha=-step)
  self._calibration_head_update_host_seconds+=(
   time.perf_counter()-host_started
  )
  value=float(loss.detach().cpu())
  self._calibration_updates+=1
  self._calibration_samples+=count
  self._calibration_class_observations+=int(torch.unique(replay_y).numel())
  self._calibration_loss_sum+=value
  self._calibration_min_loss=(
   value if self._calibration_min_loss is None
   else min(self._calibration_min_loss,value)
  )
  self._calibration_max_loss=max(self._calibration_max_loss,value)
  self._calibration_head.eval()

 def rbcl_summary(self):
  r=super().rbcl_summary()
  calibration=r['replay_feature_dual_head_calibration']
  calibration.update({
   'function':'canonical Online Bias Correction classifier update',
   'source_principles':['online_bias_correction_algorithm_1'],
   'calibration_uses_parent_replay_features':False,
   'calibration_loss_is_class_balanced_over_present_classes':False,
   'calibration_loss_is_label_smoothed_cross_entropy':True,
   'reuses_layer2_replay_features':False,
   'additional_replay_draws':self._obc_replay_draws,
   'additional_backbone_forwards':self._obc_backbone_forwards,
   'additional_training_calibration_head_forwards':self._obc_replay_draws,
   'obc_replay_samples':self._obc_replay_samples,
   'obc_second_draw_is_uniform_and_independent':True,
   'obc_update_occurs_after_base_train_and_memory_population':True,
   'feature_extractor_is_frozen_and_eval_mode_for_obc_forward':True,
  })
  r['online_bias_correction']={
   'enabled':self.calibration_lr_scale>0.0,
   'algorithm':'OBC Algorithm 1',
   'wrapped_parent':'persistent_srrd_selective_swap_1',
   'second_memory_draws':self._obc_replay_draws,
   'additional_backbone_forwards':self._obc_backbone_forwards,
   'replay_samples':self._obc_replay_samples,
   'uses_label_smoothing':self.calibration_label_smoothing>0.0,
   'uses_task_id_validation_pss_or_future_data':False,
  }
  return r


class RiskBudgetedDualHeadArbitrationERACE(
 ReplayFeatureDualHeadCalibrationERACE
):
 """Spend replay calibration gain only when arrived current evidence permits it.

 For every replay update, the strategy finds the convex optimum between the
 training and calibration heads on the already-arrived current/replay pair.
 The deployment coefficient is the cumulative mean of those causal optima, so
 no fixed alpha, validation stream, task boundary, or future sample is used.
 """

 _ARBITRATION_BISECTION_STEPS=24
 _ARBITRATION_CURRENT_WEIGHT=0.5

 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs)
  self._arbitration_batches=0
  self._arbitration_alpha_sum=0.0
  self._arbitration_alpha=0.0
  self._arbitration_alpha_min=None
  self._arbitration_alpha_max=None
  self._arbitration_zero_count=0
  self._arbitration_interior_count=0
  self._arbitration_one_count=0
  self._arbitration_current_ce_delta_sum=0.0
  self._arbitration_current_ce_delta_max=0.0
  self._arbitration_replay_ce_improvement_sum=0.0
  self._arbitration_joint_ce_improvement_sum=0.0
  self._arbitration_host_seconds=0.0
  self._arbitration_nonfinite_skips=0

 @staticmethod
 def _class_balanced_weights(labels,dtype):
  classes,inverse,counts=torch.unique(
   labels,sorted=True,return_inverse=True,return_counts=True
  )
  if int(classes.numel())==0:
   raise ValueError('arbitration labels must be nonempty')
  counts=counts.to(device=labels.device,dtype=dtype)
  return counts[inverse].reciprocal()/float(classes.numel())

 @classmethod
 def _balanced_smoothed_ce_derivative(
  cls,base_logits,logit_delta,labels,label_smoothing,alpha
 ):
  mixed=base_logits+alpha*logit_delta
  probabilities=F.softmax(mixed,dim=1)
  rows=torch.arange(labels.numel(),device=labels.device)
  target_delta=(1.0-float(label_smoothing))*logit_delta[rows,labels]
  target_delta+=float(label_smoothing)*logit_delta.mean(dim=1)
  per_item=(probabilities*logit_delta).sum(dim=1)-target_delta
  weights=cls._class_balanced_weights(labels,per_item.dtype)
  return (weights*per_item).sum()

 @classmethod
 def _joint_balanced_smoothed_loss(
  cls,current_logits,current_labels,replay_logits,replay_labels,label_smoothing
 ):
  return 0.5*(
   cls._class_balanced_smoothed_loss(
    current_logits,current_labels,label_smoothing
   )+
   cls._class_balanced_smoothed_loss(
    replay_logits,replay_labels,label_smoothing
   )
  )

 @classmethod
 def _optimal_arbitration_alpha(
  cls,
  current_training_logits,current_calibration_logits,current_labels,
  replay_training_logits,replay_calibration_logits,replay_labels,
  label_smoothing,
 ):
  current_delta=current_calibration_logits-current_training_logits
  replay_delta=replay_calibration_logits-replay_training_logits

  def derivative(alpha):
   current_weight=float(cls._ARBITRATION_CURRENT_WEIGHT)
   return (
    current_weight*
    cls._balanced_smoothed_ce_derivative(
     current_training_logits,current_delta,current_labels,
     label_smoothing,alpha
    )+(1.0-current_weight)*
    cls._balanced_smoothed_ce_derivative(
     replay_training_logits,replay_delta,replay_labels,
     label_smoothing,alpha
    )
   )

  at_zero=derivative(0.0)
  at_one=derivative(1.0)
  if not bool(torch.isfinite(torch.stack([at_zero,at_one])).all()):
   raise FloatingPointError('nonfinite arbitration derivative')
  low=torch.zeros_like(at_zero); high=torch.ones_like(at_zero)
  for _ in range(cls._ARBITRATION_BISECTION_STEPS):
   middle=(low+high)/2.0
   value=derivative(middle)
   low=torch.where(value<=0.0,middle,low)
   high=torch.where(value<=0.0,high,middle)
  interior=(low+high)/2.0
  alpha=torch.where(
   at_zero>=0.0,torch.zeros_like(interior),
   torch.where(at_one<=0.0,torch.ones_like(interior),interior),
  )
  if not bool(torch.isfinite(alpha)):
   raise FloatingPointError('nonfinite arbitration derivative')
  return float(alpha.detach().cpu())

 def _record_arbitration_logits(
  self,current_training,current_calibration,current_labels,
  replay_training,replay_calibration,replay_labels,started,
 ):
  try:
   with torch.no_grad():
    alpha=self._optimal_arbitration_alpha(
     current_training,current_calibration,current_labels,
     replay_training,replay_calibration,replay_labels,
     self.calibration_label_smoothing,
    )
    mixed_current=torch.lerp(current_training,current_calibration,alpha)
    mixed_replay=torch.lerp(replay_training,replay_calibration,alpha)
    current_base=self._class_balanced_smoothed_loss(
     current_training,current_labels,self.calibration_label_smoothing
    )
    current_mixed=self._class_balanced_smoothed_loss(
     mixed_current,current_labels,self.calibration_label_smoothing
    )
    replay_base=self._class_balanced_smoothed_loss(
     replay_training,replay_labels,self.calibration_label_smoothing
    )
    replay_mixed=self._class_balanced_smoothed_loss(
     mixed_replay,replay_labels,self.calibration_label_smoothing
    )
    joint_base=0.5*(current_base+replay_base)
    joint_mixed=0.5*(current_mixed+replay_mixed)
    values=torch.stack([
     current_base,current_mixed,replay_base,replay_mixed
    ])
    if not bool(torch.isfinite(values).all()):
     raise FloatingPointError('nonfinite arbitration loss')
    current_delta,replay_improvement,joint_improvement=torch.stack([
     current_mixed-current_base,
     replay_base-replay_mixed,
     joint_base-joint_mixed,
    ]).detach().cpu().tolist()
  except FloatingPointError:
   self._arbitration_nonfinite_skips+=1
   self._arbitration_host_seconds+=time.perf_counter()-started
   return

  self._arbitration_batches+=1
  self._arbitration_alpha_sum+=alpha
  self._arbitration_alpha=(
   self._arbitration_alpha_sum/self._arbitration_batches
  )
  self._arbitration_alpha_min=(
   alpha if self._arbitration_alpha_min is None
   else min(self._arbitration_alpha_min,alpha)
  )
  self._arbitration_alpha_max=(
   alpha if self._arbitration_alpha_max is None
   else max(self._arbitration_alpha_max,alpha)
  )
  tolerance=2.0**(-self._ARBITRATION_BISECTION_STEPS)
  if alpha<=tolerance:
   self._arbitration_zero_count+=1
  elif alpha>=1.0-tolerance:
   self._arbitration_one_count+=1
  else:
   self._arbitration_interior_count+=1
  self._arbitration_current_ce_delta_sum+=current_delta
  self._arbitration_current_ce_delta_max=max(
   self._arbitration_current_ce_delta_max,current_delta
  )
  self._arbitration_replay_ce_improvement_sum+=replay_improvement
  self._arbitration_joint_ce_improvement_sum+=joint_improvement
  self._arbitration_host_seconds+=time.perf_counter()-started

 def _after_calibration_head_update(
  self,current_features,current_labels,current_training_logits,
  replay_features,replay_labels,replay_training_logits,
 ):
  started=time.perf_counter()
  _,training_head=get_last_fc_layer(self.model)
  with torch.no_grad():
   current_training=training_head(current_features)
   replay_training=training_head(replay_features)
   current_calibration=self._calibration_head(current_features)
   replay_calibration=self._calibration_head(replay_features)
  self._record_arbitration_logits(
   current_training,current_calibration,current_labels,
   replay_training,replay_calibration,replay_labels,started,
  )

 def _deployment_logits(self,training_logits,calibration_logits):
  return torch.lerp(
   training_logits,calibration_logits,float(self._arbitration_alpha)
  )

 def _deployment_audit_state(self):
  batches=max(1,self._arbitration_batches)
  state=super()._deployment_audit_state()
  state.update({
   'arbitration_batches':self._arbitration_batches,
   'deployment_alpha':self._arbitration_alpha,
   'minimum_batch_alpha':self._arbitration_alpha_min,
   'maximum_batch_alpha':self._arbitration_alpha_max,
   'mean_current_ce_delta':self._arbitration_current_ce_delta_sum/batches,
   'mean_replay_ce_improvement':(
    self._arbitration_replay_ce_improvement_sum/batches
   ),
   'mean_joint_ce_improvement':(
    self._arbitration_joint_ce_improvement_sum/batches
   ),
  })
  return state

 def rbcl_summary(self):
  r=super().rbcl_summary(); batches=max(1,self._arbitration_batches)
  enabled=self.calibration_lr_scale>0.0
  calibration=r['replay_feature_dual_head_calibration']
  calibration['deployment_uses_calibration_head']=(
   enabled and self._calibration_updates>0 and self._arbitration_alpha>0.0
  )
  calibration['deployment_uses_risk_budgeted_blend']=(
   enabled and self._arbitration_batches>0
  )
  calibration['additional_training_arbitration_head_forwards']=(
   4*self._arbitration_batches
  )
  r['risk_budgeted_head_arbitration']={
   'enabled':enabled,
   'parent_layer3_candidate':'persistent_srrd_dual_head_calibration_1',
   'function':'causal risk-budgeted arbitration between training and calibration heads',
   'deployment_rule':'z_train + alpha * (z_calibration - z_train)',
    'batch_objective':(
     'current-only class-balanced smoothed cross entropy'
     if self._ARBITRATION_CURRENT_WEIGHT==1.0 else
     'replay-only class-balanced smoothed cross entropy'
     if self._ARBITRATION_CURRENT_WEIGHT==0.0 else
     'equal-weight current/replay class-balanced smoothed cross entropy'
    ),
    'arbitration_current_weight':self._ARBITRATION_CURRENT_WEIGHT,
   'alpha_source':'online cumulative mean of batchwise convex optima',
   'fixed_or_tuned_alpha':False,
   'bisection_steps_for_numerical_solution':self._ARBITRATION_BISECTION_STEPS,
   'arbitration_batches':self._arbitration_batches,
   'deployment_alpha':self._arbitration_alpha,
   'minimum_batch_alpha':self._arbitration_alpha_min,
   'maximum_batch_alpha':self._arbitration_alpha_max,
   'zero_alpha_batches':self._arbitration_zero_count,
   'interior_alpha_batches':self._arbitration_interior_count,
   'one_alpha_batches':self._arbitration_one_count,
   'mean_current_ce_delta':self._arbitration_current_ce_delta_sum/batches,
   'maximum_current_ce_delta':self._arbitration_current_ce_delta_max,
   'mean_replay_ce_improvement':(
    self._arbitration_replay_ce_improvement_sum/batches
   ),
   'mean_joint_ce_improvement':(
    self._arbitration_joint_ce_improvement_sum/batches
   ),
   'host_dispatch_seconds':self._arbitration_host_seconds,
   'mean_host_dispatch_seconds':self._arbitration_host_seconds/batches,
   'nonfinite_skips':self._arbitration_nonfinite_skips,
   'uses_arrived_current_and_replay_batches_only':True,
   'uses_validation_pss_future_data_or_task_boundary':False,
   'additional_replay_draws':0,
   'additional_backbone_forwards':0,
   'additional_head_only_forwards':4*self._arbitration_batches,
   'disabled_path_is_parent_layer2_noop':True,
  }
  return r


class PrequentialRiskBudgetedDualHeadArbitrationERACE(
 RiskBudgetedDualHeadArbitrationERACE
):
 """Choose the risk budget before the calibration head sees this update.

 The training and calibration heads are compared on the arrived current/replay
 pair before either the replay-only calibration update or the main optimizer
 step is applied.  The resulting coefficient therefore measures pre-update
 predictive evidence rather than reusing the replay batch that just trained
 the calibration head.
 """

 _ARBITRATION_BISECTION_STEPS=8

 def _add_auxiliary_loss(
  self,current_x,current_y,current_tid,replay,replay_output=None
 ):
  super()._add_auxiliary_loss(
   current_x,current_y,current_tid,replay,replay_output
  )
  pending=self._calibration_pending_batch
  if self.calibration_lr_scale<=0.0 or pending is None:
   return
  (
   current_features,current_labels,current_training_logits,
   replay_features,replay_labels,replay_training_logits,
  )=pending
  started=time.perf_counter()
  self._calibration_head.eval()
  with torch.no_grad():
   current_calibration=self._calibration_head(current_features)
   replay_calibration=self._calibration_head(replay_features)
  self._record_arbitration_logits(
   current_training_logits,current_calibration,current_labels,
   replay_training_logits,replay_calibration,replay_labels,started,
  )

 def _after_calibration_head_update(
  self,current_features,current_labels,current_training_logits,
  replay_features,replay_labels,replay_training_logits,
 ):
  # Arbitration was deliberately recorded before this calibration update.
  return

 def rbcl_summary(self):
  r=super().rbcl_summary()
  calibration=r['replay_feature_dual_head_calibration']
  arbitration=r['risk_budgeted_head_arbitration']
  calibration['additional_training_arbitration_head_forwards']=(
   2*self._arbitration_batches
  )
  arbitration.update({
   'function':'prequential causal risk-budgeted arbitration between training and calibration heads',
   'alpha_source':'online cumulative mean of pre-update batchwise convex optima',
   'prequential_test_then_train':True,
   'arbitration_is_recorded_before_main_optimizer_step':True,
   'arbitration_is_recorded_before_current_calibration_update':True,
   'same_update_replay_fit_and_budget_evidence_are_separated':True,
   'post_update_replay_reuse_leakage':False,
   'numerical_alpha_resolution':2.0**(-self._ARBITRATION_BISECTION_STEPS),
   'additional_head_only_forwards':2*self._arbitration_batches,
  })
  return r


class PrequentialCurrentOnlyArbitrationERACE(
 PrequentialRiskBudgetedDualHeadArbitrationERACE
):
 """Ablation whose online alpha is selected from current evidence only."""

 _ARBITRATION_CURRENT_WEIGHT=1.0


class PrequentialReplayOnlyArbitrationERACE(
 PrequentialRiskBudgetedDualHeadArbitrationERACE
):
 """Ablation whose online alpha is selected from replay evidence only."""

 _ARBITRATION_CURRENT_WEIGHT=0.0


class PrequentialLastAlphaArbitrationERACE(
 PrequentialRiskBudgetedDualHeadArbitrationERACE
):
 """Deploy the latest batch alpha instead of its online cumulative mean."""

 def _record_arbitration_logits(
  self,current_training,current_calibration,current_labels,
  replay_training,replay_calibration,replay_labels,started,
 ):
  alpha=self._optimal_arbitration_alpha(
   current_training,current_calibration,current_labels,
   replay_training,replay_calibration,replay_labels,
   self.calibration_label_smoothing,
  )
  prior_batches=self._arbitration_batches
  super()._record_arbitration_logits(
   current_training,current_calibration,current_labels,
   replay_training,replay_calibration,replay_labels,started,
  )
  if self._arbitration_batches>prior_batches:
   self._arbitration_alpha=alpha

 def rbcl_summary(self):
  cumulative_mean=(
   self._arbitration_alpha_sum/max(1,self._arbitration_batches)
  )
  r=super().rbcl_summary()
  arbitration=r['risk_budgeted_head_arbitration']
  arbitration.update({
   'alpha_source':'latest pre-update batchwise convex optimum',
   'aggregation_rule':'last-alpha',
   'cumulative_mean_alpha_for_audit':cumulative_mean,
  })
  return r


class SRRDConsequencePrequentialArbitrationERACE(
 PrequentialRiskBudgetedDualHeadArbitrationERACE
):
 """Couple lagged replay instability to prequential decision consequence.

 The frozen prequential D strategy gives every replay class equal risk mass.
 This exploratory child keeps the current/replay split at exactly 1/2, but
 redistributes only the replay half across classes in proportion to

     lagged Persistent-SRRD instability * pre-update decision consequence.

 Consequence is the absolute class-mean directional derivative from the
 training head toward the calibration head at alpha=0.  The class masses are
 fixed before the convex alpha search, so the one-dimensional objective stays
 convex.  No tunable coefficient, replay draw, or backbone forward is added.
 """

 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs)
  self._coupling_lagged_scores={}
  self._coupling_lagged_repeat_counts={}
  self._coupling_batches=0
  self._coupling_fallback_batches=0
  self._coupling_class_observations=0
  self._coupling_mature_class_observations=0
  self._coupling_instability_sum=0.0
  self._coupling_consequence_sum=0.0
  self._coupling_risk_sum=0.0
  self._coupling_min_class_mass=None
  self._coupling_max_class_mass=0.0
  self._coupling_unweighted_joint_improvement_sum=0.0

 @staticmethod
 def _smoothed_ce_per_item(logits,labels,label_smoothing):
  return F.cross_entropy(
   logits,
   labels,
   reduction='none',
   label_smoothing=float(label_smoothing),
  )

 @classmethod
 def _smoothed_ce_direction_per_item(
  cls,base_logits,logit_delta,labels,label_smoothing,alpha
 ):
  mixed=base_logits+alpha*logit_delta
  probabilities=F.softmax(mixed,dim=1)
  rows=torch.arange(labels.numel(),device=labels.device)
  target_delta=(1.0-float(label_smoothing))*logit_delta[rows,labels]
  target_delta+=float(label_smoothing)*logit_delta.mean(dim=1)
  return (probabilities*logit_delta).sum(dim=1)-target_delta

 @classmethod
 def _replay_consequence_weights(
  cls,training_logits,calibration_logits,labels,label_smoothing,
  lagged_scores,
 ):
  """Return unit-sum item weights and parameter-free class-risk audit."""
  classes,inverse,counts=torch.unique(
   labels,sorted=True,return_inverse=True,return_counts=True
  )
  if int(classes.numel())==0:
   raise ValueError('consequence-aware replay labels must be nonempty')
  delta=calibration_logits-training_logits
  directions=cls._smoothed_ce_direction_per_item(
   training_logits,delta,labels,label_smoothing,0.0
  )
  consequences=torch.stack([
   directions[labels==label].abs().mean() for label in classes
  ])
  score_values=[
   max(0.0,float(lagged_scores.get(int(label),0.0)))
   for label in classes.detach().cpu().tolist()
  ]
  instabilities=torch.tensor(
   score_values,device=labels.device,dtype=consequences.dtype
  )
  risks=instabilities*consequences
  finite=bool(torch.isfinite(risks).all())
  total=risks.sum() if finite else torch.zeros((),device=labels.device)
  fallback=(not finite) or float(total.detach().cpu())<=1e-12
  if fallback:
   class_mass=torch.full_like(risks,1.0/float(classes.numel()))
  else:
   class_mass=risks/total
  counts=counts.to(device=labels.device,dtype=class_mass.dtype)
  item_weights=class_mass[inverse]/counts[inverse]
  audit={
   'classes':[int(label) for label in classes.detach().cpu().tolist()],
   'instability':[float(value) for value in instabilities.detach().cpu().tolist()],
   'consequence':[float(value) for value in consequences.detach().cpu().tolist()],
   'risk':[float(value) for value in risks.detach().cpu().tolist()],
   'class_mass':[float(value) for value in class_mass.detach().cpu().tolist()],
   'fallback_to_uniform':fallback,
  }
  return item_weights,audit

 @staticmethod
 def _weighted_loss(logits,labels,label_smoothing,item_weights):
  losses=SRRDConsequencePrequentialArbitrationERACE._smoothed_ce_per_item(
   logits,labels,label_smoothing
  )
  return (item_weights*losses).sum()

 @classmethod
 def _weighted_derivative(
  cls,base_logits,logit_delta,labels,label_smoothing,alpha,item_weights
 ):
  per_item=cls._smoothed_ce_direction_per_item(
   base_logits,logit_delta,labels,label_smoothing,alpha
  )
  return (item_weights*per_item).sum()

 def _optimal_coupled_alpha(
  self,
  current_training,current_calibration,current_labels,
  replay_training,replay_calibration,replay_labels,replay_weights,
 ):
  current_delta=current_calibration-current_training
  replay_delta=replay_calibration-replay_training
  current_weights=self._class_balanced_weights(
   current_labels,current_training.dtype
  )

  def derivative(alpha):
   return 0.5*(
    self._weighted_derivative(
     current_training,current_delta,current_labels,
     self.calibration_label_smoothing,alpha,current_weights
    )+
    self._weighted_derivative(
     replay_training,replay_delta,replay_labels,
     self.calibration_label_smoothing,alpha,replay_weights
    )
   )

  at_zero=derivative(0.0); at_one=derivative(1.0)
  if not bool(torch.isfinite(torch.stack([at_zero,at_one])).all()):
   raise FloatingPointError('nonfinite consequence-aware derivative')
  low=torch.zeros_like(at_zero); high=torch.ones_like(at_zero)
  for _ in range(self._ARBITRATION_BISECTION_STEPS):
   middle=(low+high)/2.0; value=derivative(middle)
   low=torch.where(value<=0.0,middle,low)
   high=torch.where(value<=0.0,high,middle)
  interior=(low+high)/2.0
  alpha=torch.where(
   at_zero>=0.0,torch.zeros_like(interior),
   torch.where(at_one<=0.0,torch.ones_like(interior),interior),
  )
  if not bool(torch.isfinite(alpha)):
   raise FloatingPointError('nonfinite consequence-aware alpha')
  return float(alpha.detach().cpu())

 def _record_arbitration_logits(
  self,current_training,current_calibration,current_labels,
  replay_training,replay_calibration,replay_labels,started,
 ):
  try:
   with torch.no_grad():
    replay_weights,risk_audit=self._replay_consequence_weights(
     replay_training,replay_calibration,replay_labels,
     self.calibration_label_smoothing,self._coupling_lagged_scores,
    )
    alpha=self._optimal_coupled_alpha(
     current_training,current_calibration,current_labels,
     replay_training,replay_calibration,replay_labels,replay_weights,
    )
    mixed_current=torch.lerp(current_training,current_calibration,alpha)
    mixed_replay=torch.lerp(replay_training,replay_calibration,alpha)
    current_base=self._class_balanced_smoothed_loss(
     current_training,current_labels,self.calibration_label_smoothing
    )
    current_mixed=self._class_balanced_smoothed_loss(
     mixed_current,current_labels,self.calibration_label_smoothing
    )
    replay_base=self._weighted_loss(
     replay_training,replay_labels,self.calibration_label_smoothing,
     replay_weights,
    )
    replay_mixed=self._weighted_loss(
     mixed_replay,replay_labels,self.calibration_label_smoothing,
     replay_weights,
    )
    weighted_joint_base=0.5*(current_base+replay_base)
    weighted_joint_mixed=0.5*(current_mixed+replay_mixed)
    unweighted_joint_base=self._joint_balanced_smoothed_loss(
     current_training,current_labels,replay_training,replay_labels,
     self.calibration_label_smoothing,
    )
    unweighted_joint_mixed=self._joint_balanced_smoothed_loss(
     mixed_current,current_labels,mixed_replay,replay_labels,
     self.calibration_label_smoothing,
    )
    values=torch.stack([
     current_base,current_mixed,replay_base,replay_mixed,
     weighted_joint_base,weighted_joint_mixed,
     unweighted_joint_base,unweighted_joint_mixed,
    ])
    if not bool(torch.isfinite(values).all()):
     raise FloatingPointError('nonfinite consequence-aware loss')
    current_delta,replay_improvement,weighted_improvement,unweighted_improvement=(
     torch.stack([
      current_mixed-current_base,
      replay_base-replay_mixed,
      weighted_joint_base-weighted_joint_mixed,
      unweighted_joint_base-unweighted_joint_mixed,
     ]).detach().cpu().tolist()
    )
  except FloatingPointError:
   self._arbitration_nonfinite_skips+=1
   self._arbitration_host_seconds+=time.perf_counter()-started
   return

  self._arbitration_batches+=1
  self._arbitration_alpha_sum+=alpha
  self._arbitration_alpha=self._arbitration_alpha_sum/self._arbitration_batches
  self._arbitration_alpha_min=(
   alpha if self._arbitration_alpha_min is None
   else min(self._arbitration_alpha_min,alpha)
  )
  self._arbitration_alpha_max=(
   alpha if self._arbitration_alpha_max is None
   else max(self._arbitration_alpha_max,alpha)
  )
  tolerance=2.0**(-self._ARBITRATION_BISECTION_STEPS)
  if alpha<=tolerance: self._arbitration_zero_count+=1
  elif alpha>=1.0-tolerance: self._arbitration_one_count+=1
  else: self._arbitration_interior_count+=1
  self._arbitration_current_ce_delta_sum+=current_delta
  self._arbitration_current_ce_delta_max=max(
   self._arbitration_current_ce_delta_max,current_delta
  )
  self._arbitration_replay_ce_improvement_sum+=replay_improvement
  self._arbitration_joint_ce_improvement_sum+=weighted_improvement
  self._arbitration_host_seconds+=time.perf_counter()-started

  class_count=len(risk_audit['classes'])
  self._coupling_batches+=1
  self._coupling_fallback_batches+=int(risk_audit['fallback_to_uniform'])
  self._coupling_class_observations+=class_count
  self._coupling_mature_class_observations+=sum(
   int(self._coupling_lagged_repeat_counts.get(label,0)>0)
   for label in risk_audit['classes']
  )
  self._coupling_instability_sum+=sum(risk_audit['instability'])
  self._coupling_consequence_sum+=sum(risk_audit['consequence'])
  self._coupling_risk_sum+=sum(risk_audit['risk'])
  self._coupling_unweighted_joint_improvement_sum+=unweighted_improvement
  if risk_audit['class_mass']:
   local_min=min(risk_audit['class_mass'])
   self._coupling_min_class_mass=(
    local_min if self._coupling_min_class_mass is None
    else min(self._coupling_min_class_mass,local_min)
   )
   self._coupling_max_class_mass=max(
    self._coupling_max_class_mass,max(risk_audit['class_mass'])
   )

 def _add_auxiliary_loss(
  self,current_x,current_y,current_tid,replay,replay_output=None
 ):
  # Snapshot the exact lagged scores used by Layer 2 before the current replay
  # observation is incorporated by SelfReferencedReplayDeteriorationAuditERACE.
  self._coupling_lagged_scores=dict(self._persistent_scores)
  self._coupling_lagged_repeat_counts={
   int(label):int(state['repeat_event_count'])
   for label,state in self._srrd_classes.items()
  }
  try:
   super()._add_auxiliary_loss(
    current_x,current_y,current_tid,replay,replay_output
   )
  finally:
   self._coupling_lagged_scores={}
   self._coupling_lagged_repeat_counts={}

 def rbcl_summary(self):
  r=super().rbcl_summary()
  batches=max(1,self._coupling_batches)
  classes=max(1,self._coupling_class_observations)
  arbitration=r['risk_budgeted_head_arbitration']
  arbitration.update({
   'function':'SRRD-coupled consequence-aware prequential risk arbitration',
   'batch_objective':'equal current/replay mass with lagged instability-times-consequence replay class mass',
   'alpha_source':'online cumulative mean of consequence-aware pre-update convex optima',
   'mean_joint_ce_improvement':(
    self._arbitration_joint_ce_improvement_sum/
    max(1,self._arbitration_batches)
   ),
   'joint_ce_improvement_semantics':'coupled weighted objective',
  })
  r['srrd_consequence_coupling']={
   'enabled':self.calibration_lr_scale>0.0,
   'frozen_parent_strategy':'persistent_srrd_prequential_arbitration_1',
   'risk_definition':'lagged Persistent-SRRD instability * absolute pre-update class-mean decision-direction derivative',
   'current_risk_mass':0.5,
   'replay_risk_mass':0.5,
   'replay_class_mass':'normalized instability-times-consequence; uniform only when total risk is zero/nonfinite',
   'has_tunable_coupling_coefficient':False,
   'lagged_srrd_snapshot_is_taken_before_current_replay_observation':True,
   'decision_consequence_uses_pre_update_logits_only':True,
   'class_masses_are_fixed_before_convex_alpha_search':True,
   'coupling_batches':self._coupling_batches,
   'uniform_fallback_batches':self._coupling_fallback_batches,
   'uniform_fallback_fraction':self._coupling_fallback_batches/batches,
   'class_observations':self._coupling_class_observations,
   'mature_class_observations':self._coupling_mature_class_observations,
   'mature_class_fraction':self._coupling_mature_class_observations/classes,
   'mean_instability':self._coupling_instability_sum/classes,
   'mean_consequence':self._coupling_consequence_sum/classes,
   'mean_raw_risk':self._coupling_risk_sum/classes,
   'minimum_replay_class_mass':self._coupling_min_class_mass,
   'maximum_replay_class_mass':self._coupling_max_class_mass,
   'mean_unweighted_joint_ce_improvement':(
    self._coupling_unweighted_joint_improvement_sum/batches
   ),
   'additional_replay_draws':0,
   'additional_backbone_forwards':0,
   'additional_head_only_forwards':0,
   'uses_task_id_boundary_validation_pss_or_future_data':False,
  }
  return r


class ParetoGuardedReplayRepairERACE(SelectivePersistentSRRDSwapERACE):
 """Apply a reversible replay-only classifier-head repair after layer 2."""

 def __init__(self,*args,repair_step_scale:float=1.0,repair_interval:int=4,repair_current_ce_tolerance:float=1e-6,repair_min_replay_ce_improvement:float=1e-6,**kwargs):
  if repair_step_scale<0.0: raise ValueError('repair_step_scale must be nonnegative')
  if repair_interval<=0: raise ValueError('repair_interval must be positive')
  if repair_current_ce_tolerance<0.0: raise ValueError('repair_current_ce_tolerance must be nonnegative')
  if repair_min_replay_ce_improvement<0.0: raise ValueError('repair_min_replay_ce_improvement must be nonnegative')
  super().__init__(*args,**kwargs)
  self.repair_step_scale=float(repair_step_scale); self.repair_interval=int(repair_interval); self.repair_current_ce_tolerance=float(repair_current_ce_tolerance); self.repair_min_replay_ce_improvement=float(repair_min_replay_ce_improvement)
  self._repair_last_batch=None; self._repair_replay_calls=0; self._repair_attempts=0; self._repair_accepts=0; self._repair_guard_rejections=0; self._repair_budget_abstentions=0; self._repair_nonfinite_rejections=0; self._repair_current_ce_delta_sum=0.0; self._repair_replay_ce_improvement_sum=0.0; self._repair_max_accepted_current_ce_delta=0.0; self._repair_min_accepted_replay_ce_improvement=None
 @staticmethod
 def _repair_is_due(replay_call,repair_interval,repair_step_scale):
  return float(repair_step_scale)>0.0 and int(replay_call)>0 and int(replay_call)%int(repair_interval)==0
 @staticmethod
 def _repair_decision(current_ce_delta,replay_ce_improvement,current_tolerance,min_replay_improvement):
  values=(current_ce_delta,replay_ce_improvement,current_tolerance,min_replay_improvement)
  if not all(math.isfinite(float(value)) for value in values): return False
  return float(current_ce_delta)<=float(current_tolerance) and float(replay_ce_improvement)>=float(min_replay_improvement)
 def _add_auxiliary_loss(self,current_x,current_y,current_tid,replay,replay_output=None):
  super()._add_auxiliary_loss(current_x,current_y,current_tid,replay,replay_output)
  self._repair_last_batch=None
  if self.repair_step_scale<=0.0 or replay is None: return
  self._repair_last_batch=(current_x.detach(),current_y.detach(),current_tid.detach(),replay[0].detach(),replay[1].detach(),replay[2].detach())
 @staticmethod
 def _head_learning_rate(optimizer,parameters):
  parameter_ids={id(parameter) for parameter in parameters}; learning_rates=[]
  for group in optimizer.param_groups:
   if any(id(parameter) in parameter_ids for parameter in group['params']): learning_rates.append(float(group['lr']))
  if not learning_rates: raise RuntimeError('classifier head parameters are absent from optimizer')
  return min(learning_rates)
 @staticmethod
 def _capture_head_features(model,head,x,task_ids):
  captured={}
  def hook(_module,inputs): captured['features']=inputs[0].detach()
  handle=head.register_forward_pre_hook(hook)
  try:
   with torch.no_grad(): avalanche_forward(model,x,task_ids)
  finally: handle.remove()
  if 'features' not in captured: raise RuntimeError('classifier head input was not observed')
  return captured['features']
 def _attempt_repair(self,batch):
  current_x,current_y,current_tid,replay_x,replay_y,replay_tid=batch
  _,head=get_last_fc_layer(self.model); parameters=tuple(parameter for parameter in head.parameters() if parameter.requires_grad)
  if not parameters: return False
  was_training=self.model.training; self.model.eval(); snapshots=[parameter.detach().clone() for parameter in parameters]
  try:
   joined_x=torch.cat([current_x,replay_x],dim=0); joined_tid=torch.cat([current_tid,replay_tid],dim=0); features=self._capture_head_features(self.model,head,joined_x,joined_tid); current_count=int(current_y.numel())
   logits=head(features); current_before=F.cross_entropy(logits[:current_count],current_y); replay_before=F.cross_entropy(logits[current_count:],replay_y); gradients=torch.autograd.grad(replay_before,parameters)
   step=self.repair_step_scale*self._head_learning_rate(self.optimizer,parameters)
   with torch.no_grad():
    for parameter,gradient in zip(parameters,gradients): parameter.add_(gradient,alpha=-step)
    candidate=head(features); current_after=F.cross_entropy(candidate[:current_count],current_y); replay_after=F.cross_entropy(candidate[current_count:],replay_y)
   current_delta=float((current_after-current_before.detach()).cpu()); replay_improvement=float((replay_before.detach()-replay_after).cpu()); finite=math.isfinite(current_delta) and math.isfinite(replay_improvement)
   accepted=self._repair_decision(current_delta,replay_improvement,self.repair_current_ce_tolerance,self.repair_min_replay_ce_improvement)
   if not accepted:
    with torch.no_grad():
     for parameter,snapshot in zip(parameters,snapshots): parameter.copy_(snapshot)
    self._repair_guard_rejections+=int(finite); self._repair_nonfinite_rejections+=int(not finite); return False
   self._repair_current_ce_delta_sum+=current_delta; self._repair_replay_ce_improvement_sum+=replay_improvement; self._repair_max_accepted_current_ce_delta=max(self._repair_max_accepted_current_ce_delta,current_delta); self._repair_min_accepted_replay_ce_improvement=replay_improvement if self._repair_min_accepted_replay_ce_improvement is None else min(self._repair_min_accepted_replay_ce_improvement,replay_improvement); return True
  finally: self.model.train(was_training)
 def _after_anchor_update(self,current_x,current_y,current_tid):
  super()._after_anchor_update(current_x,current_y,current_tid)
  batch=self._repair_last_batch; self._repair_last_batch=None
  if self.repair_step_scale<=0.0 or batch is None: return
  self._repair_replay_calls+=1
  if not self._repair_is_due(self._repair_replay_calls,self.repair_interval,self.repair_step_scale): self._repair_budget_abstentions+=1; return
  self._repair_attempts+=1; self._repair_accepts+=int(self._attempt_repair(batch))
 def rbcl_summary(self):
  r=super().rbcl_summary(); calls=max(1,self._repair_replay_calls); attempts=max(1,self._repair_attempts); accepts=max(1,self._repair_accepts); enabled=self.repair_step_scale>0.0
  r['pareto_guarded_replay_repair']={'enabled':enabled,'parent_layer2':'persistent_srrd_selective_swap_1','function':'reversible replay-only classifier-head repair','repair_step_scale':self.repair_step_scale,'repair_interval':self.repair_interval,'maximum_attempt_fraction':1.0/self.repair_interval,'current_ce_tolerance':self.repair_current_ce_tolerance,'minimum_replay_ce_improvement':self.repair_min_replay_ce_improvement,'replay_calls':self._repair_replay_calls,'attempt_count':self._repair_attempts,'accepted_count':self._repair_accepts,'guard_rejection_count':self._repair_guard_rejections,'budget_abstention_count':self._repair_budget_abstentions,'nonfinite_rejection_count':self._repair_nonfinite_rejections,'attempt_rate':self._repair_attempts/calls,'accepted_rate':self._repair_accepts/calls,'acceptance_given_attempt':self._repair_accepts/attempts,'mean_accepted_current_ce_delta':self._repair_current_ce_delta_sum/accepts,'mean_accepted_replay_ce_improvement':self._repair_replay_ce_improvement_sum/accepts,'maximum_accepted_current_ce_delta':self._repair_max_accepted_current_ce_delta,'minimum_accepted_replay_ce_improvement':self._repair_min_accepted_replay_ce_improvement,'only_classifier_head_is_modified':True,'main_optimizer_state_is_not_modified_by_repair':True,'memory_and_replay_identities_are_unchanged':True,'layer2_signal_does_not_control_repair':True,'uses_arrived_current_and_replay_batches_only':True,'uses_task_id_boundary_validation_pss_or_future_data':False,'zero_step_scale_is_exact_layer2_noop':True}; return r


class MemoryCertifiedReplayRepairERACE(ParetoGuardedReplayRepairERACE):
 """Apply head repair only when all represented memory classes are guarded."""

 def __init__(self,*args,repair_worst_class_ce_tolerance:float=1e-6,**kwargs):
  if repair_worst_class_ce_tolerance<0.0: raise ValueError('repair_worst_class_ce_tolerance must be nonnegative')
  super().__init__(*args,**kwargs)
  self.repair_worst_class_ce_tolerance=float(repair_worst_class_ce_tolerance)
  self._repair_max_accepted_worst_class_ce_delta=0.0; self._repair_min_accepted_balanced_memory_ce_improvement=None
  self._repair_min_memory_size=None; self._repair_max_memory_size=0; self._repair_min_class_count=None; self._repair_max_class_count=0
 @staticmethod
 def _class_mean_ce(logits,labels):
  losses=F.cross_entropy(logits,labels,reduction='none'); classes=torch.unique(labels,sorted=True)
  return classes,torch.stack([losses[labels==label].mean() for label in classes])
 @staticmethod
 def _memory_repair_decision(current_ce_delta,memory_improvement,worst_class_delta,current_tolerance,min_memory_improvement,worst_class_tolerance):
  values=(current_ce_delta,memory_improvement,worst_class_delta,current_tolerance,min_memory_improvement,worst_class_tolerance)
  if not all(math.isfinite(float(value)) for value in values): return False
  return float(current_ce_delta)<=float(current_tolerance) and float(memory_improvement)>=float(min_memory_improvement) and float(worst_class_delta)<=float(worst_class_tolerance)
 def _attempt_repair(self,batch):
  current_x,current_y,current_tid,_,_,_=batch
  if not self._memory_x: return False
  _,head=get_last_fc_layer(self.model); parameters=tuple(parameter for parameter in head.parameters() if parameter.requires_grad)
  if not parameters: return False
  memory_x=torch.stack(self._memory_x).to(self.device); memory_y=torch.as_tensor(self._memory_y,device=self.device,dtype=torch.long); memory_tid=torch.as_tensor(self._memory_tid,device=self.device,dtype=torch.long)
  was_training=self.model.training; self.model.eval(); snapshots=[parameter.detach().clone() for parameter in parameters]
  try:
   joined_x=torch.cat([current_x,memory_x],dim=0); joined_tid=torch.cat([current_tid,memory_tid],dim=0)
   features=self._capture_head_features(self.model,head,joined_x,joined_tid); current_count=int(current_y.numel())
   logits=head(features); current_before=F.cross_entropy(logits[:current_count],current_y)
   _,memory_before_by_class=self._class_mean_ce(logits[current_count:],memory_y); memory_before=memory_before_by_class.mean()
   gradients=torch.autograd.grad(memory_before,parameters); step=self.repair_step_scale*self._head_learning_rate(self.optimizer,parameters)
   with torch.no_grad():
    for parameter,gradient in zip(parameters,gradients): parameter.add_(gradient,alpha=-step)
    candidate=head(features); current_after=F.cross_entropy(candidate[:current_count],current_y)
    _,memory_after_by_class=self._class_mean_ce(candidate[current_count:],memory_y)
   current_delta=float((current_after-current_before.detach()).cpu()); memory_improvement=float((memory_before.detach()-memory_after_by_class.mean()).cpu()); worst_class_delta=float((memory_after_by_class-memory_before_by_class.detach()).max().cpu())
   finite=all(math.isfinite(value) for value in (current_delta,memory_improvement,worst_class_delta))
   accepted=self._memory_repair_decision(current_delta,memory_improvement,worst_class_delta,self.repair_current_ce_tolerance,self.repair_min_replay_ce_improvement,self.repair_worst_class_ce_tolerance)
   if not accepted:
    with torch.no_grad():
     for parameter,snapshot in zip(parameters,snapshots): parameter.copy_(snapshot)
    self._repair_guard_rejections+=int(finite); self._repair_nonfinite_rejections+=int(not finite); return False
   memory_size=int(memory_y.numel()); class_count=int(memory_before_by_class.numel())
   self._repair_current_ce_delta_sum+=current_delta; self._repair_replay_ce_improvement_sum+=memory_improvement
   self._repair_max_accepted_current_ce_delta=max(self._repair_max_accepted_current_ce_delta,current_delta)
   self._repair_min_accepted_replay_ce_improvement=memory_improvement if self._repair_min_accepted_replay_ce_improvement is None else min(self._repair_min_accepted_replay_ce_improvement,memory_improvement)
   self._repair_max_accepted_worst_class_ce_delta=max(self._repair_max_accepted_worst_class_ce_delta,worst_class_delta)
   self._repair_min_accepted_balanced_memory_ce_improvement=memory_improvement if self._repair_min_accepted_balanced_memory_ce_improvement is None else min(self._repair_min_accepted_balanced_memory_ce_improvement,memory_improvement)
   self._repair_min_memory_size=memory_size if self._repair_min_memory_size is None else min(self._repair_min_memory_size,memory_size); self._repair_max_memory_size=max(self._repair_max_memory_size,memory_size)
   self._repair_min_class_count=class_count if self._repair_min_class_count is None else min(self._repair_min_class_count,class_count); self._repair_max_class_count=max(self._repair_max_class_count,class_count); return True
  finally: self.model.train(was_training)
 def rbcl_summary(self):
  r=super().rbcl_summary(); repair=r.pop('pareto_guarded_replay_repair')
  repair.update({'function':'reversible class-balanced full-memory classifier-head repair','minimum_balanced_memory_ce_improvement':repair.pop('minimum_replay_ce_improvement'),'mean_accepted_balanced_memory_ce_improvement':repair.pop('mean_accepted_replay_ce_improvement'),'minimum_accepted_balanced_memory_ce_improvement':self._repair_min_accepted_balanced_memory_ce_improvement,'worst_class_ce_tolerance':self.repair_worst_class_ce_tolerance,'maximum_accepted_worst_class_ce_delta':self._repair_max_accepted_worst_class_ce_delta,'minimum_accepted_memory_size':self._repair_min_memory_size,'maximum_accepted_memory_size':self._repair_max_memory_size,'minimum_accepted_class_count':self._repair_min_class_count,'maximum_accepted_class_count':self._repair_max_class_count,'gradient_uses_class_balanced_full_memory_ce':True,'guard_uses_every_represented_memory_class':True,'uses_sampled_replay_loss_for_repair_or_guard':False,'uses_arrived_current_and_past_memory_only':True})
  repair.pop('minimum_accepted_replay_ce_improvement'); r['memory_certified_replay_repair']=repair; return r


class PersistentSRRDLossRedistributionERACE(SelectivePersistentSRRDSwapERACE):
 """Transfer bounded replay-loss mass after a confident one-swap correction."""

 def __init__(self,*args,loss_transfer:float=.5,**kwargs):
  if not 0.0<=loss_transfer<=1.0: raise ValueError('loss_transfer must be in [0,1]')
  super().__init__(*args,**kwargs)
  self.loss_transfer=float(loss_transfer); self._last_replay_loss_weights=None; self._loss_transfer_replay_calls=0; self._loss_transfer_eligible_swap_batches=0; self._loss_transfer_pair_count=0; self._loss_transfer_confidence_rejections=0; self._loss_transfer_mass_error_max=0.0; self._loss_transfer_min_weight=1.0; self._loss_transfer_max_weight=1.0; self._loss_transfer_target_score_sum=0.0; self._loss_transfer_donor_score_sum=0.0; self._loss_transfer_confidence_gap_min=None; self._loss_transfer_objective_delta_sum=0.0
 def _loss_transfer_weights(self,labels):
  if self.loss_transfer<=0.0 or not self._persistent_last_swap_records: return None
  count=int(labels.numel()); weights=torch.ones(count,dtype=torch.float32,device=labels.device)
  self._loss_transfer_eligible_swap_batches+=1; record=self._persistent_last_swap_records[-1]; target_position=int(record['target_position']); target_label=int(record['target_label']); target_state=self._srrd_classes.get(target_label)
  if target_state is None or target_state['repeat_event_count']<=0:
   self._loss_transfer_confidence_rejections+=1; return None
  target_low,_=self._wilson_interval(target_state['positive_ce_event_count'],target_state['repeat_event_count'],self.confidence_z); candidates=[]
  for position,label_value in enumerate(labels.detach().cpu().tolist()):
   if position==target_position: continue
   label=int(label_value); state=self._srrd_classes.get(label)
   if state is None or state['repeat_event_count']<=0: continue
   _,donor_high=self._wilson_interval(state['positive_ce_event_count'],state['repeat_event_count'],self.confidence_z); gap=target_low-donor_high
   if gap>0.0: candidates.append((gap,-donor_high,-float(self._persistent_scores.get(label,1.0)),-label,-position,position,label,donor_high))
  if not candidates:
   self._loss_transfer_confidence_rejections+=1; return None
  best=max(candidates); donor_position=int(best[-3]); donor_label=int(best[-2]); donor_high=float(best[-1]); gap=float(target_low-donor_high)
  weights[target_position]+=self.loss_transfer; weights[donor_position]-=self.loss_transfer; mass_error=abs(float(weights.sum().detach())-float(count)); self._loss_transfer_mass_error_max=max(self._loss_transfer_mass_error_max,mass_error); self._loss_transfer_min_weight=min(self._loss_transfer_min_weight,float(weights.min().detach())); self._loss_transfer_max_weight=max(self._loss_transfer_max_weight,float(weights.max().detach())); self._loss_transfer_pair_count+=1; self._loss_transfer_target_score_sum+=float(self._persistent_scores.get(target_label,1.0)); self._loss_transfer_donor_score_sum+=float(self._persistent_scores.get(donor_label,1.0)); self._loss_transfer_confidence_gap_min=gap if self._loss_transfer_confidence_gap_min is None else min(self._loss_transfer_confidence_gap_min,gap)
  return weights
 def _sample_memory(self):
  replay=super()._sample_memory(); self._last_replay_loss_weights=None
  if replay is None: return None
  self._loss_transfer_replay_calls+=1; self._last_replay_loss_weights=self._loss_transfer_weights(replay[1]); return replay
 @staticmethod
 def _redistributed_replay_loss(replay_output,labels,weights):
  losses=F.cross_entropy(replay_output,labels,reduction='none'); return (losses*weights).sum()/weights.sum()
 def _add_auxiliary_loss(self,current_x,current_y,current_tid,replay,replay_output=None):
  if replay is not None and replay_output is not None and self._last_replay_loss_weights is not None:
   weighted=self._redistributed_replay_loss(replay_output,replay[1],self._last_replay_loss_weights); base=self._auxiliary_replay_loss if self._auxiliary_replay_loss is not None else F.cross_entropy(replay_output,replay[1]); delta=.5*(weighted-base); self.loss+=delta; self._loss_transfer_objective_delta_sum+=float(delta.detach())
  super()._add_auxiliary_loss(current_x,current_y,current_tid,replay,replay_output)
 def rbcl_summary(self):
  r=super().rbcl_summary(); pairs=max(1,self._loss_transfer_pair_count); enabled=self.loss_transfer>0.0; r['persistent_srrd_loss_redistribution']={'enabled':enabled,'parent_layer2':'persistent_srrd_selective_swap_1','loss_transfer_units':self.loss_transfer,'minimum_replay_weight':1.0-self.loss_transfer,'maximum_replay_weight':1.0+self.loss_transfer,'replay_calls':self._loss_transfer_replay_calls,'eligible_swap_batches':self._loss_transfer_eligible_swap_batches,'pair_count':self._loss_transfer_pair_count,'batches_without_transfer':self._loss_transfer_replay_calls-self._loss_transfer_pair_count,'confidence_rejected_batches':self._loss_transfer_confidence_rejections,'transfer_applies_only_after_layer2_swap':True,'total_replay_weight_fixed':True,'maximum_observed_mass_error':self._loss_transfer_mass_error_max,'minimum_observed_weight':self._loss_transfer_min_weight,'maximum_observed_weight':self._loss_transfer_max_weight,'mean_target_persistence':self._loss_transfer_target_score_sum/pairs,'mean_donor_persistence':self._loss_transfer_donor_score_sum/pairs,'minimum_wilson_gap':self._loss_transfer_confidence_gap_min,'mean_objective_delta':self._loss_transfer_objective_delta_sum/pairs,'lagged_signal_is_read_before_current_batch_update':True,'uses_current_or_future_loss_to_choose_weights':False,'uses_historical_logits_or_margins_as_targets':False,'uses_task_id_gradient_validation_pss_or_future_data':False,'main_sampler_rng_receives_no_extra_draws':True,'independent_rng_receives_no_extra_draws':True,'zero_transfer_is_exact_layer2_noop':True}; return r


class SupportCalibratedPersistentSRRDLossRedistributionERACE(PersistentSRRDLossRedistributionERACE):
 """Require replicated, independently supported evidence before loss transfer."""

 def __init__(self,*args,min_distinct_items_per_partition:int=2,**kwargs):
  if min_distinct_items_per_partition<1: raise ValueError('min_distinct_items_per_partition must be positive')
  super().__init__(*args,**kwargs)
  self.min_distinct_items_per_partition=int(min_distinct_items_per_partition); self._support_partition_stats={}; self._support_event_count=0; self._support_candidate_evaluations=0; self._support_insufficient_rejection_batches=0; self._support_partition_overlap_rejection_batches=0; self._support_selected_target_distinct_min=None; self._support_selected_donor_distinct_min=None
 @staticmethod
 def _support_hash_value(key):
  return int(str(key).rsplit(':',1)[-1],16)
 @classmethod
 def _support_partition_and_bit(cls,key,repeat_ordinal):
  value=cls._support_hash_value(key); partition=(value+int(repeat_ordinal))%2; bit=1<<((value>>8)%64); return partition,bit
 @staticmethod
 def _empty_support_partition():
  return {'repeat_event_count':0,'positive_ce_event_count':0,'item_bitmap':0}
 def _support_class_partitions(self,label):
  return self._support_partition_stats.setdefault(int(label),[self._empty_support_partition(),self._empty_support_partition()])
 def _update_support_statistics(self,replay,replay_output):
  if replay is None or replay_output is None or len(self._srrd_last_keys)!=int(replay[1].numel()): return
  with torch.no_grad():
   device_labels=replay[1].detach(); losses=F.cross_entropy(replay_output.detach(),device_labels,reduction='none').cpu(); labels=device_labels.cpu(); grouped={}
   for position,key in enumerate(self._srrd_last_keys): grouped.setdefault(key,[]).append(position)
   for key,positions in grouped.items():
    item=self._srrd_items.get(key)
    if item is None or int(item['observation_count'])<2 or item['best_ema_ce'] is None: continue
    selected=torch.tensor(positions,dtype=torch.long); label=int(labels[positions[0]]); current_ce=float(losses[selected].mean()); positive=int(current_ce>float(item['best_ema_ce'])+1e-12); repeat_ordinal=int(item['observation_count'])-1; partition,bit=self._support_partition_and_bit(key,repeat_ordinal); state=self._support_class_partitions(label)[partition]; state['repeat_event_count']+=1; state['positive_ce_event_count']+=positive; state['item_bitmap']|=bit; self._support_event_count+=1
 def _partition_interval(self,label,partition):
  states=self._support_partition_stats.get(int(label))
  if states is None: return None
  state=states[int(partition)]; distinct=int(state['item_bitmap']).bit_count()
  if distinct<self.min_distinct_items_per_partition or state['repeat_event_count']<=0: return None
  low,high=self._wilson_interval(state['positive_ce_event_count'],state['repeat_event_count'],self.confidence_z); return low,high,distinct
 def _replicated_support_gap(self,target_label,donor_label):
  gaps=[]; target_distinct=[]; donor_distinct=[]
  for partition in range(2):
   target=self._partition_interval(target_label,partition); donor=self._partition_interval(donor_label,partition)
   if target is None or donor is None: return None,'insufficient_support',None,None
   gap=float(target[0]-donor[1])
   if gap<=0.0: return None,'partition_overlap',None,None
   gaps.append(gap); target_distinct.append(int(target[2])); donor_distinct.append(int(donor[2]))
  return min(gaps),'accepted',min(target_distinct),min(donor_distinct)
 def _donor_mass_guard(self,target_label,donor_label,label_counts):
  return True
 def _record_selected_mass_guard(self,target_label,donor_label,label_counts):
  return None
 def _loss_transfer_weights(self,labels):
  if self.loss_transfer<=0.0 or not self._persistent_last_swap_records: return None
  count=int(labels.numel()); weights=torch.ones(count,dtype=torch.float32,device=labels.device); self._loss_transfer_eligible_swap_batches+=1; record=self._persistent_last_swap_records[-1]; target_position=int(record['target_position']); target_label=int(record['target_label']); candidates=[]; saw_overlap=False; label_values=labels.detach().cpu().tolist(); label_counts={}
  for label_value in label_values: label_counts[int(label_value)]=label_counts.get(int(label_value),0)+1
  for position,label_value in enumerate(label_values):
   if position==target_position: continue
   donor_label=int(label_value)
   if not self._donor_mass_guard(target_label,donor_label,label_counts): continue
   self._support_candidate_evaluations+=1; gap,reason,target_distinct,donor_distinct=self._replicated_support_gap(target_label,donor_label)
   if reason=='partition_overlap': saw_overlap=True
   if gap is None: continue
   candidates.append((float(gap),-float(self._persistent_scores.get(donor_label,1.0)),-donor_label,-position,position,donor_label,int(target_distinct),int(donor_distinct)))
  if not candidates:
   self._loss_transfer_confidence_rejections+=1
   if saw_overlap: self._support_partition_overlap_rejection_batches+=1
   else: self._support_insufficient_rejection_batches+=1
   return None
  best=max(candidates); gap=float(best[0]); donor_position=int(best[4]); donor_label=int(best[5]); target_distinct=int(best[6]); donor_distinct=int(best[7]); self._record_selected_mass_guard(target_label,donor_label,label_counts); weights[target_position]+=self.loss_transfer; weights[donor_position]-=self.loss_transfer; mass_error=abs(float(weights.sum().detach())-float(count)); self._loss_transfer_mass_error_max=max(self._loss_transfer_mass_error_max,mass_error); self._loss_transfer_min_weight=min(self._loss_transfer_min_weight,float(weights.min().detach())); self._loss_transfer_max_weight=max(self._loss_transfer_max_weight,float(weights.max().detach())); self._loss_transfer_pair_count+=1; self._loss_transfer_target_score_sum+=float(self._persistent_scores.get(target_label,1.0)); self._loss_transfer_donor_score_sum+=float(self._persistent_scores.get(donor_label,1.0)); self._loss_transfer_confidence_gap_min=gap if self._loss_transfer_confidence_gap_min is None else min(self._loss_transfer_confidence_gap_min,gap); self._support_selected_target_distinct_min=target_distinct if self._support_selected_target_distinct_min is None else min(self._support_selected_target_distinct_min,target_distinct); self._support_selected_donor_distinct_min=donor_distinct if self._support_selected_donor_distinct_min is None else min(self._support_selected_donor_distinct_min,donor_distinct)
  return weights
 def _add_auxiliary_loss(self,current_x,current_y,current_tid,replay,replay_output=None):
  super()._add_auxiliary_loss(current_x,current_y,current_tid,replay,replay_output); self._update_support_statistics(replay,replay_output)
 def rbcl_summary(self):
  r=super().rbcl_summary(); transfer=r['persistent_srrd_loss_redistribution']; eligible=max(1,self._loss_transfer_eligible_swap_batches); partition_events=[sum(states[index]['repeat_event_count'] for states in self._support_partition_stats.values()) for index in range(2)]; partition_distinct=[sum(int(states[index]['item_bitmap']).bit_count() for states in self._support_partition_stats.values()) for index in range(2)]; transfer.update({'controller_mode':'support_calibrated_split_evidence_abstention','support_partition_count':2,'support_partition_rule':'deterministic_item_hash_plus_repeat_ordinal_parity','support_partitions_are_disjoint':True,'support_bitmap_bits_per_class_partition':64,'minimum_distinct_items_per_partition':self.min_distinct_items_per_partition,'support_statistics_update_after_current_transfer_decision':True,'replicated_wilson_non_overlap_required_in_every_partition':True,'support_event_count':self._support_event_count,'partition_repeat_event_counts':partition_events,'partition_distinct_item_estimates':partition_distinct,'support_candidate_evaluations':self._support_candidate_evaluations,'insufficient_support_rejection_batches':self._support_insufficient_rejection_batches,'partition_overlap_rejection_batches':self._support_partition_overlap_rejection_batches,'transfer_activation_rate':self._loss_transfer_pair_count/eligible,'minimum_selected_target_distinct_items_per_partition':self._support_selected_target_distinct_min,'minimum_selected_donor_distinct_items_per_partition':self._support_selected_donor_distinct_min,'retains_raw_evicted_samples_or_historical_logits':False,'support_gate_uses_current_batch_for_current_weights':False,'zero_transfer_is_exact_support_calibrated_noop':True}); r['persistent_srrd_support_calibrated_loss_redistribution']=dict(transfer); return r


class SupportBalancedPersistentSRRDLossRedistributionERACE(SupportCalibratedPersistentSRRDLossRedistributionERACE):
 """Transfer only from a replay class with more batch mass than the target."""

 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs); self._mass_guard_rejection_candidates=0; self._mass_guard_rejection_batches=0; self._mass_guard_rejected_current_batch=False; self._selected_donor_count_min=None; self._selected_target_count_max=0; self._selected_class_count_gap_min=None
 def _donor_mass_guard(self,target_label,donor_label,label_counts):
  accepted=int(label_counts.get(int(donor_label),0))>int(label_counts.get(int(target_label),0))
  if not accepted: self._mass_guard_rejection_candidates+=1; self._mass_guard_rejected_current_batch=True
  return accepted
 def _record_selected_mass_guard(self,target_label,donor_label,label_counts):
  target_count=int(label_counts[int(target_label)]); donor_count=int(label_counts[int(donor_label)]); gap=donor_count-target_count; self._selected_donor_count_min=donor_count if self._selected_donor_count_min is None else min(self._selected_donor_count_min,donor_count); self._selected_target_count_max=max(self._selected_target_count_max,target_count); self._selected_class_count_gap_min=gap if self._selected_class_count_gap_min is None else min(self._selected_class_count_gap_min,gap)
 def _loss_transfer_weights(self,labels):
  before=self._loss_transfer_eligible_swap_batches; self._mass_guard_rejected_current_batch=False; weights=super()._loss_transfer_weights(labels)
  if self._loss_transfer_eligible_swap_batches>before and weights is None and self._mass_guard_rejected_current_batch: self._mass_guard_rejection_batches+=1
  return weights
 def rbcl_summary(self):
  r=super().rbcl_summary(); transfer=r['persistent_srrd_loss_redistribution']; transfer.update({'controller_mode':'support_calibrated_post_swap_class_mass_abstention','post_swap_target_donor_class_count_guard':True,'donor_class_count_must_exceed_target_class_count':True,'singleton_donor_is_never_downweighted':True,'mass_guard_rejection_candidates':self._mass_guard_rejection_candidates,'mass_guard_rejection_batches':self._mass_guard_rejection_batches,'minimum_selected_donor_class_count':self._selected_donor_count_min,'maximum_selected_target_class_count':self._selected_target_count_max,'minimum_selected_donor_minus_target_class_count':self._selected_class_count_gap_min,'uses_replay_labels_only_for_mass_guard':True,'does_not_interpret_high_exposure_as_vulnerability':True,'zero_transfer_is_exact_support_balanced_noop':True}); r['persistent_srrd_support_balanced_loss_redistribution']=dict(transfer); return r


class TIRPSemanticRepresentativeERACE(SemanticRepresentativeERACE):
 """D33: a fixed offline-learned sampler over D30's 75/25 memory."""
 def __init__(self,*args,policy_checkpoint:str,allow_unapproved_policy:bool=False,preserve_policy_init_rng:bool=False,**kwargs):
  super().__init__(*args,**kwargs)
  checkpoint=torch.load(Path(policy_checkpoint),map_location='cpu',weights_only=True)
  self._tirp_format_version=int(checkpoint.get('format_version',1))
  deployment=checkpoint.get('deployment',{})
  approved=bool(deployment.get('level_one_gate_passed',False))
  exploratory_only=bool(deployment.get('exploratory_only',False))
  if self._tirp_format_version>=2 and not approved:
   if not (allow_unapproved_policy and exploratory_only):
    raise ValueError('TIRP V2 checkpoint is not approved by the level-one replay-policy gate')
  self._tirp_exploratory_only=self._tirp_format_version>=2 and not approved
  sampling=checkpoint.get('sampling',{})
  self._tirp_entropy_floor=float(sampling.get('entropy_floor_ratio',0.0))
  self._tirp_class_coverage_mode=str(sampling.get('class_coverage_mode','maximum_ratio'))
  self._tirp_min_class_coverage=float(sampling.get('min_class_coverage_ratio',0.0))
  self._tirp_boundary_floor=bool(sampling.get('low_quality_quartile_uniform_expected_floor',False))
  self._tirp_projection=checkpoint['projection'].float().to(self.device)
  cpu_rng_state=torch.random.get_rng_state() if preserve_policy_init_rng else None; cuda_rng_states=torch.cuda.get_rng_state_all() if preserve_policy_init_rng and torch.cuda.is_available() else None
  try:
   self._tirp_policy=_TIRPPolicy().to(self.device)
   self._tirp_policy.load_state_dict(checkpoint['state_dict'])
  finally:
   if cpu_rng_state is not None: torch.random.set_rng_state(cpu_rng_state)
   if cuda_rng_states is not None: torch.cuda.set_rng_state_all(cuda_rng_states)
  self._tirp_policy.eval()
  for parameter in self._tirp_policy.parameters(): parameter.requires_grad_(False)
  self._tirp_checkpoint=str(policy_checkpoint)
 def _tirp_inputs(self,x,y,raw=None):
  raw=self._features(x).float().to(self.device) if raw is None else raw
  z=F.normalize(torch.cat((raw,torch.ones((raw.shape[0],1),device=self.device)),dim=1),dim=1) @ self._tirp_projection
  clock=max(1,self._seen_samples,self._hybrid_seen)
  age=torch.tensor([clock-self._memory_arrival.get(self._memory_key(item,label),clock) for item,label in zip(self._memory_x,self._memory_y)],dtype=torch.float32,device=self.device)/clock
  quality=[]
  for feature,label in zip(raw,y.tolist()):
   proto=self.proto_sum.get(label); quality.append(0.0 if proto is None else float(F.cosine_similarity(feature.cpu(),proto.float(),dim=0)))
  class_count=torch.bincount(y,minlength=max(100,int(y.max())+1)).float()[y]/len(y)
  return raw,torch.cat((z,age[:,None],torch.tensor(quality,device=self.device)[:,None],class_count[:,None]),dim=1)
 def _tirp_weights(self,inputs):
  logits=self._tirp_policy(inputs)
  if self._tirp_format_version<2: return torch.softmax(logits,dim=0)
  return constrained_probabilities(logits,self._tirp_entropy_floor)
 def _tirp_sample(self,weights,y,count,quality=None):
  if self._tirp_format_version<2: return torch.multinomial(weights,count,replacement=False)
  class_floor=int(math.ceil(expected_uniform_class_coverage(y,count))) if self._tirp_class_coverage_mode=='uniform_expected' else None
  priority_mask=None; priority_count=0
  if self._tirp_boundary_floor and quality is not None:
   priority_mask=quality<=torch.quantile(quality.float(),.25); priority_count=int(math.ceil(count*float(priority_mask.float().mean())))
  indices,_=sample_without_replacement(weights,y,count,self._tirp_min_class_coverage,min_class_coverage=class_floor,priority_mask=priority_mask,min_priority_count=priority_count)
  return indices
 def _sample_memory(self):
  if not self._memory_x: return None
  count=min(self.batch_size_mem,len(self._memory_x)); x=torch.stack(self._memory_x).to(self.device); y=torch.tensor(self._memory_y,dtype=torch.long,device=self.device)
  with torch.no_grad():
   _,inp=self._tirp_inputs(x,y)
   weights=self._tirp_weights(inp); indices=self._tirp_sample(weights,y,count,inp[:,-2])
   self._last_replay_memory_indices=indices.detach().cpu().tolist()
   self._record_memory_trace_replay(self._last_replay_memory_indices)
  self._replay_examples+=count
  return x[indices],y[indices],torch.tensor([self._memory_tid[int(i)] for i in indices.tolist()],dtype=torch.long,device=self.device)
 def rbcl_summary(self):
  r=super().rbcl_summary(); r['taskification_invariant_replay_policy']={'enabled':True,'checkpoint':self._tirp_checkpoint,'format_version':self._tirp_format_version,'exploratory_only':self._tirp_exploratory_only,'policy_is_fixed_at_deployment':True,'policy_uses_task_id_loss_gradient_or_validation':False,'entropy_floor_ratio':self._tirp_entropy_floor,'class_coverage_mode':self._tirp_class_coverage_mode,'min_class_coverage_ratio':self._tirp_min_class_coverage,'low_quality_quartile_uniform_expected_floor':self._tirp_boundary_floor,'weighted_sampling_without_replacement':True}; return r


class FIRPScoreAuditERACE(TIRPSemanticRepresentativeERACE):
 """Audit frozen TIRP scores while retaining Semantic Memory uniform replay."""
 def __init__(self,*args,**kwargs):
  super().__init__(*args,preserve_policy_init_rng=True,**kwargs)
  self._firp_score_history=[]; self._firp_global=self._firp_empty_stats(); self._firp_exp=self._firp_empty_stats(); self._firp_last=None
 @staticmethod
 def _firp_empty_stats():
  return {'audit_calls':0,'pair_count':0,'score_sum':0.0,'loss_sum':0.0,'score_sq_sum':0.0,'loss_sq_sum':0.0,'score_loss_sum':0.0,'top_count':0,'top_loss_sum':0.0,'top_negative_margin_count':0,'bottom_count':0,'bottom_loss_sum':0.0,'bottom_negative_margin_count':0,'class_score_sum':{},'class_score_count':{}}
 @staticmethod
 def _firp_pearson(stats):
  n=stats['pair_count']; sx=stats['score_sum']; sy=stats['loss_sum']; sxx=stats['score_sq_sum']; syy=stats['loss_sq_sum']; sxy=stats['score_loss_sum']
  if n<2: return 0.0
  first=max(0.0,n*sxx-sx*sx); second=max(0.0,n*syy-sy*sy); denom=math.sqrt(first*second)
  return (n*sxy-sx*sy)/denom if denom>1e-12 else 0.0
 @classmethod
 def _firp_snapshot(cls,stats):
  top=max(1,stats['top_count']); bottom=max(1,stats['bottom_count'])
  labels=sorted(set(stats['class_score_sum']).union(stats['class_score_count']))
  return {'audit_calls':stats['audit_calls'],'pair_count':stats['pair_count'],'score_vs_replay_ce_pearson':cls._firp_pearson(stats),'top_score_quartile':{'count':stats['top_count'],'mean_replay_ce':stats['top_loss_sum']/top,'negative_margin_fraction':stats['top_negative_margin_count']/top},'bottom_score_quartile':{'count':stats['bottom_count'],'mean_replay_ce':stats['bottom_loss_sum']/bottom,'negative_margin_fraction':stats['bottom_negative_margin_count']/bottom},'class_mean_relative_score':{str(label):stats['class_score_sum'].get(label,0.0)/max(1,stats['class_score_count'].get(label,0)) for label in labels}}
 @staticmethod
 def _firp_merge_class_scores(stats,labels,relative_scores):
  for label,score in zip(labels,relative_scores):
   key=int(label); stats['class_score_sum'][key]=stats['class_score_sum'].get(key,0.0)+float(score); stats['class_score_count'][key]=stats['class_score_count'].get(key,0)+1
 @staticmethod
 def _firp_update_pairs(stats,scores,losses,margins,top_mask,bottom_mask):
  scores=scores.float(); losses=losses.float(); margins=margins.float(); count=int(scores.numel())
  stats['pair_count']+=count; stats['score_sum']+=float(scores.sum()); stats['loss_sum']+=float(losses.sum()); stats['score_sq_sum']+=float(scores.square().sum()); stats['loss_sq_sum']+=float(losses.square().sum()); stats['score_loss_sum']+=float((scores*losses).sum())
  for prefix,mask in (('top',top_mask.bool()),('bottom',bottom_mask.bool())):
   selected=int(mask.sum()); stats[f'{prefix}_count']+=selected
   if selected:
    stats[f'{prefix}_loss_sum']+=float(losses[mask].sum()); stats[f'{prefix}_negative_margin_count']+=int(margins[mask].le(0).sum())
 def _before_training_exp(self,**kwargs):
  self._firp_exp=self._firp_empty_stats(); super()._before_training_exp(**kwargs)
 def _after_training_exp(self,**kwargs):
  self._firp_score_history.append(self._firp_snapshot(self._firp_exp)); super()._after_training_exp(**kwargs)
 def _sample_memory(self):
  if not self._memory_x:
   self._firp_last=None; return None
  x=torch.stack(self._memory_x).to(self.device); y=torch.tensor(self._memory_y,dtype=torch.long,device=self.device)
  with torch.no_grad():
   _,inputs=self._tirp_inputs(x,y); weights=self._tirp_weights(inputs); relative=weights*len(weights); low=torch.quantile(relative,.25); high=torch.quantile(relative,.75)
   for stats in (self._firp_exp,self._firp_global):
    stats['audit_calls']+=1; self._firp_merge_class_scores(stats,y.detach().cpu().tolist(),relative.detach().cpu().tolist())
  replay=SemanticRepresentativeERACE._sample_memory(self)
  indices=torch.tensor(self._last_replay_memory_indices,dtype=torch.long,device=self.device); selected=relative[indices]
  self._firp_last={'scores':selected.detach().cpu(),'top':selected.ge(high).detach().cpu(),'bottom':selected.le(low).detach().cpu()}
  return replay
 def _add_auxiliary_loss(self,current_x,current_y,current_tid,replay,replay_output=None):
  if replay is None or replay_output is None or self._firp_last is None: return
  with torch.no_grad():
   losses=F.cross_entropy(replay_output,replay[1],reduction='none').detach().cpu(); chosen=torch.arange(replay_output.shape[0],device=replay_output.device); correct=replay_output[chosen,replay[1]]; rivals=replay_output.clone(); rivals[chosen,replay[1]]=-torch.inf; margins=(correct-rivals.max(dim=1).values).detach().cpu()
   for stats in (self._firp_exp,self._firp_global): self._firp_update_pairs(stats,self._firp_last['scores'],losses,margins,self._firp_last['top'],self._firp_last['bottom'])
 def rbcl_summary(self):
  r=super().rbcl_summary(); r['firp_retention_score_audit']={'enabled':True,'audit_only':True,'policy_is_never_used_for_sampling':True,'training_objective_modified':False,'uniform_replay_parent':'semantic_proto_hybrid_75_25','history':self._firp_score_history,'global':self._firp_snapshot(self._firp_global)}; r['taskification_invariant_replay_policy']['enabled']=False; r['taskification_invariant_replay_policy']['audit_only_score_source']=True; return r


class ClassFIRPERACE(TIRPSemanticRepresentativeERACE):
 """Use frozen FIRP only for a bounded class-level replay quota."""
 def __init__(self,*args,firp_quota_fraction:float=.25,firp_score_refresh_samples:int=512,**kwargs):
  super().__init__(*args,preserve_policy_init_rng=True,**kwargs)
  if not 0.0<=firp_quota_fraction<1.0: raise ValueError('firp_quota_fraction must be in [0,1)')
  if firp_score_refresh_samples<=0: raise ValueError('firp_score_refresh_samples must be positive')
  self.firp_quota_fraction=float(firp_quota_fraction); self.firp_score_refresh_samples=int(firp_score_refresh_samples)
  self._class_firp_scores={}; self._class_firp_last_refresh=-self.firp_score_refresh_samples; self._class_firp_refresh_count=0
  self._class_firp_uniform_draws=0; self._class_firp_quota_draws=0
 @staticmethod
 def _aggregate_class_scores(labels,relative_scores):
  sums={}; counts={}
  for label,score in zip(labels,relative_scores):
   key=int(label); sums[key]=sums.get(key,0.0)+float(score); counts[key]=counts.get(key,0)+1
  means={key:sums[key]/counts[key] for key in sums}
  scale=sum(means.values())/max(1,len(means))
  return {key:value/max(scale,1e-12) for key,value in means.items()}
 @staticmethod
 def _candidate_weights(labels,class_scores):
  counts={}
  for label in labels: counts[int(label)]=counts.get(int(label),0)+1
  return [max(float(class_scores.get(int(label),1.0)),1e-12)/counts[int(label)] for label in labels]
 @staticmethod
 def _weighted_without_replacement(indices,weights,count,rng):
  if count<=0: return []
  if count>=len(indices): return list(indices)
  keys=[]
  for index,weight in zip(indices,weights):
   key=math.log(max(rng.random(),1e-12))/max(float(weight),1e-12)
   keys.append((key,-int(index),int(index)))
  return [item[2] for item in sorted(keys,reverse=True)[:count]]
 @torch.no_grad()
 def _refresh_class_firp_scores(self,x,y,clock):
  _,inputs=self._tirp_inputs(x,y); relative=self._tirp_weights(inputs)*len(y)
  self._class_firp_scores=self._aggregate_class_scores(y.detach().cpu().tolist(),relative.detach().cpu().tolist())
  self._class_firp_last_refresh=int(clock); self._class_firp_refresh_count+=1
 def _sample_memory(self):
  if not self._memory_x: return None
  count=min(self.batch_size_mem,len(self._memory_x))
  if self.firp_quota_fraction<=0.0:
   return SemanticRepresentativeERACE._sample_memory(self)
  x=torch.stack(self._memory_x).to(self.device); y=torch.tensor(self._memory_y,dtype=torch.long,device=self.device)
  clock=max(self._seen_samples,self._hybrid_seen)
  if not self._class_firp_scores or clock-self._class_firp_last_refresh>=self.firp_score_refresh_samples:
   self._refresh_class_firp_scores(x,y,clock)
  quota_count=min(count-1,int(round(count*self.firp_quota_fraction))) if count>1 else 0
  uniform_count=count-quota_count
  uniform_indices=self._rng.sample(range(len(self._memory_x)),uniform_count)
  selected=set(uniform_indices); candidates=[index for index in range(len(self._memory_x)) if index not in selected]
  candidate_labels=[self._memory_y[index] for index in candidates]
  weights=self._candidate_weights(candidate_labels,self._class_firp_scores)
  quota_indices=self._weighted_without_replacement(candidates,weights,quota_count,self._rng)
  indices=uniform_indices+quota_indices
  self._last_replay_memory_indices=list(indices); self._record_memory_trace_replay(indices)
  self._class_firp_uniform_draws+=len(uniform_indices); self._class_firp_quota_draws+=len(quota_indices); self._replay_examples+=count
  chosen=torch.tensor(indices,dtype=torch.long,device=self.device)
  return x[chosen],y[chosen],torch.tensor([self._memory_tid[index] for index in indices],dtype=torch.long,device=self.device)
 def rbcl_summary(self):
  r=super().rbcl_summary(); r['class_firp_replay']={'enabled':self.firp_quota_fraction>0.0,'uniform_replay_parent':'semantic_proto_hybrid_75_25','firp_quota_fraction':self.firp_quota_fraction,'uniform_fraction':1.0-self.firp_quota_fraction,'score_refresh_samples':self.firp_score_refresh_samples,'score_refresh_count':self._class_firp_refresh_count,'uniform_draws':self._class_firp_uniform_draws,'class_firp_draws':self._class_firp_quota_draws,'policy_is_fixed_at_deployment':True,'uses_class_aggregated_score_only':True,'within_class_sampling_is_uniform':True,'uses_online_loss_gradient_task_id_validation_or_controller':False,'replay_batch_size_unchanged':True}; r['taskification_invariant_replay_policy']['enabled']=False; r['taskification_invariant_replay_policy']['class_score_source_only']=True; return r


class ClassFIRPDebtSwapERACE(ClassFIRPERACE):
 """Apply bounded class-risk swaps after the parent's exact uniform draw."""
 def __init__(self,*args,max_swaps_per_batch:int=4,force_neutral_scores:bool=False,**kwargs):
  super().__init__(*args,**kwargs)
  if max_swaps_per_batch<0: raise ValueError('max_swaps_per_batch must be nonnegative')
  self.max_swaps_per_batch=int(max_swaps_per_batch); self.force_neutral_scores=bool(force_neutral_scores)
  self._class_firp_debt={}; self._class_firp_replay_calls=0; self._class_firp_swap_count=0; self._class_firp_batches_with_swaps=0; self._class_firp_max_abs_debt=0.0
 @staticmethod
 def _risk_target_deltas(labels,class_scores,count):
  counts={}
  for label in labels: counts[int(label)]=counts.get(int(label),0)+1
  total=max(1,len(labels)); weighted={label:(amount/total)*max(float(class_scores.get(label,1.0)),1e-12) for label,amount in counts.items()}; normalizer=sum(weighted.values())
  return {label:count*(weighted[label]/max(normalizer,1e-12)-counts[label]/total) for label in counts}
 def _record_debt_swap(self,source_label,target_label,source_debt,target_debt):
  """Optional audit hook called after a swap without changing selection."""
 def _apply_debt_swaps(self,indices):
  swaps=0; labels=self._memory_y
  while swaps<self.max_swaps_per_batch:
   selected=set(indices)
   positive=[label for label,debt in self._class_firp_debt.items() if debt>=.5 and any(index not in selected and labels[index]==label for index in range(len(labels)))]
   negative=[label for label,debt in self._class_firp_debt.items() if debt<=-.5 and any(labels[index]==label for index in indices)]
   if not positive or not negative: break
   target_label=max(positive,key=lambda label:(self._class_firp_debt[label],-label)); source_label=min(negative,key=lambda label:(self._class_firp_debt[label],label))
   target_candidates=[index for index in range(len(labels)) if index not in selected and labels[index]==target_label]
   source_positions=[position for position,index in enumerate(indices) if labels[index]==source_label]
   target_index=self._audit_rng.choice(target_candidates); source_position=self._audit_rng.choice(source_positions)
   target_debt=self._class_firp_debt[target_label]; source_debt=self._class_firp_debt[source_label]
   indices[source_position]=target_index; self._class_firp_debt[target_label]-=1.0; self._class_firp_debt[source_label]+=1.0; swaps+=1
   self._record_debt_swap(source_label,target_label,source_debt,target_debt)
  return swaps
 def _sample_memory(self):
  if not self._memory_x: return None
  count=min(self.batch_size_mem,len(self._memory_x)); indices=self._rng.sample(range(len(self._memory_x)),count)
  x=torch.stack(self._memory_x).to(self.device); y=torch.tensor(self._memory_y,dtype=torch.long,device=self.device); clock=max(self._seen_samples,self._hybrid_seen)
  if not self._class_firp_scores or clock-self._class_firp_last_refresh>=self.firp_score_refresh_samples:
   self._refresh_class_firp_scores(x,y,clock)
  scores=({label:1.0 for label in set(self._memory_y)} if self.force_neutral_scores else self._class_firp_scores)
  for label,delta in self._risk_target_deltas(self._memory_y,scores,count).items():
   self._class_firp_debt[label]=max(-float(count),min(float(count),self._class_firp_debt.get(label,0.0)+delta))
  swaps=self._apply_debt_swaps(indices)
  self._class_firp_replay_calls+=1; self._class_firp_swap_count+=swaps; self._class_firp_batches_with_swaps+=int(swaps>0)
  self._class_firp_max_abs_debt=max(self._class_firp_max_abs_debt,max((abs(value) for value in self._class_firp_debt.values()),default=0.0))
  self._last_replay_memory_indices=list(indices); self._record_memory_trace_replay(indices); self._replay_examples+=count
  chosen=torch.tensor(indices,dtype=torch.long,device=self.device)
  return x[chosen],y[chosen],torch.tensor([self._memory_tid[index] for index in indices],dtype=torch.long,device=self.device)
 def rbcl_summary(self):
  r=super().rbcl_summary(); calls=max(1,self._class_firp_replay_calls)
  r['class_firp_replay']['enabled']=not self.force_neutral_scores and self.max_swaps_per_batch>0
  r['class_firp_replay']['debt_swap']={'enabled':not self.force_neutral_scores and self.max_swaps_per_batch>0,'force_neutral_scores':self.force_neutral_scores,'parent_uniform_draw_is_exact':True,'main_sampler_rng_receives_no_extra_draws':True,'independent_within_class_rng':True,'max_swaps_per_batch':self.max_swaps_per_batch,'replay_calls':self._class_firp_replay_calls,'swap_count':self._class_firp_swap_count,'mean_swaps_per_batch':self._class_firp_swap_count/calls,'batches_with_swaps':self._class_firp_batches_with_swaps,'max_abs_debt':self._class_firp_max_abs_debt,'neutral_score_is_exact_noop':True}
  return r


class ClassFIRPExposureAuditERACE(ClassFIRPDebtSwapERACE):
 """Audit whether Debt-Swap protects classes already exposed by the stream."""
 def __init__(self,*args,exposure_window_samples:int=512,**kwargs):
  super().__init__(*args,**kwargs)
  if exposure_window_samples<=0: raise ValueError('exposure_window_samples must be positive')
  self.exposure_window_samples=int(exposure_window_samples); self._exposure_recent_labels=deque(); self._exposure_recent_counts=Counter(); self._exposure_current_counts=Counter(); self._exposure_memory_counts=Counter()
  self._exposure_global=self._empty_exposure_stats(); self._exposure_exp=self._empty_exposure_stats(); self._exposure_history=[]
 @staticmethod
 def _empty_exposure_stats():
  return {'swap_count':0,'target_current_overlap_count':0,'source_current_overlap_count':0,'target_recent_present_count':0,'source_recent_present_count':0,'target_recent_share_sum':0.0,'source_recent_share_sum':0.0,'target_memory_share_sum':0.0,'source_memory_share_sum':0.0,'target_risk_score_sum':0.0,'source_risk_score_sum':0.0,'target_debt_sum':0.0,'source_debt_abs_sum':0.0,'by_target_class':{}}
 @staticmethod
 def _snapshot_exposure_stats(stats):
  count=max(1,stats['swap_count']); classes={}
  for label,row in sorted(stats['by_target_class'].items()):
   swaps=max(1,row['target_swaps']); classes[str(label)]={'target_swaps':row['target_swaps'],'current_overlap_fraction':row['current_overlap_count']/swaps,'recent_present_fraction':row['recent_present_count']/swaps,'mean_recent_share':row['recent_share_sum']/swaps,'mean_memory_share':row['memory_share_sum']/swaps,'mean_risk_score':row['risk_score_sum']/swaps}
  return {'swap_count':stats['swap_count'],'target_current_overlap_fraction':stats['target_current_overlap_count']/count,'source_current_overlap_fraction':stats['source_current_overlap_count']/count,'target_recent_present_fraction':stats['target_recent_present_count']/count,'source_recent_present_fraction':stats['source_recent_present_count']/count,'mean_target_recent_share':stats['target_recent_share_sum']/count,'mean_source_recent_share':stats['source_recent_share_sum']/count,'mean_target_memory_share':stats['target_memory_share_sum']/count,'mean_source_memory_share':stats['source_memory_share_sum']/count,'mean_target_risk_score':stats['target_risk_score_sum']/count,'mean_source_risk_score':stats['source_risk_score_sum']/count,'mean_target_debt_before_swap':stats['target_debt_sum']/count,'mean_source_abs_debt_before_swap':stats['source_debt_abs_sum']/count,'by_target_class':classes}
 def _before_training_exp(self,**kwargs):
  self._exposure_exp=self._empty_exposure_stats(); super()._before_training_exp(**kwargs)
 def _after_training_exp(self,**kwargs):
  self._exposure_history.append(self._snapshot_exposure_stats(self._exposure_exp)); super()._after_training_exp(**kwargs)
 def _before_replay_selection(self,current_x,current_y,current_tid):
  labels=[int(label) for label in current_y.detach().cpu().tolist()]; self._exposure_current_counts=Counter(labels)
  if self._epoch_index==0:
   for label in labels:
    self._exposure_recent_labels.append(label); self._exposure_recent_counts[label]+=1
    if len(self._exposure_recent_labels)>self.exposure_window_samples:
     old=self._exposure_recent_labels.popleft(); self._exposure_recent_counts[old]-=1
     if self._exposure_recent_counts[old]<=0: del self._exposure_recent_counts[old]
  super()._before_replay_selection(current_x,current_y,current_tid)
 def _sample_memory(self):
  self._exposure_memory_counts=Counter(int(label) for label in self._memory_y)
  return super()._sample_memory()
 def _record_debt_swap(self,source_label,target_label,source_debt,target_debt):
  recent_total=max(1,len(self._exposure_recent_labels)); memory_total=max(1,len(self._memory_y)); target_score=float(self._class_firp_scores.get(target_label,1.0)); source_score=float(self._class_firp_scores.get(source_label,1.0))
  target_recent=self._exposure_recent_counts.get(target_label,0)/recent_total; source_recent=self._exposure_recent_counts.get(source_label,0)/recent_total; target_memory=self._exposure_memory_counts.get(target_label,0)/memory_total; source_memory=self._exposure_memory_counts.get(source_label,0)/memory_total
  for stats in (self._exposure_exp,self._exposure_global):
   stats['swap_count']+=1; stats['target_current_overlap_count']+=int(target_label in self._exposure_current_counts); stats['source_current_overlap_count']+=int(source_label in self._exposure_current_counts); stats['target_recent_present_count']+=int(target_label in self._exposure_recent_counts); stats['source_recent_present_count']+=int(source_label in self._exposure_recent_counts); stats['target_recent_share_sum']+=target_recent; stats['source_recent_share_sum']+=source_recent; stats['target_memory_share_sum']+=target_memory; stats['source_memory_share_sum']+=source_memory; stats['target_risk_score_sum']+=target_score; stats['source_risk_score_sum']+=source_score; stats['target_debt_sum']+=float(target_debt); stats['source_debt_abs_sum']+=abs(float(source_debt))
   row=stats['by_target_class'].setdefault(int(target_label),{'target_swaps':0,'current_overlap_count':0,'recent_present_count':0,'recent_share_sum':0.0,'memory_share_sum':0.0,'risk_score_sum':0.0}); row['target_swaps']+=1; row['current_overlap_count']+=int(target_label in self._exposure_current_counts); row['recent_present_count']+=int(target_label in self._exposure_recent_counts); row['recent_share_sum']+=target_recent; row['memory_share_sum']+=target_memory; row['risk_score_sum']+=target_score
 def rbcl_summary(self):
  r=super().rbcl_summary(); r['class_firp_exposure_redundancy_audit']={'enabled':True,'audit_only':True,'sampling_modified_by_audit':False,'training_objective_modified':False,'uses_current_labels_and_causal_recent_exposure_only':True,'uses_loss_gradient_task_id_validation_or_future_data':False,'recent_window_samples':self.exposure_window_samples,'history':self._exposure_history,'global':self._snapshot_exposure_stats(self._exposure_global)}; return r


class TIRPSemanticBoundaryERACE(TIRPSemanticRepresentativeERACE):
 """D37: retain TIRP replay while reserving a fixed semantic boundary quota.

 The rule only uses the frozen semantic embedding of the arrived inputs and
 already stored labels.  It is not a loss/gradient/controller signal and has
 no access to a task identifier or boundary annotation.
 """
 def __init__(self,*args,boundary_fraction:float=.25,boundary_audit_only:bool=False,boundary_audit_every:int=512,**kwargs):
  super().__init__(*args,**kwargs)
  if not 0.0<boundary_fraction<1.0: raise ValueError('boundary_fraction must be in (0,1)')
  self.boundary_fraction=float(boundary_fraction); self.boundary_audit_only=bool(boundary_audit_only); self.boundary_audit_every=int(boundary_audit_every)
  self._boundary_context=None; self._boundary_current_labels=set(); self._boundary_current_count=0; self._boundary_audit_history=[]
 def _before_replay_selection(self,current_x,current_y,current_tid):
  with torch.no_grad(): self._boundary_context=F.normalize(self._features(current_x).float().mean(dim=0),dim=0).to(self.device)
  self._boundary_current_labels={int(label) for label in current_y.detach().cpu().tolist()}; self._boundary_current_count=int(current_y.numel())
 def _candidate_indices(self,weights,raw,y,count,quality=None):
  boundary_count=min(max(1,int(round(count*self.boundary_fraction))),max(0,count-1)); retention_count=count-boundary_count
  retain=self._tirp_sample(weights,y,retention_count,quality) if retention_count else torch.empty(0,dtype=torch.long,device=self.device)
  selected=set(retain.tolist()); candidates=[index for index,label in enumerate(y.tolist()) if index not in selected and int(label) not in self._boundary_current_labels]
  if len(candidates)<boundary_count: candidates=[index for index in range(len(y)) if index not in selected]
  similarity=F.normalize(raw,dim=1) @ self._boundary_context
  ordered=sorted(candidates,key=lambda index:(float(similarity[index]),-index),reverse=True)[:boundary_count]
  return torch.tensor(retain.tolist()+ordered,dtype=torch.long,device=self.device),similarity
 def _sample_memory(self):
  if not self._memory_x: return None
  count=min(self.batch_size_mem,len(self._memory_x)); x=torch.stack(self._memory_x).to(self.device); y=torch.tensor(self._memory_y,dtype=torch.long,device=self.device)
  with torch.no_grad():
   raw,inp=self._tirp_inputs(x,y)
   weights=self._tirp_weights(inp); baseline=self._tirp_sample(weights,y,count,inp[:,-2])
   candidate,similarity=self._candidate_indices(weights,raw,y,count,inp[:,-2])
   next_clock=self._seen_samples+max(1,self._boundary_current_count)
   if self._epoch_index==0 and self.boundary_audit_every>0 and next_clock%self.boundary_audit_every==0:
    def entropy(indices):
     labels=y[indices]; p=torch.bincount(labels,minlength=max(100,int(y.max())+1)).float(); p=p[p>0]/len(labels); return float(-(p*p.log()).sum()/torch.log(torch.tensor(float(len(labels)),device=self.device)))
    self._boundary_audit_history.append({'sample_clock':float(next_clock),'baseline_semantic_similarity':float(similarity[baseline].mean()),'candidate_semantic_similarity':float(similarity[candidate].mean()),'baseline_entropy':entropy(baseline),'candidate_entropy':entropy(candidate),'candidate_old_class_fraction':float(sum(int(y[index]) not in self._boundary_current_labels for index in candidate.tolist())/len(candidate))})
   indices=baseline if self.boundary_audit_only else candidate
   self._last_replay_memory_indices=indices.detach().cpu().tolist()
  self._replay_examples+=count
  return x[indices],y[indices],torch.tensor([self._memory_tid[int(i)] for i in indices.tolist()],dtype=torch.long,device=self.device)
 def rbcl_summary(self):
  r=super().rbcl_summary(); h=self._boundary_audit_history
  r['semantic_boundary_retrieval']={'enabled':not self.boundary_audit_only,'audit_only':self.boundary_audit_only,'boundary_fraction':self.boundary_fraction,'uses_frozen_current_input_semantics':True,'uses_task_id_loss_gradient_validation_or_boundary_signal':False,'audit_count':len(h),'history':h,'mean_similarity_improvement':(sum(v['candidate_semantic_similarity']-v['baseline_semantic_similarity'] for v in h)/len(h) if h else 0.0),'mean_entropy_ratio':(sum(v['candidate_entropy']/max(v['baseline_entropy'],1e-12) for v in h)/len(h) if h else 0.0)}; return r


class TIRPProxyContrastiveERACE(TIRPSemanticRepresentativeERACE):
 """Fixed task-free proxy contrastive calibration over current and TIRP replay."""
 def __init__(self,*args,proxy_lambda:float=.05,proxy_temperature:float=1.0,**kwargs):
  super().__init__(*args,**kwargs)
  if proxy_lambda < 0 or proxy_temperature <= 0: raise ValueError('invalid proxy calibration hyperparameter')
  self.proxy_lambda=float(proxy_lambda); self.proxy_temperature=float(proxy_temperature)
  self._proxy_audit_history=[]
 def _trainable_features(self,x):
  m=self.model; out=F.relu(m.bn1(m.conv1(x))); out=m.layer1(out); out=m.layer2(out); out=m.layer3(out); out=m.layer4(out); return F.adaptive_avg_pool2d(out,1).flatten(1)
 def _add_auxiliary_loss(self,current_x,current_y,current_tid,replay,replay_output=None):
  if replay is None or len(self._observed_classes)<2: return
  x=torch.cat((current_x.to(self.device),replay[0]),dim=0); labels=torch.cat((current_y.to(self.device),replay[1]),dim=0)
  observed=torch.tensor(sorted(self._observed_classes),dtype=torch.long,device=self.device)
  features=F.normalize(self._trainable_features(x),dim=1); proxies=F.normalize(self.model.linear.weight[observed],dim=1)
  positions=torch.searchsorted(observed,labels)
  proxy_loss=F.cross_entropy((features @ proxies.T)/self.proxy_temperature,positions)
  clock=self._seen_samples+int(current_y.numel())
  if self._epoch_index==0 and clock%512==0:
   self._proxy_audit_history.append({'sample_clock':float(clock),'proxy_loss':float(proxy_loss.detach()),'observed_proxy_count':int(observed.numel())})
  self.loss+=self.proxy_lambda*proxy_loss
 def rbcl_summary(self):
  r=super().rbcl_summary(); h=self._proxy_audit_history
  r['fixed_proxy_contrastive_calibration']={'enabled':True,'lambda':self.proxy_lambda,'temperature':self.proxy_temperature,'uses_current_and_replay_labels_only':True,'uses_historical_logits_or_margins':False,'uses_task_id_loss_gradient_validation_or_controller':False,'audit_count':len(h),'history':h}; return r


class TIRPSemanticRelationERACE(TIRPSemanticRepresentativeERACE):
 """Fixed current-to-replay semantic-relation calibration.

 The frozen ImageNet encoder is already part of the representative-memory
 layer.  This optional third layer therefore does not introduce a second
 teacher, save historical model outputs, or alter memory/replay selection.
 It only asks the trainable representation to preserve the *cross* relation
 between the arrived minibatch and the TIRP-replayed minibatch.  The relation
 target is computed from the frozen semantic encoder on samples that have
 already arrived.  In audit mode it records the target's availability and
 geometry but does not change the training objective.
 """
 def __init__(self,*args,relation_lambda:float=.025,relation_audit_only:bool=False,relation_audit_every:int=512,relation_mechanism_audit:bool=False,relation_replay_audit:bool=False,relation_mode:str='raw',**kwargs):
  super().__init__(*args,**kwargs)
  if relation_lambda < 0: raise ValueError('relation_lambda must be non-negative')
  if relation_audit_every <= 0: raise ValueError('relation_audit_every must be positive')
  if relation_mode not in {'raw','centered','standardized'}: raise ValueError('relation_mode must be raw, centered, or standardized')
  self.semantic_relation_lambda=float(relation_lambda); self.semantic_relation_audit_only=bool(relation_audit_only); self.semantic_relation_audit_every=int(relation_audit_every); self.semantic_relation_replay_audit=bool(relation_replay_audit); self.semantic_relation_mechanism_audit=bool(relation_mechanism_audit or relation_replay_audit); self.semantic_relation_mode=str(relation_mode); self._semantic_relation_audit_history=[]
 def _trainable_features(self,x):
  m=self.model; out=F.relu(m.bn1(m.conv1(x))); out=m.layer1(out); out=m.layer2(out); out=m.layer3(out); out=m.layer4(out); return F.adaptive_avg_pool2d(out,1).flatten(1)
 @staticmethod
 def _centered_correlation(a,b):
  a=a.flatten(); b=b.flatten(); a=a-a.mean(); b=b-b.mean(); denom=torch.linalg.vector_norm(a)*torch.linalg.vector_norm(b)
  return 0.0 if float(denom.detach()) <= 1e-12 else float((a @ b / denom).detach())
 @staticmethod
 def _standardize_relation(matrix):
  centered=matrix-matrix.mean(); return centered/centered.std(unbiased=False).clamp_min(1e-6)
 @staticmethod
 def _center_relation(matrix): return matrix-matrix.mean()
 @staticmethod
 def _replay_group_statistics(labels,margins,losses,ages,risks):
  def summarize(mask):
   count=int(mask.sum())
   if count==0: return {'count':0,'mean_ce':None,'mean_margin':None,'accuracy':None,'negative_margin_fraction':None,'mean_age':None,'mean_semantic_risk':None}
   return {'count':count,'mean_ce':float(losses[mask].mean()),'mean_margin':float(margins[mask].mean()),'accuracy':float((margins[mask]>0).float().mean()),'negative_margin_fraction':float((margins[mask]<=0).float().mean()),'mean_age':float(ages[mask].mean()),'mean_semantic_risk':float(risks[mask].mean())}
  by_class={str(int(label)):summarize(labels.eq(label)) for label in labels.unique(sorted=True)}
  age_groups={'young':summarize(ages<1/3),'middle':summarize((ages>=1/3)&(ages<2/3)),'old':summarize(ages>=2/3)}
  return {'overall':summarize(torch.ones_like(labels,dtype=torch.bool)),'by_class':by_class,'by_age':age_groups}
 def _add_auxiliary_loss(self,current_x,current_y,current_tid,replay,replay_output=None):
  if replay is None or not len(self._observed_classes): return
  base_loss=self.loss
  current_features=F.normalize(self._trainable_features(current_x.to(self.device)),dim=1)
  replay_features=F.normalize(self._trainable_features(replay[0]),dim=1)
  learned_relation=current_features @ replay_features.T
  with torch.no_grad():
   semantic_current=F.normalize(self._features(current_x).float().to(self.device),dim=1)
   semantic_replay=F.normalize(self._features(replay[0]).float().to(self.device),dim=1)
   semantic_relation=semantic_current @ semantic_replay.T
  learned_for_loss=learned_relation; semantic_for_loss=semantic_relation
  if self.semantic_relation_mode=='centered':
   learned_for_loss=self._center_relation(learned_relation); semantic_for_loss=self._center_relation(semantic_relation)
  elif self.semantic_relation_mode=='standardized':
   learned_for_loss=self._standardize_relation(learned_relation); semantic_for_loss=self._standardize_relation(semantic_relation)
  pair_loss=F.smooth_l1_loss(learned_for_loss,semantic_for_loss,reduction='none'); relation_loss=pair_loss.mean()
  clock=self._seen_samples+int(current_y.numel())
  if self._epoch_index==0 and clock%self.semantic_relation_audit_every==0:
   labels_current=current_y.to(self.device)[:,None]; labels_replay=replay[1][None,:]
   same=labels_current.eq(labels_replay)
   record={'sample_clock':float(clock),'relation_mode':self.semantic_relation_mode,'target_mean':float(semantic_relation.mean().detach()),'target_std':float(semantic_relation.std().detach()),'learned_mean':float(learned_relation.mean().detach()),'learned_std':float(learned_relation.std().detach()),'learned_target_correlation':self._centered_correlation(learned_relation.detach(),semantic_relation),'semantic_same_minus_different':float((semantic_relation[same].mean()-semantic_relation[~same].mean()).detach()) if bool(same.any()) and bool((~same).any()) else None,'relation_loss':float(relation_loss.detach()),'pair_count':int(semantic_relation.numel())}
   if self.semantic_relation_mechanism_audit:
    weighted=self.semantic_relation_lambda*relation_loss; base_value=abs(float(base_loss.detach())); pair_detached=pair_loss.detach(); total_pair_loss=float(pair_detached.sum())
    class_pairs=torch.stack((labels_current.expand_as(pair_loss),labels_replay.expand_as(pair_loss)),dim=-1).reshape(-1,2); _,inverse=torch.unique(class_pairs,dim=0,return_inverse=True); pair_sums=torch.zeros(int(inverse.max())+1,device=self.device).scatter_add_(0,inverse,pair_detached.reshape(-1)); shares=pair_sums/pair_sums.sum().clamp_min(1e-12)
    params=[parameter for name,parameter in self.model.named_parameters() if parameter.requires_grad and not name.startswith('linear.')]
    base_grad=torch.autograd.grad(base_loss,params,retain_graph=True,allow_unused=True); relation_grad=torch.autograd.grad(relation_loss,params,retain_graph=True,allow_unused=True)
    dot=torch.zeros((),device=self.device); base_norm_sq=torch.zeros((),device=self.device); relation_norm_sq=torch.zeros((),device=self.device)
    for first,second in zip(base_grad,relation_grad):
     if first is not None: base_norm_sq+=first.detach().square().sum()
     if second is not None: relation_norm_sq+=second.detach().square().sum()
     if first is not None and second is not None: dot+=(first.detach()*second.detach()).sum()
    base_norm=base_norm_sq.sqrt(); relation_norm=relation_norm_sq.sqrt(); denom=base_norm*relation_norm
    indices=getattr(self,'_last_replay_memory_indices',[]); memory_clock=max(1,self._seen_samples,self._hybrid_seen); ages=[]
    for index in indices:
     item=self._memory_x[index]; label=self._memory_y[index]; ages.append((memory_clock-self._memory_arrival.get(self._memory_key(item,label),memory_clock))/memory_clock)
    record.update({'base_erace_loss':float(base_loss.detach()),'weighted_relation_loss':float(weighted.detach()),'weighted_relation_to_base_loss_ratio':float(weighted.detach())/max(base_value,1e-12),'backbone_gradient_cosine':float((dot/denom).detach()) if float(denom.detach())>1e-12 else None,'base_backbone_gradient_norm':float(base_norm.detach()),'relation_backbone_gradient_norm':float(relation_norm.detach()),'weighted_relation_to_base_gradient_norm_ratio':self.semantic_relation_lambda*float(relation_norm.detach())/max(float(base_norm.detach()),1e-12),'current_class_count':int(current_y.unique().numel()),'replay_class_count':int(replay[1].unique().numel()),'same_class_pair_ratio':float(same.float().mean()),'same_pair_relation_loss':float(pair_detached[same].mean()) if bool(same.any()) else None,'different_pair_relation_loss':float(pair_detached[~same].mean()) if bool((~same).any()) else None,'class_pair_max_loss_fraction':float(shares.max()) if total_pair_loss>0 else 0.0,'class_pair_loss_hhi':float(shares.square().sum()) if total_pair_loss>0 else 0.0,'replay_age_mean':sum(ages)/len(ages) if ages else None,'replay_age_min':min(ages) if ages else None,'replay_age_max':max(ages) if ages else None,'classifier_head_relation_gradient_is_structurally_zero':True})
    if self.semantic_relation_replay_audit:
     current_loss=getattr(self,'_auxiliary_current_loss',None); replay_loss=getattr(self,'_auxiliary_replay_loss',None)
     if current_loss is None or replay_loss is None or replay_output is None: raise RuntimeError('replay audit requires current and replay losses')
     current_grad=torch.autograd.grad(current_loss,params,retain_graph=True,allow_unused=True); replay_grad=torch.autograd.grad(replay_loss,params,retain_graph=True,allow_unused=True)
     def gradient_geometry(first,second):
      product=torch.zeros((),device=self.device); first_sq=torch.zeros((),device=self.device); second_sq=torch.zeros((),device=self.device)
      for left,right in zip(first,second):
       if left is not None: first_sq+=left.detach().square().sum()
       if right is not None: second_sq+=right.detach().square().sum()
       if left is not None and right is not None: product+=(left.detach()*right.detach()).sum()
      first_norm=first_sq.sqrt(); second_norm=second_sq.sqrt(); divisor=first_norm*second_norm
      return {'cosine':float((product/divisor).detach()) if float(divisor.detach())>1e-12 else None,'first_norm':float(first_norm.detach()),'second_norm':float(second_norm.detach())}
     current_relation=gradient_geometry(current_grad,relation_grad); replay_relation=gradient_geometry(replay_grad,relation_grad); current_replay=gradient_geometry(current_grad,replay_grad)
     different=~same; high_different=torch.zeros_like(different); high_threshold=None; high_loss=None; high_geometry=None
     if bool(different.any()):
      high_threshold=torch.quantile(semantic_relation[different].detach(),.75); high_different=different & semantic_relation.ge(high_threshold)
      if bool(high_different.any()):
       high_loss=pair_loss[high_different].mean(); high_grad=torch.autograd.grad(high_loss,params,retain_graph=True,allow_unused=True); high_geometry=gradient_geometry(replay_grad,high_grad)
     replay_losses=F.cross_entropy(replay_output,replay[1],reduction='none'); chosen=torch.arange(replay_output.shape[0],device=self.device); correct=replay_output[chosen,replay[1]]; rivals=replay_output.detach().clone(); rivals[chosen,replay[1]]=-torch.inf; margins=(correct-rivals.max(dim=1).values).detach()
     semantic_risks=[]
     for column,label in enumerate(replay[1]):
      valid=current_y.to(self.device).ne(label)
      semantic_risks.append(semantic_relation[valid,column].max() if bool(valid.any()) else semantic_relation[:,column].max())
     age_tensor=torch.tensor(ages,dtype=torch.float32,device=self.device) if len(ages)==len(replay[1]) else torch.zeros_like(replay[1],dtype=torch.float32)
     groups=self._replay_group_statistics(replay[1].detach(),margins,replay_losses.detach(),age_tensor,torch.stack(semantic_risks).detach())
     different_loss_total=float(pair_detached[different].sum()) if bool(different.any()) else 0.0; all_loss_total=float(pair_detached.sum())
     record.update({'relation_replay_audit':True,'current_classification_loss':float(current_loss.detach()),'replay_classification_loss':float(replay_loss.detach()),'current_relation_gradient_cosine':current_relation['cosine'],'replay_relation_gradient_cosine':replay_relation['cosine'],'current_replay_gradient_cosine':current_replay['cosine'],'current_backbone_gradient_norm':current_relation['first_norm'],'replay_backbone_gradient_norm':replay_relation['first_norm'],'high_similarity_different_threshold':float(high_threshold) if high_threshold is not None else None,'high_similarity_different_pair_fraction':float(high_different.float().mean()),'high_similarity_different_loss_fraction_of_different':float(pair_detached[high_different].sum())/max(different_loss_total,1e-12) if bool(high_different.any()) else None,'high_similarity_different_loss_fraction_of_total':float(pair_detached[high_different].sum())/max(all_loss_total,1e-12) if bool(high_different.any()) else None,'high_similarity_different_relation_loss':float(high_loss.detach()) if high_loss is not None else None,'high_similarity_different_replay_gradient_cosine':high_geometry['cosine'] if high_geometry is not None else None,'replay_group_statistics':groups})
   self._semantic_relation_audit_history.append(record)
  if not self.semantic_relation_audit_only: self.loss+=self.semantic_relation_lambda*relation_loss
 def rbcl_summary(self):
  r=super().rbcl_summary(); h=self._semantic_relation_audit_history
  valid=[v for v in h if v['target_std']>1e-6 and v['pair_count']>0]
  r['fixed_cross_semantic_relation_calibration']={'enabled':not self.semantic_relation_audit_only,'audit_only':self.semantic_relation_audit_only,'mechanism_audit':self.semantic_relation_mechanism_audit,'replay_audit':self.semantic_relation_replay_audit,'mode':self.semantic_relation_mode,'lambda':self.semantic_relation_lambda,'uses_frozen_semantic_encoder_already_owned_by_memory_layer':True,'uses_only_arrived_current_and_replay_samples':True,'uses_historical_logits_or_margins':False,'uses_task_id_loss_gradient_validation_or_controller':False,'audit_count':len(h),'valid_target_fraction':float(len(valid)/len(h)) if h else 0.0,'mean_learned_target_correlation':sum(v['learned_target_correlation'] for v in h)/len(h) if h else 0.0,'history':h}; return r


class TIRPSemanticRelationGuardERACE(TIRPSemanticRepresentativeERACE):
 """Bounded lazy replay protection driven by causal semantic relations."""
 def __init__(self,*args,guard_max_relative_deviation:float=.25,guard_audit_every:int=512,guard_semantic_confidence:float|None=None,guard_min_vulnerable_fraction:float=0.0,guard_probability_margin_threshold:float|None=None,**kwargs):
  super().__init__(*args,**kwargs)
  if not 0<guard_max_relative_deviation<1: raise ValueError('guard_max_relative_deviation must be in (0,1)')
  if guard_audit_every<=0: raise ValueError('guard_audit_every must be positive')
  if guard_semantic_confidence is not None and not .5<guard_semantic_confidence<1: raise ValueError('guard_semantic_confidence must be in (0.5,1) or None')
  if not 0<=guard_min_vulnerable_fraction<=1: raise ValueError('guard_min_vulnerable_fraction must be in [0,1]')
  if guard_probability_margin_threshold is not None and not 0<=guard_probability_margin_threshold<=1: raise ValueError('guard_probability_margin_threshold must be in [0,1] or None')
  self.relation_guard_max_relative_deviation=float(guard_max_relative_deviation); self.relation_guard_audit_every=int(guard_audit_every); self.relation_guard_semantic_confidence=None if guard_semantic_confidence is None else float(guard_semantic_confidence); self.relation_guard_min_vulnerable_fraction=float(guard_min_vulnerable_fraction); self.relation_guard_probability_margin_threshold=None if guard_probability_margin_threshold is None else float(guard_probability_margin_threshold); self._relation_guard_history=[]
 @staticmethod
 def _rank01(values):
  if values.numel()<=1: return torch.zeros_like(values,dtype=torch.float32)
  order=torch.argsort(values,stable=True); ranks=torch.empty_like(values,dtype=torch.float32); ranks[order]=torch.arange(values.numel(),device=values.device,dtype=torch.float32); return ranks/(values.numel()-1)
 @staticmethod
 def _guard_weights(semantic_risk,vulnerability,max_relative_deviation=.25,semantic_confidence=None,min_vulnerable_fraction=0.0,vulnerable_mask=None):
  count=int(semantic_risk.numel()); uniform=torch.full_like(semantic_risk,1/max(1,count),dtype=torch.float32)
  if vulnerable_mask is not None and int(vulnerable_mask.numel())!=count: raise ValueError('vulnerable_mask must match semantic_risk')
  vulnerable_indicator=vulnerability.float().ge(0) if vulnerable_mask is None else vulnerable_mask.bool()
  noise_floor=1/math.sqrt(max(1,count)); vulnerable_fraction=float(vulnerable_indicator.float().mean()) if count else 0.0
  if count<4:
   diagnostics={'semantic_vulnerability_correlation':0.0,'activation_noise_floor':noise_floor,'semantic_reliability_lower_bound':-1.0,'semantic_reliability_confidence':semantic_confidence,'vulnerable_replay_fraction':vulnerable_fraction,'minimum_vulnerable_replay_fraction':float(min_vulnerable_fraction),'vulnerability_uses_explicit_mask':vulnerable_mask is not None,'semantic_reliability_passes':False,'vulnerable_replay_fraction_passes':vulnerable_fraction>=float(min_vulnerable_fraction),'active':False}
   return uniform,diagnostics
  semantic=semantic_risk.float(); vulnerable=vulnerability.float(); first=semantic-semantic.mean(); second=vulnerable-vulnerable.mean(); denom=torch.linalg.vector_norm(first)*torch.linalg.vector_norm(second)
  correlation=0.0 if float(denom)<=1e-12 else float((first @ second/denom))
  lower_bound=correlation
  if semantic_confidence is not None:
   clipped=max(-.999999,min(.999999,correlation)); critical=NormalDist().inv_cdf(float(semantic_confidence)); lower_bound=math.tanh(math.atanh(clipped)-critical/math.sqrt(count-3))
  semantic_reliable=correlation>noise_floor and lower_bound>0 and float(semantic.std(unbiased=False))>1e-6
  vulnerability_sufficient=vulnerable_fraction>=float(min_vulnerable_fraction)
  diagnostics={'semantic_vulnerability_correlation':correlation,'activation_noise_floor':noise_floor,'semantic_reliability_lower_bound':lower_bound,'semantic_reliability_confidence':semantic_confidence,'vulnerable_replay_fraction':vulnerable_fraction,'minimum_vulnerable_replay_fraction':float(min_vulnerable_fraction),'vulnerability_uses_explicit_mask':vulnerable_mask is not None,'semantic_reliability_passes':bool(semantic_reliable),'vulnerable_replay_fraction_passes':bool(vulnerability_sufficient),'active':bool(semantic_reliable and vulnerability_sufficient)}
  if not diagnostics['active']: return uniform,diagnostics
  semantic_permission=TIRPSemanticRelationGuardERACE._rank01(semantic_risk); classification_vulnerability=TIRPSemanticRelationGuardERACE._rank01(vulnerability); protection=semantic_permission*classification_vulnerability; centered=protection-protection.mean(); scale=centered.abs().max().clamp_min(1e-12); relative=1+float(max_relative_deviation)*centered/scale; weights=relative/relative.sum()
  return weights,diagnostics
 def _add_auxiliary_loss(self,current_x,current_y,current_tid,replay,replay_output=None):
  if replay is None or replay_output is None: return
  with torch.no_grad():
   semantic_current=F.normalize(self._features(current_x).float().to(self.device),dim=1); semantic_replay=F.normalize(self._features(replay[0]).float().to(self.device),dim=1); relation=semantic_current @ semantic_replay.T
   different=current_y.to(self.device)[:,None].ne(replay[1][None,:]); risks=[]
   for column in range(replay[1].numel()):
    valid=different[:,column]; risks.append(relation[valid,column].max() if bool(valid.any()) else relation[:,column].max())
   semantic_risk=torch.stack(risks); chosen=torch.arange(replay_output.shape[0],device=self.device); correct=replay_output.detach()[chosen,replay[1]]; rivals=replay_output.detach().clone(); rivals[chosen,replay[1]]=-torch.inf; margins=correct-rivals.max(dim=1).values; vulnerability=-margins; probability_margins=None; vulnerable_mask=None
   if self.relation_guard_probability_margin_threshold is not None:
    probabilities=replay_output.detach().softmax(dim=1); correct_probability=probabilities[chosen,replay[1]]; rival_probabilities=probabilities.clone(); rival_probabilities[chosen,replay[1]]=-torch.inf; probability_margins=correct_probability-rival_probabilities.max(dim=1).values; vulnerable_mask=probability_margins.le(self.relation_guard_probability_margin_threshold)
   weights,gate=self._guard_weights(semantic_risk,vulnerability,self.relation_guard_max_relative_deviation,self.relation_guard_semantic_confidence,self.relation_guard_min_vulnerable_fraction,vulnerable_mask); active=gate['active']
  replay_losses=F.cross_entropy(replay_output,replay[1],reduction='none'); uniform_loss=replay_losses.mean(); guarded_loss=(weights.detach()*replay_losses).sum(); self.loss+=.5*(guarded_loss-uniform_loss)
  clock=self._seen_samples+int(current_y.numel())
  if self._epoch_index==0 and clock%self.relation_guard_audit_every==0:
   count=len(weights); uniform=torch.full_like(weights,1/count); class_count=max(100,int(replay[1].max())+1); guarded_mass=torch.bincount(replay[1],weights=weights,minlength=class_count); uniform_mass=torch.bincount(replay[1],weights=uniform,minlength=class_count); indices=getattr(self,'_last_replay_memory_indices',[]); memory_clock=max(1,self._seen_samples,self._hybrid_seen); ages=[]
   for index in indices:
    item=self._memory_x[index]; label=self._memory_y[index]; ages.append((memory_clock-self._memory_arrival.get(self._memory_key(item,label),memory_clock))/memory_clock)
   age_tensor=torch.tensor(ages,dtype=torch.float32,device=self.device) if len(ages)==count else torch.zeros(count,device=self.device); normalized_entropy=float(-(weights*weights.clamp_min(1e-12).log()).sum()/math.log(max(2,count))); ess_ratio=float(1/(weights.square().sum()*count))
   self._relation_guard_history.append({'sample_clock':float(clock),**gate,'semantic_risk_mean':float(semantic_risk.mean()),'semantic_risk_std':float(semantic_risk.std(unbiased=False)),'replay_margin_mean':float(margins.mean()),'replay_negative_margin_fraction':float((margins<=0).float().mean()),'replay_probability_margin_mean':float(probability_margins.mean()) if probability_margins is not None else None,'probability_margin_threshold':self.relation_guard_probability_margin_threshold,'uniform_replay_loss':float(uniform_loss.detach()),'guarded_replay_loss':float(guarded_loss.detach()),'guard_correction':float((.5*(guarded_loss-uniform_loss)).detach()),'minimum_weight_to_uniform_ratio':float((weights/uniform).min()),'maximum_weight_to_uniform_ratio':float((weights/uniform).max()),'normalized_weight_entropy':normalized_entropy,'effective_sample_size_ratio':ess_ratio,'class_mass_tv_from_uniform':float(.5*(guarded_mass-uniform_mass).abs().sum()),'uniform_replay_age_mean':float(age_tensor.mean()),'guarded_replay_age_mean':float((weights*age_tensor).sum())})
 def rbcl_summary(self):
  r=super().rbcl_summary(); h=self._relation_guard_history; active=[row for row in h if row['active']]
  version='probability_margin_v2_1' if self.relation_guard_probability_margin_threshold is not None else ('reliability_and_vulnerability_v2' if self.relation_guard_semantic_confidence is not None or self.relation_guard_min_vulnerable_fraction>0 else 'correlation_v1')
  r['bounded_lazy_semantic_relation_guard']={'enabled':True,'gate_version':version,'max_relative_weight_deviation':self.relation_guard_max_relative_deviation,'semantic_reliability_confidence':self.relation_guard_semantic_confidence,'minimum_vulnerable_replay_fraction':self.relation_guard_min_vulnerable_fraction,'probability_margin_threshold':self.relation_guard_probability_margin_threshold,'activation_uses_semantic_risk_margin_correlation':True,'activation_noise_floor_is_inverse_sqrt_batch':True,'activation_requires_positive_confidence_lower_bound':self.relation_guard_semantic_confidence is not None,'activation_requires_minimum_vulnerable_fraction':self.relation_guard_min_vulnerable_fraction>0,'vulnerability_uses_probability_margin':self.relation_guard_probability_margin_threshold is not None,'weights_are_detached':True,'replaces_replay_loss_without_changing_its_total_coefficient':True,'exact_uniform_fallback_when_inactive':True,'uses_task_id_boundary_validation_or_future_data':False,'audit_count':len(h),'active_count':len(active),'active_fraction':len(active)/len(h) if h else 0.0,'history':h}; return r


class TIRPPrototypeBoundaryRelationERACE(TIRPSemanticRepresentativeERACE):
 """Relation V3: causal EMA prototype-boundary preservation.

 The target is a replay sample's own-class versus strongest-rival prototype
 margin in a causal EMA teacher. It never regresses the ImageNet
 current-to-replay cosine matrix used by Relation V1.
 """
 def __init__(self,*args,relation_lambda:float=.025,relation_audit_only:bool=True,relation_audit_every:int=512,ema_decay:float=.99,min_prototype_count:int=8,margin_slack:float=.05,margin_cap:float=.5,**kwargs):
  super().__init__(*args,**kwargs)
  if relation_lambda<0: raise ValueError('relation_lambda must be non-negative')
  if relation_audit_every<=0: raise ValueError('relation_audit_every must be positive')
  if not 0<=ema_decay<1: raise ValueError('ema_decay must be in [0,1)')
  if min_prototype_count<=0: raise ValueError('min_prototype_count must be positive')
  if margin_slack<0 or margin_cap<=0: raise ValueError('invalid margin guard')
  self.prototype_relation_lambda=float(relation_lambda); self.prototype_relation_audit_only=bool(relation_audit_only); self.prototype_relation_audit_every=int(relation_audit_every); self.prototype_relation_ema_decay=float(ema_decay); self.prototype_relation_min_count=int(min_prototype_count); self.prototype_relation_margin_slack=float(margin_slack); self.prototype_relation_margin_cap=float(margin_cap)
  self._prototype_teacher=copy.deepcopy(self.model).to(self.device).eval()
  for parameter in self._prototype_teacher.parameters(): parameter.requires_grad_(False)
  self._teacher_proto_sum={}; self._teacher_proto_count={}; self._prototype_relation_history=[]
 @staticmethod
 def _model_features(model,x):
  out=F.relu(model.bn1(model.conv1(x))); out=model.layer1(out); out=model.layer2(out); out=model.layer3(out); out=model.layer4(out); return F.adaptive_avg_pool2d(out,1).flatten(1)
 @staticmethod
 def _centered_correlation(first,second):
  if first.numel()<2: return 0.0
  first=first.float()-first.float().mean(); second=second.float()-second.float().mean(); denom=torch.linalg.vector_norm(first)*torch.linalg.vector_norm(second)
  return 0.0 if float(denom.detach())<=1e-12 else float((first@second/denom).detach())
 @staticmethod
 def _lazy_margin_loss(student_margin,teacher_margin,reliable,slack=.05,cap=.5):
  if student_margin.shape!=teacher_margin.shape or reliable.shape!=student_margin.shape: raise ValueError('margin tensors and reliable mask must match')
  target=teacher_margin.detach().clamp(min=0.0,max=float(cap)); deficit=F.relu(target-float(slack)-student_margin)
  loss=deficit[reliable].square().mean() if bool(reliable.any()) else student_margin.sum()*0.0
  return loss,deficit,reliable & deficit.detach().gt(0)
 @staticmethod
 def _gradient_cosine(first_loss,second_loss,params):
  first=torch.autograd.grad(first_loss,params,retain_graph=True,allow_unused=True); second=torch.autograd.grad(second_loss,params,retain_graph=True,allow_unused=True); dot=first_loss.new_tensor(0.0); first_norm=first_loss.new_tensor(0.0); second_norm=first_loss.new_tensor(0.0)
  for left,right in zip(first,second):
   if left is None or right is None: continue
   dot+=left.flatten()@right.flatten(); first_norm+=left.square().sum(); second_norm+=right.square().sum()
  denom=first_norm.sqrt()*second_norm.sqrt(); return None if float(denom.detach())<=1e-12 else float((dot/denom).detach())
 def _prototype_matrix(self):
  labels=[label for label in sorted(self._teacher_proto_sum) if self._teacher_proto_count[label]>=self.prototype_relation_min_count]
  if len(labels)<2: return labels,None
  matrix=torch.stack([self._teacher_proto_sum[label]/self._teacher_proto_count[label] for label in labels]).to(self.device)
  return labels,F.normalize(matrix.float(),dim=1)
 def _add_auxiliary_loss(self,current_x,current_y,current_tid,replay,replay_output=None):
  if replay is None or replay_output is None: return
  prototype_labels,prototypes=self._prototype_matrix()
  if prototypes is None: return
  label_to_index={label:index for index,label in enumerate(prototype_labels)}; replay_labels=replay[1]; valid=torch.tensor([int(label) in label_to_index for label in replay_labels.tolist()],dtype=torch.bool,device=self.device)
  if not bool(valid.any()): return
  teacher_features=F.normalize(self._model_features(self._prototype_teacher,replay[0]).detach().float(),dim=1)
  was_training=self.model.training; self.model.eval()
  try: student_features=F.normalize(self._model_features(self.model,replay[0]).float(),dim=1)
  finally: self.model.train(was_training)
  teacher_relation=teacher_features@prototypes.T; student_relation=student_features@prototypes.T
  own_indices=torch.tensor([label_to_index.get(int(label),0) for label in replay_labels.tolist()],dtype=torch.long,device=self.device); rows=torch.arange(replay_labels.numel(),device=self.device); teacher_own=teacher_relation[rows,own_indices]; student_own=student_relation[rows,own_indices]; teacher_rivals=teacher_relation.clone(); student_rivals=student_relation.clone(); teacher_rivals[rows,own_indices]=-torch.inf; student_rivals[rows,own_indices]=-torch.inf; teacher_margin=teacher_own-teacher_rivals.max(dim=1).values; student_margin=student_own-student_rivals.max(dim=1).values; reliable=valid & teacher_relation.argmax(dim=1).eq(own_indices) & teacher_margin.gt(0)
  relation_loss,deficit,active=self._lazy_margin_loss(student_margin,teacher_margin,reliable,self.prototype_relation_margin_slack,self.prototype_relation_margin_cap)
  if not self.prototype_relation_audit_only: self.loss+=self.prototype_relation_lambda*relation_loss
  clock=self._seen_samples+int(current_y.numel())
  if self._epoch_index!=0 or clock%self.prototype_relation_audit_every: return
  chosen=torch.arange(replay_output.shape[0],device=self.device); correct=replay_output.detach()[chosen,replay_labels]; rivals=replay_output.detach().clone(); rivals[chosen,replay_labels]=-torch.inf; classification_margin=correct-rivals.max(dim=1).values; replay_losses=F.cross_entropy(replay_output,replay_labels,reduction='none'); params=[parameter for name,parameter in self.model.named_parameters() if parameter.requires_grad and not name.startswith('linear.')]
  replay_cosine=self._gradient_cosine(relation_loss,self._auxiliary_replay_loss,params) if bool(active.any()) and self._auxiliary_replay_loss is not None else None; current_cosine=self._gradient_cosine(relation_loss,self._auxiliary_current_loss,params) if bool(active.any()) and self._auxiliary_current_loss is not None else None
  inactive=reliable & ~active; reliable_count=max(1,int(reliable.sum())); active_count=int(active.sum()); inactive_count=int(inactive.sum())
  self._prototype_relation_history.append({'sample_clock':float(clock),'prototype_class_count':len(prototype_labels),'target_coverage':float(valid.float().mean()),'teacher_label_accuracy_on_covered':float((teacher_relation.argmax(dim=1)[valid]==own_indices[valid]).float().mean()),'reliable_target_fraction':float(reliable.float().mean()),'teacher_margin_mean':float(teacher_margin[reliable].mean()) if bool(reliable.any()) else None,'teacher_margin_std':float(teacher_margin[reliable].std(unbiased=False)) if bool(reliable.any()) else None,'student_teacher_margin_correlation':self._centered_correlation(student_margin[reliable].detach(),teacher_margin[reliable]) if int(reliable.sum())>=2 else None,'active_fraction':float(active.float().mean()),'active_fraction_of_reliable':active_count/reliable_count,'active_replay_ce':float(replay_losses[active].mean()) if active_count else None,'inactive_reliable_replay_ce':float(replay_losses[inactive].mean()) if inactive_count else None,'active_negative_classification_margin_fraction':float((classification_margin[active]<=0).float().mean()) if active_count else None,'inactive_negative_classification_margin_fraction':float((classification_margin[inactive]<=0).float().mean()) if inactive_count else None,'teacher_margin_vulnerability_correlation':self._centered_correlation(teacher_margin[reliable].detach(),-classification_margin[reliable]) if int(reliable.sum())>=2 else None,'relation_loss':float(relation_loss.detach()),'mean_active_deficit':float(deficit[active].mean()) if active_count else 0.0,'replay_relation_gradient_cosine':replay_cosine,'current_relation_gradient_cosine':current_cosine})
 def _after_anchor_update(self,current_x,current_y,current_tid):
  with torch.no_grad():
   decay=self.prototype_relation_ema_decay
   for teacher,student in zip(self._prototype_teacher.parameters(),self.model.parameters()): teacher.mul_(decay).add_(student.detach(),alpha=1-decay)
   for teacher,student in zip(self._prototype_teacher.buffers(),self.model.buffers()): teacher.copy_(student.detach())
   features=self._model_features(self._prototype_teacher,current_x.to(self.device)).detach().float()
   for feature,label in zip(features,current_y.tolist()):
    label=int(label); self._teacher_proto_sum[label]=self._teacher_proto_sum.get(label,torch.zeros_like(feature))+feature; self._teacher_proto_count[label]=self._teacher_proto_count.get(label,0)+1
 def rbcl_summary(self):
  r=super().rbcl_summary(); h=self._prototype_relation_history
  r['causal_prototype_boundary_relation_v3']={'enabled':not self.prototype_relation_audit_only,'audit_only':self.prototype_relation_audit_only,'lambda':self.prototype_relation_lambda,'ema_decay':self.prototype_relation_ema_decay,'minimum_prototype_count':self.prototype_relation_min_count,'margin_slack':self.prototype_relation_margin_slack,'margin_cap':self.prototype_relation_margin_cap,'target_is_ema_teacher_class_prototype_boundary':True,'regresses_imagenet_current_replay_cosine':False,'lazy_exact_zero_when_student_is_safe':True,'uses_task_id_boundary_validation_or_future_data':False,'audit_count':len(h),'history':h}; return r


class TIRPMaturityNormalizedRelationERACE(TIRPPrototypeBoundaryRelationERACE):
 """Relation V4: maturity-gated, scale-normalized prototype preservation."""
 def __init__(self,*args,relative_slack:float=.25,margin_scale_floor:float=.005,maturity_threshold:float=.5,maturity_ema_decay:float=.9,maturity_min_updates:int=8,**kwargs):
  super().__init__(*args,margin_slack=0.0,**kwargs)
  if not 0<=relative_slack<1: raise ValueError('relative_slack must be in [0,1)')
  if margin_scale_floor<=0: raise ValueError('margin_scale_floor must be positive')
  if not 0<maturity_threshold<=1: raise ValueError('maturity_threshold must be in (0,1]')
  if not 0<=maturity_ema_decay<1: raise ValueError('maturity_ema_decay must be in [0,1)')
  if maturity_min_updates<=0: raise ValueError('maturity_min_updates must be positive')
  self.prototype_relation_relative_slack=float(relative_slack); self.prototype_relation_margin_scale_floor=float(margin_scale_floor); self.prototype_relation_maturity_threshold=float(maturity_threshold); self.prototype_relation_maturity_ema_decay=float(maturity_ema_decay); self.prototype_relation_maturity_min_updates=int(maturity_min_updates)
  self._prototype_reliability_ema=None; self._prototype_reliability_updates=0; self._prototype_maturity_streak=0; self._prototype_mature=False; self._prototype_maturity_clock=None; self._prototype_v4_history=[]
 @staticmethod
 def _ema_update(previous,current,decay):
  return float(current) if previous is None else float(decay)*float(previous)+(1-float(decay))*float(current)
 @staticmethod
 def _normalized_margin_loss(student_margin,teacher_margin,reliable,mature,relative_slack=.25,scale_floor=.005,cap=.5):
  if student_margin.shape!=teacher_margin.shape or reliable.shape!=student_margin.shape: raise ValueError('margin tensors and reliable mask must match')
  target=teacher_margin.detach().clamp(min=0.0,max=float(cap))
  scale=target[reliable].median().clamp_min(float(scale_floor)) if bool(reliable.any()) else target.new_tensor(float(scale_floor))
  normalized_gap=(target-student_margin)/scale
  deficit=F.relu(normalized_gap-float(relative_slack))
  active=reliable & deficit.detach().gt(0) & bool(mature)
  loss=F.smooth_l1_loss(deficit[reliable],torch.zeros_like(deficit[reliable])) if bool(mature) and bool(reliable.any()) else student_margin.sum()*0.0
  return loss,deficit,active,scale
 def _add_auxiliary_loss(self,current_x,current_y,current_tid,replay,replay_output=None):
  if replay is None or replay_output is None: return
  prototype_labels,prototypes=self._prototype_matrix()
  if prototypes is None: return
  label_to_index={label:index for index,label in enumerate(prototype_labels)}; replay_labels=replay[1]; valid=torch.tensor([int(label) in label_to_index for label in replay_labels.tolist()],dtype=torch.bool,device=self.device)
  if not bool(valid.any()): return
  teacher_features=F.normalize(self._model_features(self._prototype_teacher,replay[0]).detach().float(),dim=1)
  was_training=self.model.training; self.model.eval()
  try: student_features=F.normalize(self._model_features(self.model,replay[0]).float(),dim=1)
  finally: self.model.train(was_training)
  teacher_relation=teacher_features@prototypes.T; student_relation=student_features@prototypes.T
  own_indices=torch.tensor([label_to_index.get(int(label),0) for label in replay_labels.tolist()],dtype=torch.long,device=self.device); rows=torch.arange(replay_labels.numel(),device=self.device); teacher_own=teacher_relation[rows,own_indices]; student_own=student_relation[rows,own_indices]; teacher_rivals=teacher_relation.clone(); student_rivals=student_relation.clone(); teacher_rivals[rows,own_indices]=-torch.inf; student_rivals[rows,own_indices]=-torch.inf; teacher_margin=teacher_own-teacher_rivals.max(dim=1).values; student_margin=student_own-student_rivals.max(dim=1).values; reliable=valid & teacher_relation.argmax(dim=1).eq(own_indices) & teacher_margin.gt(0)
  clock=self._seen_samples+int(current_y.numel())
  if self._epoch_index==0:
   batch_reliability=float(reliable.float().sum()/valid.float().sum().clamp_min(1)); self._prototype_reliability_ema=self._ema_update(self._prototype_reliability_ema,batch_reliability,self.prototype_relation_maturity_ema_decay); self._prototype_reliability_updates+=1
   self._prototype_maturity_streak=self._prototype_maturity_streak+1 if self._prototype_reliability_ema>=self.prototype_relation_maturity_threshold else 0
   if not self._prototype_mature and self._prototype_maturity_streak>=self.prototype_relation_maturity_min_updates: self._prototype_mature=True; self._prototype_maturity_clock=float(clock)
  relation_loss,deficit,active,margin_scale=self._normalized_margin_loss(student_margin,teacher_margin,reliable,self._prototype_mature,self.prototype_relation_relative_slack,self.prototype_relation_margin_scale_floor,self.prototype_relation_margin_cap)
  if not self.prototype_relation_audit_only: self.loss+=self.prototype_relation_lambda*relation_loss
  if self._epoch_index!=0 or clock%self.prototype_relation_audit_every: return
  chosen=torch.arange(replay_output.shape[0],device=self.device); correct=replay_output.detach()[chosen,replay_labels]; rivals=replay_output.detach().clone(); rivals[chosen,replay_labels]=-torch.inf; classification_margin=correct-rivals.max(dim=1).values; replay_losses=F.cross_entropy(replay_output,replay_labels,reduction='none'); params=[parameter for name,parameter in self.model.named_parameters() if parameter.requires_grad and not name.startswith('linear.')]
  replay_cosine=self._gradient_cosine(relation_loss,self._auxiliary_replay_loss,params) if bool(active.any()) and self._auxiliary_replay_loss is not None else None; current_cosine=self._gradient_cosine(relation_loss,self._auxiliary_current_loss,params) if bool(active.any()) and self._auxiliary_current_loss is not None else None
  inactive=reliable & ~active; reliable_count=max(1,int(reliable.sum())); active_count=int(active.sum()); inactive_count=int(inactive.sum())
  self._prototype_v4_history.append({'sample_clock':float(clock),'prototype_class_count':len(prototype_labels),'target_coverage':float(valid.float().mean()),'teacher_label_accuracy_on_covered':float((teacher_relation.argmax(dim=1)[valid]==own_indices[valid]).float().mean()),'reliable_target_fraction':float(reliable.float().mean()),'causal_reliability_ema':self._prototype_reliability_ema,'maturity_streak':self._prototype_maturity_streak,'mature':self._prototype_mature,'maturity_clock':self._prototype_maturity_clock,'teacher_margin_mean':float(teacher_margin[reliable].mean()) if bool(reliable.any()) else None,'normalized_margin_scale':float(margin_scale.detach()),'active_fraction':float(active.float().mean()),'active_fraction_of_reliable':active_count/reliable_count,'active_replay_ce':float(replay_losses[active].mean()) if active_count else None,'inactive_reliable_replay_ce':float(replay_losses[inactive].mean()) if inactive_count else None,'active_negative_classification_margin_fraction':float((classification_margin[active]<=0).float().mean()) if active_count else None,'inactive_negative_classification_margin_fraction':float((classification_margin[inactive]<=0).float().mean()) if inactive_count else None,'relation_loss':float(relation_loss.detach()),'mean_active_normalized_deficit':float(deficit[active].mean()) if active_count else 0.0,'replay_relation_gradient_cosine':replay_cosine,'current_relation_gradient_cosine':current_cosine})
 def _after_anchor_update(self,current_x,current_y,current_tid):
  with torch.no_grad():
   decay=self.prototype_relation_ema_decay
   for teacher,student in zip(self._prototype_teacher.parameters(),self.model.parameters()): teacher.mul_(decay).add_(student.detach(),alpha=1-decay)
   for teacher,student in zip(self._prototype_teacher.buffers(),self.model.buffers()): teacher.copy_(student.detach())
   if self._epoch_index!=0: return
   features=self._model_features(self._prototype_teacher,current_x.to(self.device)).detach().float()
   for feature,label in zip(features,current_y.tolist()):
    label=int(label); self._teacher_proto_sum[label]=self._teacher_proto_sum.get(label,torch.zeros_like(feature))+feature; self._teacher_proto_count[label]=self._teacher_proto_count.get(label,0)+1
 def rbcl_summary(self):
  r=super().rbcl_summary(); r.pop('causal_prototype_boundary_relation_v3',None); h=self._prototype_v4_history
  r['maturity_normalized_prototype_relation_v4']={'enabled':not self.prototype_relation_audit_only,'audit_only':self.prototype_relation_audit_only,'lambda':self.prototype_relation_lambda,'ema_decay':self.prototype_relation_ema_decay,'minimum_unique_arrival_count_per_prototype':self.prototype_relation_min_count,'prototypes_count_unique_epoch_zero_arrivals_only':True,'relative_slack':self.prototype_relation_relative_slack,'margin_scale_floor':self.prototype_relation_margin_scale_floor,'maturity_threshold':self.prototype_relation_maturity_threshold,'maturity_ema_decay':self.prototype_relation_maturity_ema_decay,'maturity_required_consecutive_updates':self.prototype_relation_maturity_min_updates,'maturity_is_causal_and_latched':True,'mature':self._prototype_mature,'maturity_clock':self._prototype_maturity_clock,'uses_batch_median_reliable_teacher_margin_scale':True,'uses_absolute_margin_slack':False,'uses_task_id_boundary_validation_or_future_data':False,'audit_count':len(h),'history':h}; return r


class TIRPSparseBudgetedRelationERACE(TIRPMaturityNormalizedRelationERACE):
 """Relation V5: batch-local reliable targets with a sparse activation budget."""
 def __init__(self,*args,min_reliable_targets:int=4,max_active_fraction:float=.125,**kwargs):
  super().__init__(*args,**kwargs)
  if min_reliable_targets<=0: raise ValueError('min_reliable_targets must be positive')
  if not 0<max_active_fraction<=1: raise ValueError('max_active_fraction must be in (0,1]')
  self.prototype_relation_min_reliable_targets=int(min_reliable_targets); self.prototype_relation_max_active_fraction=float(max_active_fraction); self._prototype_v5_history=[]
 @staticmethod
 def _sparse_budgeted_margin_loss(student_margin,teacher_margin,reliable,relative_slack=.25,scale_floor=.005,cap=.5,min_reliable_targets=4,max_active_fraction=.125):
  if student_margin.shape!=teacher_margin.shape or reliable.shape!=student_margin.shape: raise ValueError('margin tensors and reliable mask must match')
  if min_reliable_targets<=0: raise ValueError('min_reliable_targets must be positive')
  if not 0<max_active_fraction<=1: raise ValueError('max_active_fraction must be in (0,1]')
  target=teacher_margin.detach().clamp(min=0.0,max=float(cap)); scale=target[reliable].median().clamp_min(float(scale_floor)) if bool(reliable.any()) else target.new_tensor(float(scale_floor)); normalized_gap=(target-student_margin)/scale; deficit=F.relu(normalized_gap-float(relative_slack)); active=torch.zeros_like(reliable,dtype=torch.bool)
  reliable_count=int(reliable.sum()); candidates=torch.where(reliable & deficit.detach().gt(0))[0]
  if reliable_count>=int(min_reliable_targets) and int(candidates.numel())>0:
   budget=max(1,math.ceil(student_margin.numel()*float(max_active_fraction))); count=min(budget,int(candidates.numel())); selected=torch.topk(deficit.detach()[candidates],k=count,largest=True,sorted=False).indices; active[candidates[selected]]=True
  loss=F.smooth_l1_loss(deficit[active],torch.zeros_like(deficit[active])) if bool(active.any()) else student_margin.sum()*0.0
  return loss,deficit,active,scale
 def _add_auxiliary_loss(self,current_x,current_y,current_tid,replay,replay_output=None):
  if replay is None or replay_output is None: return
  prototype_labels,prototypes=self._prototype_matrix()
  if prototypes is None: return
  label_to_index={label:index for index,label in enumerate(prototype_labels)}; replay_labels=replay[1]; valid=torch.tensor([int(label) in label_to_index for label in replay_labels.tolist()],dtype=torch.bool,device=self.device)
  if not bool(valid.any()): return
  teacher_features=F.normalize(self._model_features(self._prototype_teacher,replay[0]).detach().float(),dim=1)
  was_training=self.model.training; self.model.eval()
  try: student_features=F.normalize(self._model_features(self.model,replay[0]).float(),dim=1)
  finally: self.model.train(was_training)
  teacher_relation=teacher_features@prototypes.T; student_relation=student_features@prototypes.T
  own_indices=torch.tensor([label_to_index.get(int(label),0) for label in replay_labels.tolist()],dtype=torch.long,device=self.device); rows=torch.arange(replay_labels.numel(),device=self.device); teacher_own=teacher_relation[rows,own_indices]; student_own=student_relation[rows,own_indices]; teacher_rivals=teacher_relation.clone(); student_rivals=student_relation.clone(); teacher_rivals[rows,own_indices]=-torch.inf; student_rivals[rows,own_indices]=-torch.inf; teacher_margin=teacher_own-teacher_rivals.max(dim=1).values; student_margin=student_own-student_rivals.max(dim=1).values; reliable=valid & teacher_relation.argmax(dim=1).eq(own_indices) & teacher_margin.gt(0)
  relation_loss,deficit,active,margin_scale=self._sparse_budgeted_margin_loss(student_margin,teacher_margin,reliable,self.prototype_relation_relative_slack,self.prototype_relation_margin_scale_floor,self.prototype_relation_margin_cap,self.prototype_relation_min_reliable_targets,self.prototype_relation_max_active_fraction)
  if not self.prototype_relation_audit_only: self.loss+=self.prototype_relation_lambda*relation_loss
  clock=self._seen_samples+int(current_y.numel())
  if self._epoch_index!=0 or clock%self.prototype_relation_audit_every: return
  chosen=torch.arange(replay_output.shape[0],device=self.device); correct=replay_output.detach()[chosen,replay_labels]; rivals=replay_output.detach().clone(); rivals[chosen,replay_labels]=-torch.inf; classification_margin=correct-rivals.max(dim=1).values; replay_losses=F.cross_entropy(replay_output,replay_labels,reduction='none'); params=[parameter for name,parameter in self.model.named_parameters() if parameter.requires_grad and not name.startswith('linear.')]
  replay_cosine=self._gradient_cosine(relation_loss,self._auxiliary_replay_loss,params) if bool(active.any()) and self._auxiliary_replay_loss is not None else None; current_cosine=self._gradient_cosine(relation_loss,self._auxiliary_current_loss,params) if bool(active.any()) and self._auxiliary_current_loss is not None else None
  inactive=reliable & ~active; reliable_count=int(reliable.sum()); active_count=int(active.sum()); inactive_count=int(inactive.sum()); candidate_count=int((reliable & deficit.detach().gt(0)).sum()); budget=max(1,math.ceil(replay_labels.numel()*self.prototype_relation_max_active_fraction))
  self._prototype_v5_history.append({'sample_clock':float(clock),'prototype_class_count':len(prototype_labels),'target_coverage':float(valid.float().mean()),'teacher_label_accuracy_on_covered':float((teacher_relation.argmax(dim=1)[valid]==own_indices[valid]).float().mean()),'reliable_target_fraction':float(reliable.float().mean()),'reliable_target_count':reliable_count,'minimum_reliable_target_count':self.prototype_relation_min_reliable_targets,'batch_gate_open':reliable_count>=self.prototype_relation_min_reliable_targets,'candidate_active_count':candidate_count,'activation_budget_count':budget,'active_count':active_count,'active_fraction':float(active.float().mean()),'active_fraction_of_reliable':active_count/max(1,reliable_count),'teacher_margin_mean':float(teacher_margin[reliable].mean()) if reliable_count else None,'normalized_margin_scale':float(margin_scale.detach()),'active_replay_ce':float(replay_losses[active].mean()) if active_count else None,'inactive_reliable_replay_ce':float(replay_losses[inactive].mean()) if inactive_count else None,'active_negative_classification_margin_fraction':float((classification_margin[active]<=0).float().mean()) if active_count else None,'inactive_negative_classification_margin_fraction':float((classification_margin[inactive]<=0).float().mean()) if inactive_count else None,'relation_loss':float(relation_loss.detach()),'mean_active_normalized_deficit':float(deficit[active].mean()) if active_count else 0.0,'replay_relation_gradient_cosine':replay_cosine,'current_relation_gradient_cosine':current_cosine})
 def rbcl_summary(self):
  r=super().rbcl_summary(); r.pop('maturity_normalized_prototype_relation_v4',None); h=self._prototype_v5_history
  r['sparse_budgeted_prototype_relation_v5']={'enabled':not self.prototype_relation_audit_only,'audit_only':self.prototype_relation_audit_only,'lambda':self.prototype_relation_lambda,'ema_decay':self.prototype_relation_ema_decay,'minimum_unique_arrival_count_per_prototype':self.prototype_relation_min_count,'prototypes_count_unique_epoch_zero_arrivals_only':True,'relative_slack':self.prototype_relation_relative_slack,'margin_scale_floor':self.prototype_relation_margin_scale_floor,'minimum_reliable_targets_per_batch':self.prototype_relation_min_reliable_targets,'maximum_active_fraction_per_batch':self.prototype_relation_max_active_fraction,'uses_global_maturity_latch':False,'uses_batch_local_reversible_gate':True,'reliable_targets_require_observed_label_agreement_and_positive_teacher_margin':True,'uses_batch_median_reliable_teacher_margin_scale':True,'uses_task_id_boundary_validation_or_future_data':False,'audit_count':len(h),'history':h}; return r


class TIRPProxyRelationERACE(TIRPSemanticRepresentativeERACE):
 """Fixed TIRP plus causal, scale-free sample-to-classifier-proxy relations."""
 def __init__(self,*args,relation_lambda:float=.1,relation_audit_only:bool=False,relation_audit_every:int=512,**kwargs):
  super().__init__(*args,**kwargs)
  self.relation_lambda=float(relation_lambda); self.relation_audit_only=bool(relation_audit_only); self.relation_audit_every=int(relation_audit_every)
  self._relation_cache={}; self._last_relation_items=[]; self._last_relation_x=None; self._relation_audit_history=[]
 def _trainable_features(self,x):
  m=self.model; out=F.relu(m.bn1(m.conv1(x))); out=m.layer1(out); out=m.layer2(out); out=m.layer3(out); out=m.layer4(out); return F.adaptive_avg_pool2d(out,1).flatten(1)
 def _relations(self,x):
  features=F.normalize(self._trainable_features(x),dim=1); weights=F.normalize(self.model.linear.weight,dim=1)
  return features @ weights.T
 def _sample_memory(self):
  replay=super()._sample_memory(); self._last_relation_items=[]
  if replay is not None:
   for index in getattr(self,'_last_replay_memory_indices',[]):
    self._last_relation_items.append(self._memory_key(self._memory_x[index],self._memory_y[index]))
  return replay
 def _add_auxiliary_loss(self,current_x,current_y,current_tid,replay,replay_output=None):
  if replay is None or not self._last_relation_items: return
  self._last_relation_x=replay[0].detach()
  relation=self._relations(replay[0]); cached=[self._relation_cache.get(key) for key in self._last_relation_items]
  valid=torch.tensor([item is not None for item in cached],device=self.device)
  if bool(valid.any()):
   teacher=torch.stack([item if item is not None else torch.zeros_like(relation[0].detach().cpu()) for item in cached]).to(self.device)
   drift=F.mse_loss(relation[valid],teacher[valid])
   clock=self._seen_samples+int(current_y.numel())
   if self._epoch_index==0 and clock%self.relation_audit_every==0:
    self._relation_audit_history.append({'sample_clock':float(clock),'relation_target_coverage':float(valid.float().mean()),'relation_drift':float(drift.detach())})
   if not self.relation_audit_only: self.loss+=self.relation_lambda*drift
 def _after_update(self,**kwargs):
  if not self._last_relation_items or self._last_relation_x is None: return
  was_training=self.model.training; self.model.eval()
  try:
   with torch.no_grad(): relation=self._relations(self._last_relation_x).detach().cpu()
   for key,row in zip(self._last_relation_items,relation): self._relation_cache[key]=row.clone()
  finally:
   self.model.train(was_training)
 def rbcl_summary(self):
  r=super().rbcl_summary(); h=self._relation_audit_history
  r['causal_proxy_relation']={'enabled':not self.relation_audit_only,'audit_only':self.relation_audit_only,'lambda':self.relation_lambda,'audit_count':len(h),'history':h,'uses_task_id_loss_gradient_validation_or_controller':False}; return r


class TIRPDecisionConsolidatedERACE(TIRPSemanticRepresentativeERACE):
 """D40: frozen TIRP plus causal, fixed-weight self-logit consolidation.

 Logits are captured only after a sample's first causal update and only if the
 sample actually entered the already-bounded memory.  The stored outputs are
 therefore a past model state, not an external teacher or future-experience
 snapshot.  Memory and TIRP selection are inherited unchanged.
 """
 def __init__(self,*args,decision_lambda:float=.1,decision_audit_only:bool=False,decision_audit_every:int=512,decision_mature_age:int=0,decision_require_correct_target:bool=False,decision_mode:str='full_logits',**kwargs):
  super().__init__(*args,**kwargs)
  if decision_lambda<0: raise ValueError('decision_lambda must be non-negative')
  if decision_audit_every<=0: raise ValueError('decision_audit_every must be positive')
  if decision_mature_age<0: raise ValueError('decision_mature_age must be non-negative')
  if decision_mode not in {'full_logits','label_anchored_margin'}: raise ValueError('unsupported decision_mode')
  self.decision_lambda=float(decision_lambda); self.decision_audit_only=bool(decision_audit_only); self.decision_audit_every=int(decision_audit_every); self.decision_mature_age=int(decision_mature_age); self.decision_require_correct_target=bool(decision_require_correct_target); self.decision_mode=decision_mode
  self._dark_logits={}; self._dark_masks={}; self._dark_rivals={}; self._dark_margins={}; self._last_replay_dark_logits=None; self._last_replay_dark_masks=None; self._last_replay_dark_rivals=None; self._last_replay_dark_margins=None; self._decision_audit_history=[]
 def _cache_memory_logits(self,indices):
  if not indices: return
  was_training=self.model.training; self.model.eval()
  try:
   with torch.no_grad(): logits=self.model(torch.stack([self._memory_x[index] for index in indices]).to(self.device)).detach().cpu().float()
  finally: self.model.train(was_training)
  seen=sorted(self._observed_classes)
  for index,row in zip(indices,logits):
   if self.decision_require_correct_target and int(row.argmax())!=int(self._memory_y[index]): continue
   memory_x=self._memory_x[index]; key=self._memory_key(memory_x,self._memory_y[index]); self._dark_logits[key]=row.clone()
   mask=torch.zeros(row.shape[0],dtype=torch.bool); mask[seen]=True; self._dark_masks[key]=mask
   label=int(self._memory_y[index]); rival_mask=mask.clone(); rival_mask[label]=False
   if bool(rival_mask.any()):
    rival=int(row.masked_fill(~rival_mask,float('-inf')).argmax())
    self._dark_rivals[key]=rival; self._dark_margins[key]=float(row[label]-row[rival])
 def _after_memory_update(self,current_x,current_y,current_tid):
  present={self._memory_key(item,label) for item,label in zip(self._memory_x,self._memory_y)}
  self._dark_logits={key:value for key,value in self._dark_logits.items() if key in present}
  self._dark_masks={key:value for key,value in self._dark_masks.items() if key in present}
  self._dark_rivals={key:value for key,value in self._dark_rivals.items() if key in present}
  self._dark_margins={key:value for key,value in self._dark_margins.items() if key in present}
  if self.decision_mature_age>0:
   clock=max(self._seen_samples,self._hybrid_seen)
   previous=max(0,clock-int(current_y.numel()))
   if previous//self.decision_audit_every==clock//self.decision_audit_every: return
   mature=[index for index,memory_x in enumerate(self._memory_x) if self._memory_key(memory_x,self._memory_y[index]) not in self._dark_logits and clock-self._memory_arrival.get(self._memory_key(memory_x,self._memory_y[index]),clock)>=self.decision_mature_age]
   self._cache_memory_logits(mature); return
  was_training=self.model.training; self.model.eval()
  try:
   with torch.no_grad(): logits=self.model(current_x.to(self.device)).detach().cpu().float()
  finally: self.model.train(was_training)
  current_cpu=current_x.detach().cpu(); seen=sorted(self._observed_classes)
  for memory_index,memory_x in enumerate(self._memory_x):
   for index,sample_x in enumerate(current_cpu):
    if torch.equal(memory_x,sample_x):
     key=self._memory_key(memory_x,self._memory_y[memory_index]); self._dark_logits[key]=logits[index].clone()
     mask=torch.zeros(logits.shape[1],dtype=torch.bool); mask[seen]=True; self._dark_masks[key]=mask
     break
 def _sample_memory(self):
  replay=super()._sample_memory()
  self._last_replay_dark_logits=None; self._last_replay_dark_masks=None
  self._last_replay_dark_rivals=None; self._last_replay_dark_margins=None
  if replay is None: return None
  indices=getattr(self,'_last_replay_memory_indices',[])
  if len(indices)!=replay[1].numel(): return replay
  stored=[self._dark_logits.get(self._memory_key(self._memory_x[index],self._memory_y[index])) for index in indices]
  masks=[self._dark_masks.get(self._memory_key(self._memory_x[index],self._memory_y[index])) for index in indices]
  available=[item is not None and mask is not None for item,mask in zip(stored,masks)]
  if any(available):
   width=next(item.shape[0] for item in stored if item is not None)
   self._last_replay_dark_logits=torch.stack([item if item is not None else torch.zeros(width) for item in stored]).to(self.device)
   self._last_replay_dark_masks=torch.stack([mask if mask is not None else torch.zeros(width,dtype=torch.bool) for mask in masks]).to(self.device)
  rivals=[self._dark_rivals.get(self._memory_key(self._memory_x[index],self._memory_y[index]),-1) for index in indices]
  margins=[self._dark_margins.get(self._memory_key(self._memory_x[index],self._memory_y[index]),0.0) for index in indices]
  if any(rival>=0 for rival in rivals):
   self._last_replay_dark_rivals=torch.tensor(rivals,dtype=torch.long,device=self.device)
   self._last_replay_dark_margins=torch.tensor(margins,dtype=torch.float32,device=self.device)
  return replay
 def _add_auxiliary_loss(self,current_x,current_y,current_tid,replay,replay_output=None):
  if replay is None or replay_output is None: return
  if self.decision_mode=='label_anchored_margin':
   if self._last_replay_dark_rivals is None or self._last_replay_dark_margins is None: return
   labels=replay[1]; rivals=self._last_replay_dark_rivals; valid=(rivals>=0)&(rivals<replay_output.shape[1])&(labels<replay_output.shape[1])
   if not bool(valid.any()): return
   chosen=torch.arange(replay_output.shape[0],device=self.device)[valid]
   current_margin=replay_output[chosen,labels[valid]]-replay_output[chosen,rivals[valid]]
   teacher_margin=self._last_replay_dark_margins[valid]
   drift=(current_margin-teacher_margin).pow(2).mean()
   next_clock=self._seen_samples+int(current_y.numel())
   if self._epoch_index==0 and next_clock%self.decision_audit_every==0:
    self._decision_audit_history.append({'sample_clock':float(next_clock),'replay_target_coverage':float(valid.float().mean()),'replay_margin_drift':float(drift.detach()),'stored_margin_mean':float(teacher_margin.mean()),'current_margin_mean':float(current_margin.detach().mean()),'stored_logit_label_accuracy':1.0,'current_replay_label_accuracy':float((replay_output[valid].argmax(dim=1)==labels[valid]).float().mean())})
   if not self.decision_audit_only: self.loss+=self.decision_lambda*drift
   return
  if self._last_replay_dark_logits is None: return
  width=min(replay_output.shape[1],self._last_replay_dark_logits.shape[1])
  teacher=self._last_replay_dark_logits[:,:width]; mask=self._last_replay_dark_masks[:,:width].float()
  drift=(((replay_output[:,:width]-teacher).pow(2))*mask).sum()/mask.sum().clamp_min(1.0)
  next_clock=self._seen_samples+int(current_y.numel())
  if self._epoch_index==0 and next_clock%self.decision_audit_every==0:
   labels=replay[1]; valid=mask.sum(dim=1)>0; teacher_pred=teacher[valid].masked_fill(mask[valid]==0,float('-inf')).argmax(dim=1)
   self._decision_audit_history.append({'sample_clock':float(next_clock),'replay_target_coverage':float(valid.float().mean()),'replay_logit_drift':float(drift.detach()),'stored_logit_label_accuracy':float((teacher_pred==labels[valid]).float().mean()) if bool(valid.any()) else 0.0,'current_replay_label_accuracy':float((replay_output[valid].argmax(dim=1)==labels[valid]).float().mean()) if bool(valid.any()) else 0.0})
  if not self.decision_audit_only: self.loss+=self.decision_lambda*drift
 def rbcl_summary(self):
  r=super().rbcl_summary(); h=self._decision_audit_history
  drift_key='replay_margin_drift' if self.decision_mode=='label_anchored_margin' else 'replay_logit_drift'
  r['causal_decision_consolidation']={'enabled':not self.decision_audit_only,'audit_only':self.decision_audit_only,'mode':self.decision_mode,'lambda':self.decision_lambda,'mature_age_samples':self.decision_mature_age,'requires_correct_target_at_cache':self.decision_require_correct_target,'stores_only_bounded_memory_logits':True,'captures_logits_after_first_causal_update':self.decision_mature_age==0,'uses_task_id_loss_gradient_validation_or_controller':False,'audit_count':len(h),'history':h,'mean_replay_target_drift':(sum(v[drift_key] for v in h)/len(h) if h else 0.0),'mean_target_coverage':(sum(v['replay_target_coverage'] for v in h)/len(h) if h else 0.0)}; return r
