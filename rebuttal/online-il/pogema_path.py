"""Live original Pogema/MAGAT path with only lifelong dependency adaptation.

Keeps original lazy per-agent distance tables and model preprocessing. Reuses
the observations returned by Pogema.step rather than constructing them twice.
"""
import time
import resource
import numpy as np
from tasks import assign_goals
from inputs import combine_inputs


class ObservedEnv:
    def __init__(self,env,observations):self.env=env;self.observations=observations
    def __getattr__(self,name):return getattr(self.env,name)
    def _obs(self):return self.observations


class OriginalPogemaLifelong:
    def __init__(self,tasks,prepare=True,with_inputs=True,mode='lifelong'):
        from pogema.envs import PogemaCoopFinish
        from pogema.grid_config import GridConfig
        from magat_plus.additional_data.cost_to_go_calculator import CostToGoCalculator
        self.tasks=tasks;self.envs=[];self.calculators=[];self.observations=[];self.with_inputs=with_inputs
        self.mode=mode
        self.goals=tasks.goals.copy();self.cursors=np.zeros(tasks.goals.shape[:2],dtype=np.int32)
        self.arrived=np.array(getattr(tasks,'initial_arrived',np.all(tasks.positions==tasks.goals,axis=-1)),dtype=bool,copy=True)
        self.steps=0;self.goal_bins=[];self.build_seconds=0.;self.step_seconds=0.;self.refreshed_goals=0
        grids=getattr(tasks,'grids',[tasks.grid]*len(tasks.positions))
        for pos,goal,grid in zip(tasks.positions,tasks.goals,grids):
            cfg=GridConfig(map=grid.tolist(),agents_xy=pos.tolist(),targets_xy=goal.tolist(),
                num_agents=len(pos),obs_radius=5,on_target='nothing',collision_system='soft',
                observation_type='MAPF',max_episode_steps=4096,empty_outside=True)
            env=PogemaCoopFinish(cfg);env._initialize_grid();env.update_was_on_goal()
            self.envs.append(env);self.observations.append(env._obs())
            if with_inputs:
                self.calculators.append(CostToGoCalculator(env,5,dtype='float32',pad_cost_to_go=True,
                    clamp_value=1.,clamp_values_doubled=False))
        if prepare:
            self.build();self.build_seconds=0.

    def positions(self):
        return np.asarray([e.grid.get_agents_xy(ignore_borders=True) for e in self.envs],dtype=np.int16)

    def rows(self,actions=None,env_ids=None):
        n,a=self.goals.shape[:2];rows=np.zeros((n,a,8),dtype=np.int16)
        rows[:,:,0]=np.arange(n)[:,None] if env_ids is None else np.asarray(env_ids)[:,None]
        rows[:,:,1]=np.arange(a);rows[:,:,2:4]=self.positions();rows[:,:,4:6]=self.goals
        if actions is not None:rows[:,:,6]=actions
        return rows.reshape(-1,8)

    def _refresh(self,e,changed):
        from magat_plus.additional_data.cost_to_go_calculator import DistTable
        self.refreshed_goals+=int(np.sum(changed))
        if not self.with_inputs:return
        env=self.envs[e];calc=self.calculators[e]
        for agent in np.flatnonzero(changed):
            grid=calc.dist_tables[agent].grid
            calc.dist_tables[agent]=DistTable(grid,tuple(map(int,self.goals[e,agent])))

    def references(self,labels=None):
        if not self.with_inputs:raise RuntimeError('expert-only environment has no input calculator')
        from expert.magat_reference_oracle import build_magat_reference_batch_from_env
        rows=self.rows(labels).reshape(len(self.envs),-1,8)
        return [build_magat_reference_batch_from_env(ObservedEnv(env,obs),row,
            cost_to_go_calculator=calc) for env,obs,row,calc in
            zip(self.envs,self.observations,rows,self.calculators)]

    def build(self):
        started=time.perf_counter();refs=self.references()
        arrays=combine_inputs([(r.x,r.edge_index,r.edge_attr) for r in refs],self.goals.shape[1])
        self.build_seconds+=time.perf_counter()-started
        return arrays

    def step(self,actions):
        started=time.perf_counter()
        for e,(env,action) in enumerate(zip(self.envs,actions)):
            effective=np.array(action,dtype=np.int64,copy=True)
            if self.mode=='standard_mapf':effective[self.arrived[e]]=0
            observations,_,_,_,_=env.step(effective)
            pos=np.asarray(env.grid.get_agents_xy(ignore_borders=True),dtype=np.int16)
            reached=np.all(pos==self.goals[e],axis=1)
            if self.mode=='standard_mapf':
                newly=reached & ~self.arrived[e]
                self.cursors[e,newly]+=1;self.arrived[e]|=reached
                self.observations[e]=observations
                continue
            cells=self.tasks.cells_for_env(e) if hasattr(self.tasks,'cells_for_env') else self.tasks.cells
            assign_goals(pos,self.goals[e],self.cursors[e],self.tasks.candidates[e],cells,reached)
            self.arrived[e]=np.all(pos==self.goals[e],axis=1)
            border=env.grid_config.obs_radius
            env.grid.finishes_xy=[tuple(map(int,g+border)) for g in self.goals[e]]
            self._refresh(e,reached)
            if reached.any():
                relative=env.grid.get_targets_xy_relative()
                for i in np.flatnonzero(reached):
                    observations[i]['global_target_xy']=env.grid.finishes_xy[i]
                    observations[i]['target_xy']=relative[i]
            self.observations[e]=observations
        self.steps+=1
        if self.steps%128==0:self.goal_bins.append(int(self.cursors.sum()))
        self.step_seconds+=time.perf_counter()-started

    def load_rows(self,rows):
        """Restore a recorded configuration in persistent envs, keeping valid caches."""
        rows=np.asarray(rows).reshape(len(self.envs),self.goals.shape[1],8)
        for e,(env,row) in enumerate(zip(self.envs,rows)):
            border=env.grid_config.obs_radius
            changed=np.any(self.goals[e]!=row[:,4:6],axis=1);self.goals[e]=row[:,4:6]
            env.grid.positions[:]=0
            env.grid.positions_xy=[tuple(map(int,p+border)) for p in row[:,2:4]]
            env.grid.finishes_xy=[tuple(map(int,g+border)) for g in self.goals[e]]
            for p in env.grid.positions_xy:env.grid.positions[p]=1
            self._refresh(e,changed);self.observations[e]=env._obs()

    def result(self):
        return dict(positions=self.positions(),goals=self.goals,cursors=self.cursors,arrived=self.arrived,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            cached_goals=sum(len(c.dist_tables) for c in self.calculators),goal_bins=self.goal_bins,
            build_seconds=self.build_seconds,step_seconds=self.step_seconds,refreshed_goals=self.refreshed_goals)
