"""Real Sample Factory entry; only official environment/model/observer extensions."""
import bootstrap
import json
import os
import sys
import time
from multiprocessing.util import Finalize
from pathlib import Path
os.environ.setdefault('WANDB_MODE','disabled')
os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD','1')
import numpy as np
import torch
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
import gymnasium as gym
from sample_factory.algo.utils.context import global_model_factory
from sample_factory.cfg.arguments import parse_sf_args,parse_full_cfg
from sample_factory.envs.env_utils import register_env
from sample_factory.model.encoder import Encoder
from sample_factory.model.actor_critic import default_make_actor_critic_func
from sample_factory.train import make_runner
from sample_factory.algo.runners.runner import AlgoObserver
from sample_factory.algo.utils.misc import ExperimentStatus
from model import CostTracer,configure_torch
from environments import CpuEnvironment
from sf_tasks import SF_ROOT,prepare,load


class CostTracerPogema(gym.Env):
    """128 agents per SF environment, with the validated Pogema CostTracer view/reward."""
    is_multiagent=True
    num_agents=128
    metadata={}

    def __init__(self,cfg,env_config=None,render_mode=None):
        self.seed_value=int(cfg.seed or 0)
        self.env_id=int(env_config.get('env_id',0)) if env_config else 0
        self.tasks=load(self.seed_value,self.env_id,num_envs=cfg.num_workers*cfg.num_envs_per_worker,
            bank_episodes=getattr(cfg,'costtracer_task_bank_episodes',64))
        self.inner=None
        self.observation_space=gym.spaces.Box(-1.,1.,shape=(2,11,11),dtype=np.float32)
        self.action_space=gym.spaces.Discrete(5)
        self.goals=np.zeros(128,np.float32)
        # SF's SequentialVectorizeWrapper concatenates per-agent info sequences when
        # a split contains multiple environments. Reuse empty infos on ordinary steps.
        self.empty_infos=tuple({} for _ in range(128))
        self.reward_log=bool(getattr(cfg,'costtracer_reward_log',False))
        self.progress_log=bool(getattr(cfg,'costtracer_progress_log',False))
        if self.reward_log:
            self.reward_folder=Path(cfg.costtracer_observer_dir)
            self.reward_records=[];self.reward_steps=0
            self.block_reward=0.;self.block_goals=0.;self.episode_reward=0.;self.episode_goals=0.
            self.reward_episodes=0;self.reward_flushed=False
            Finalize(None,self.flush_rewards,exitpriority=10)

    def reset(self,*,seed=None,options=None):
        self.inner=CpuEnvironment(self.tasks)
        self.goals.fill(0)
        return self.inner.observe()[0],self.empty_infos

    def step(self,actions):
        action=np.asarray(actions,dtype=np.int64).reshape(1,128)
        obs,reward,hit,trunc=self.inner.step(action)
        if self.reward_log:
            r=float(np.sum(reward[0],dtype=np.float64));g=float(np.sum(hit[0],dtype=np.float64))
            self.reward_steps+=1;self.block_reward+=r;self.block_goals+=g
            self.episode_reward+=r;self.episode_goals+=g
            if self.reward_steps%32==0:
                self.reward_records.append(dict(kind='rollout',env_id=self.env_id,
                    env_joint_steps=self.reward_steps,agent_samples=32*128,
                    reward_sum=self.block_reward,mean_progress_reward=self.block_reward/(32*128),
                    goals=self.block_goals,completed_perf_counter_ns=time.perf_counter_ns()))
                self.block_reward=0.;self.block_goals=0.
            if trunc[0]:
                self.reward_episodes+=1
                self.reward_records.append(dict(kind='episode',env_id=self.env_id,
                    episode=self.reward_episodes,env_joint_steps=self.reward_steps,
                    mean_agent_episode_return=self.episode_reward/128,
                    goals=self.episode_goals,completed_perf_counter_ns=time.perf_counter_ns()))
                if self.progress_log:
                    self.reward_folder.mkdir(parents=True,exist_ok=True)
                    path=self.reward_folder/f'reward-progress-env-{self.env_id}.json'
                    tmp=path.with_suffix('.tmp')
                    tmp.write_text(json.dumps(self.reward_records[-1]));tmp.replace(path)
                self.episode_reward=0.;self.episode_goals=0.
        self.goals+=hit[0]
        terminal=np.zeros(128,dtype=bool);truncated=np.full(128,trunc[0],dtype=bool)
        infos=self.empty_infos
        if trunc[0]:
            infos=[{'costtracer_goals':float(g)} for g in self.goals]
            obs=self.inner.reset_due();self.goals.fill(0)
        return obs[0],reward[0],terminal,truncated,infos

    def render(self):return None
    def flush_rewards(self):
        if not self.reward_log or self.reward_flushed:return
        self.reward_folder.mkdir(parents=True,exist_ok=True)
        (self.reward_folder/f'reward-env-{self.env_id}-pid-{os.getpid()}.json').write_text(json.dumps(dict(
            env_id=self.env_id,agents=128,total_joint_steps=self.reward_steps,records=self.reward_records)))
        self.reward_flushed=True

    def close(self):
        if self.reward_log:self.flush_rewards()


def make_env(full_env_name,cfg=None,env_config=None,render_mode=None):
    return CostTracerPogema(cfg,env_config,render_mode)


class CostTracerEncoder(Encoder):
    def __init__(self,cfg,obs_space):
        super().__init__(cfg)
        # SF supplies its native shared actor/critic, identity core/decoder and action distribution.
        canonical=CostTracer()
        self.conv_head=canonical.encoder.conv_head
        self.extra_linear=canonical.encoder.extra_linear

    def get_out_size(self):return 32
    def forward(self,obs_dict):
        return self.extra_linear(self.conv_head(obs_dict['obs']).flatten(1))


def make_encoder(cfg,obs_space):return CostTracerEncoder(cfg,obs_space)


def make_actor_critic(cfg,obs_space,action_space):
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    model=default_make_actor_critic_func(cfg,obs_space,action_space)
    initial=torch.load(Path(cfg.costtracer_initial_weights),map_location='cpu',weights_only=True)
    model.load_state_dict(initial,strict=True)
    if os.environ.get('JOINTLOOM_NEURAL_PROFILE_DIR'):
        from neural_timing import instrument_native_model
        instrument_native_model(model)
    folder=Path(cfg.costtracer_observer_dir)
    if folder.exists():
        (folder/f'model-origin-{os.getpid()}.json').write_text(json.dumps(dict(
            actor_critic_class=type(model).__module__+'.'+type(model).__name__,
            parameter_count=sum(p.numel() for p in model.parameters()),
            cudnn_benchmark=torch.backends.cudnn.benchmark,
            cudnn_tf32=torch.backends.cudnn.allow_tf32,matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
            canonical_state_keys=list(initial)),indent=2))
    return model


class TimingObserver(AlgoObserver):
    def __init__(self,folder,iterations):
        self.folder=Path(folder);self.iterations=iterations;self.previous=None;self.rows=[]

    def on_start(self,runner):
        self.samples_per_iteration=runner.cfg.batch_size*runner.cfg.num_batches_per_epoch
        self.previous=time.perf_counter()
        if getattr(runner.cfg,'costtracer_reward_log',False):
            (self.folder/'training-clock.json').write_text(json.dumps(dict(
                start_perf_counter_ns=time.perf_counter_ns(),
                timing='native runner start; excludes process/model initialization; includes training collection/update/control')))

    def on_training_step(self,runner,training_iteration_since_resume):
        now=time.perf_counter();iteration=int(training_iteration_since_resume)
        row=dict(iteration=iteration,interval_seconds=now-self.previous,
            agent_samples=iteration*self.samples_per_iteration,reported_env_steps=dict(runner.env_steps),
            timestamp_perf_counter=now)
        deadline=getattr(runner.cfg,'costtracer_stop_perf_ns',0)
        stopfile=getattr(runner.cfg,'costtracer_stop_file','')
        timed_out=bool(deadline and time.perf_counter_ns()>=deadline)
        requested=bool(stopfile and Path(stopfile).exists())
        row.update(wall_deadline_reached=timed_out,stop_requested=requested)
        self.previous=now;self.rows.append(row)
        with (self.folder/'native-iterations.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        # Official observer controls the bounded experiment; no learner/sampler implementation is replaced.
        if iteration>=self.iterations or timed_out or requested:runner._stop_training()

    def on_stop(self,runner):
        measured=self.rows[2:]
        data=dict(callbacks=len(self.rows),rows=self.rows,
            steady_iteration_seconds=float(np.mean([r['interval_seconds'] for r in measured])) if measured else None,
            native_framework='sample_factory 2.1.1',async_rl=runner.cfg.async_rl,
            batched_sampling=runner.cfg.batched_sampling,env_steps_reported=dict(runner.env_steps),
            samples_per_iteration=self.samples_per_iteration,
            num_envs=runner.cfg.num_workers*runner.cfg.num_envs_per_worker,
            cpu_affinity=sorted(os.sched_getaffinity(0)))
        (self.folder/'native-timing.json').write_text(json.dumps(data,indent=2))


def register():
    register_env('jointloom_costtracer_pogema',make_env)
    global_model_factory().register_encoder_factory(make_encoder)
    global_model_factory().register_actor_critic_factory(make_actor_critic)


def configuration(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    help_requested = '--help' in argv or '-h' in argv
    initial = [arg for arg in argv if arg not in ('--help', '-h')]
    if help_requested and not any(arg == '--env' or arg.startswith('--env=') for arg in initial):
        initial.append('--env=jointloom_costtracer_pogema')
    parser,_=parse_sf_args(initial if help_requested else argv)
    parser.add_argument('--costtracer_initial_weights',type=str,required=True)
    parser.add_argument('--costtracer_observer_dir',type=str,required=True)
    parser.add_argument('--costtracer_iterations',type=int,default=16)
    parser.add_argument('--costtracer_reward_log',action='store_true')
    parser.add_argument('--costtracer_task_bank_episodes',type=int,default=64)
    parser.add_argument('--costtracer_stop_perf_ns',type=int,default=0)
    parser.add_argument('--costtracer_stop_file',type=str,default='')
    parser.add_argument('--costtracer_progress_log',action='store_true')
    parser.set_defaults(env='jointloom_costtracer_pogema',device='gpu',
        num_workers=8,num_envs_per_worker=2,worker_num_splits=2,policy_workers_per_policy=1,
        batched_sampling=True,async_rl=False,serial_mode=False,
        set_workers_cpu_affinity=False,
        use_rnn=False,recurrence=1,rollout=32,batch_size=2048,num_batches_per_epoch=32,num_epochs=1,
        actor_critic_share_weights=True,decoder_mlp_layers=[],policy_initialization='orthogonal',policy_init_gain=1.,
        normalize_input=False,normalize_returns=False,obs_subtract_mean=0.,obs_scale=1.,
        gamma=.96,gae_lambda=.95,learning_rate=.00013,lr_schedule='constant',
        ppo_clip_ratio=.2,ppo_clip_value=1.,value_loss_coeff=.5,exploration_loss='entropy',
        exploration_loss_coeff=.06878,kl_loss_coeff=0.,max_grad_norm=4.,optimizer='adam',
        adam_beta1=.9,adam_beta2=.999,adam_eps=1e-6,with_vtrace=False,value_bootstrap=True,
        reward_scale=1.,reward_clip=1000.,with_wandb=False,decorrelate_experience_max_seconds=0,
        decorrelate_envs_on_one_worker=False,save_every_sec=1000000,save_best_every_sec=1000000,
        train_for_seconds=600,train_for_env_steps=1048575)
    if help_requested:
        parser.print_help()
        raise SystemExit(0)
    return parse_full_cfg(parser,argv)


def main(argv=None):
    os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:16]))
    register();cfg=configuration(argv);configure_torch(int(cfg.seed or 0))
    folder=Path(cfg.costtracer_observer_dir);folder.mkdir(parents=True,exist_ok=True)
    native=__import__('sample_factory')
    (folder/'framework-origin.json').write_text(json.dumps(dict(
        sample_factory_file=native.__file__,torch=torch.__version__,numpy=np.__version__,
        network_parameters=161734,framework_source_modified=False,
        customizations=['official environment factory','official encoder/model factories','official AlgoObserver']),indent=2))
    cfg,runner=make_runner(cfg)
    runner.register_observer(TimingObserver(folder,cfg.costtracer_iterations))
    status=runner.init()
    if status==ExperimentStatus.SUCCESS:status=runner.run()
    return int(status)


if __name__=='__main__':sys.exit(main())
