"""Standalone figure: time, identical learning trajectory, identical task quality."""
import runtime
import argparse
from pathlib import Path
import json
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--base',type=Path,default=runtime.ART/'controlled-parity')
    parser.add_argument('--desktop',type=Path,default=runtime.DESKTOP/'identical-learning-controlled-comparison')
    parser.add_argument('--training-envs',type=int,default=16)
    parser.add_argument('--updates',type=int,default=64)
    parser.add_argument('--loss-window',type=int,default=1)
    parser.add_argument('--pretrained-reference',action='store_true')
    parser.add_argument('--previous-reference',action='store_true')
    args=parser.parse_args()
    base=args.base
    result=json.loads((base/'summary.json').read_text())
    rows=result['rows']
    colors={'cpu':'#476A93','jointloom':'#DC8553'}
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(1,3,figsize=(13.2,3.8))
    ax=axes[0];positions=np.arange(2);width=.32
    for i,backend in enumerate(('cpu','jointloom')):
        values=np.asarray([[r[f'{backend}_online_seconds'],r[f'{backend}_process_seconds']] for r in rows])
        offset=(i-.5)*width
        bars=ax.bar(positions+offset,values.mean(0),width,color=colors[backend],
            label='CPU inputs' if backend=='cpu' else 'JointLoom',
            yerr=values.std(0,ddof=1),capsize=3,alpha=.9)
        for x,y in zip(positions+offset,values.mean(0)):
            ax.text(x,y+1.5,f'{y:.1f}',ha='center',fontsize=9)
        for j in range(2):ax.scatter(np.full(3,positions[j]+offset),values[:,j],s=12,color='black',alpha=.6,zorder=4)
    ax.set_xticks(positions,['Online loop','Full process']);ax.set_ylabel('Wall-clock time (s)')
    ax.set_title('(a) Matched online work',loc='left',weight='bold')
    ax.legend(frameon=False,fontsize=8);ax.set_ylim(0,max(r['cpu_process_seconds'] for r in rows)*1.25)
    ax=axes[1]
    losses={backend:np.asarray([[
        v['loss'] for v in json.loads((base/f'timing-{backend}-n{args.training_envs}-s{seed}'/'update-trace.json').read_text())]
        for seed in (0,1,2)]) for backend in ('cpu','jointloom')}
    for backend in losses:
        assert losses[backend].shape==(3,args.updates)
    assert np.array_equal(losses['cpu'],losses['jointloom'])
    if args.loss_window>1:
        losses={key:value.reshape(3,-1,args.loss_window).mean(2) for key,value in losses.items()}
    x=np.arange(args.loss_window,args.updates+1,args.loss_window)
    average=losses['cpu'].mean(0);sd=losses['cpu'].std(0,ddof=1)
    ax.fill_between(x,average-sd,average+sd,color=colors['cpu'],alpha=.12,linewidth=0)
    ax.plot(x,average,color=colors['cpu'],lw=2,label='CPU inputs')
    ax.plot(x,losses['jointloom'].mean(0),color=colors['jointloom'],lw=1.8,ls='--',label='JointLoom')
    ax.set_xlabel('Optimizer updates')
    ax.set_ylabel('Cross-entropy (3-seed mean)' if args.loss_window==1 else f'Cross-entropy ({args.loss_window}-update mean)')
    ax.set_title('(b) Identical learning losses',loc='left',weight='bold')
    ax.text(.03,.97,'Final model + Adam states: identical',transform=ax.transAxes,va='top',fontsize=8)
    ax.legend(frameon=False,fontsize=8,loc='lower left');ax.set_xlim(1,args.updates)
    ax=axes[2]
    quality=result['evaluation']['summaries']
    for backend,marker in [('cpu','o'),('jointloom','x')]:
        y=[next(v['success_rate'] for v in quality if v['checkpoint']==f'timing-{backend}-n{args.training_envs}-s{seed}')*100
            for seed in (0,1,2)]
        ax.plot([0,1,2],y,marker=marker,color=colors[backend],lw=1.3,ms=8,
            ls='-' if backend=='cpu' else '--',label='CPU inputs' if backend=='cpu' else 'JointLoom')
    if args.pretrained_reference:
        reference=next(v['success_rate'] for v in quality if v['checkpoint']=='official-pretrained')*100
        ax.axhline(reference,color='#777777',ls=':',lw=1.3,label=f'Pretrained ({reference:.2f}%)')
    if args.previous_reference:
        previous=[next(v['success_rate'] for v in quality
            if v['checkpoint']==f'reference256-cpu-n{args.training_envs}-s{seed}')*100 for seed in (0,1,2)]
        ax.plot([0,1,2],previous,color='#888888',marker='^',ls=':',lw=1.1,ms=5,
            label='Previous 256-step')
    ax.set_xticks([0,1,2]);ax.set_xlabel('Paired training seed');ax.set_ylabel('MAPF success (%)')
    ax.set_ylim(0,105);ax.set_title('(c) Identical heldout quality',loc='left',weight='bold')
    ax.text(.03,.50 if args.previous_reference else .05,
        f'{result["evaluation"]["tasks"]} fresh tasks per model\nSoC and makespan also identical',
        transform=ax.transAxes,fontsize=8)
    ax.legend(frameon=False,fontsize=8,loc='upper right' if max(y)<50 else 'lower right')
    fig.tight_layout(pad=1.5,w_pad=2.2)
    for suffix in ('png','pdf'):
        path=base/f'controlled-parity.{suffix}'
        fig.savefig(path,dpi=220,bbox_inches='tight')
        shutil.copy2(path,args.desktop/f'identical-learning-faster-execution.{suffix}')
    plt.close(fig)


if __name__=='__main__':main()
