"""Pure-Python musical event grammar, with limits checked before time advances."""
from collections import Counter,defaultdict
from .tokenizer import TO_ID,TOKENS,GRID,BOS,BAR,EOS,TEMPOS
from .midi import Note


class NoteGrammar:
    def __init__(self,*,target_beats,bpm=120,meter=(4,4),programs=(0,),
                 drum_pitches=(),max_gap_beats=2.,max_attacks_at_position=12):
        self.target=max(1,round(target_beats*GRID)); self.meter=tuple(meter)
        self.bar_steps=round(meter[0]*4/meter[1]*GRID)
        if not 1<=self.bar_steps<=96: raise ValueError('Unsupported meter')
        self.programs=tuple(sorted(set(programs)))
        if not self.programs or not all(0<=p<=128 for p in self.programs): raise ValueError('No valid training instruments')
        self.drum_pitches=tuple(sorted(set(drum_pitches)))
        if 128 in self.programs and not self.drum_pitches: raise ValueError('Missing training drum pitch information')
        if max_gap_beats<.25: raise ValueError('Maximum onset gap must be at least one grid step')
        self.gap=max(1,round(max_gap_beats*GRID)); self.attack_limit=max_attacks_at_position
        self.bar=0; self.position=0; self.last_onset=None; self.at_position=0
        self.phase='boundary'; self.pending={}; self.notes=[]; self.done=False
        tempo=min(TEMPOS,key=lambda t:abs(t-bpm))
        self.ids=[BOS,TO_ID[f'BPM_{tempo}'],TO_ID[f'METER_{meter[0]}_{meter[1]}'],BAR]

    def allowed(self):
        if self.done:return []
        if self.phase=='boundary':
            base=self.bar*self.bar_steps
            earliest=self.position+(1 if self.at_position>=self.attack_limit else 0)
            ceiling=(self.last_onset if self.last_onset is not None else 0)+self.gap
            latest=min(self.bar_steps-1,self.target-1-base,ceiling-base)
            choices=[TO_ID[f'POS_{pos}'] for pos in range(max(0,earliest),latest+1)]
            next_bar=(self.bar+1)*self.bar_steps
            if next_bar<self.target and next_bar<=ceiling: choices.append(BAR)
            if self.bar>=(self.target-1)//self.bar_steps and self.last_onset is not None and self.target-self.last_onset<=self.gap:
                choices.append(EOS)
            if not choices:raise ValueError('No feasible next event; grammar state is invalid')
            return choices
        if self.phase=='program':return [TO_ID[f'PROGRAM_{n}'] for n in self.programs]
        if self.phase=='pitch':
            pr=self.pending['program']
            if pr==128:pitches=self.drum_pitches
            else:
                low,high=(28,72) if 32<=pr<=39 else ((40,96) if 24<=pr<=31 else (21,108))
                pitches=range(low,high+1)
            return [TO_ID[f'PITCH_{n}'] for n in pitches]
        if self.phase=='duration':
            limit=min(128,self.target-self.pending['start'])
            if self.pending['program']==128:limit=min(limit,4)
            return [TO_ID[f'DUR_{n}'] for n in range(1,limit+1)]
        if self.phase=='velocity':return [TO_ID[f'VEL_{n}'] for n in range(4,17)]
        raise ValueError('Unknown grammar phase')

    def accept(self,choice):
        if choice not in self.allowed():raise ValueError('Illegal generated token: '+str(choice))
        self.ids.append(choice); token=TOKENS[choice]
        if self.phase=='boundary':
            if choice==EOS:self.done=True
            elif choice==BAR:self.bar+=1;self.position=0;self.at_position=0
            else:
                pos=int(token[4:])
                if pos!=self.position:self.at_position=0
                self.position=pos;self.pending={'start':self.bar*self.bar_steps+pos};self.phase='program'
        elif self.phase=='program':self.pending['program']=int(token[8:]);self.phase='pitch'
        elif self.phase=='pitch':self.pending['pitch']=int(token[6:]);self.phase='duration'
        elif self.phase=='duration':self.pending['duration']=int(token[4:]);self.phase='velocity'
        elif self.phase=='velocity':
            n=self.pending
            self.notes.append(Note(n['start']/GRID,(n['start']+n['duration'])/GRID,
                                   n['pitch'],n['program'],min(127,int(token[4:])*8)))
            self.last_onset=n['start'];self.at_position+=1;self.phase='boundary'


def resolve_render_overlaps(notes):
    """A fresh same-pitch attack closes the prior note; adjacent attacks survive."""
    from dataclasses import replace
    voices=defaultdict(list); result=[]; dropped=0
    for n in notes:voices[n.program,n.pitch].append(n)
    for seq in voices.values():
        kept=[]
        for n in sorted(seq,key=lambda n:(n.start,n.end,n.velocity)):
            if kept and n.start==kept[-1].start:
                kept[-1]=n;dropped+=1;continue
            if kept and n.start<kept[-1].end:kept[-1]=replace(kept[-1],end=n.start)
            kept.append(n)
        result.extend(kept)
    return sorted(result,key=lambda n:(n.start,n.program,n.pitch)),dropped
