"""Match inference prompts to complete-event training chunk framing."""
from .tokenizer import TOKENS,BOS,BAR


def framed_context(ids, context, *, budget=None, return_positions=False):
    if context<16:raise ValueError('Context must be at least 16')
    if len(ids)<4 or ids[0]!=BOS or not TOKENS[ids[1]].startswith('BPM_') or not TOKENS[ids[2]].startswith('METER_') or ids[3]!=BAR:
        raise ValueError('Expected BOS/BPM/METER/BAR generation prefix')
    if len(ids)<=context:
        result=list(ids)
        return (result,list(range(len(ids)))) if return_positions else result
    budget=context if budget is None else int(budget)
    if not 16<=budget<=context:raise ValueError('Invalid framed context budget')
    prefix=list(ids[:3])
    earliest=max(3,len(ids)-(budget-3))
    for start in range(earliest,len(ids)):
        token=ids[start]
        if token==BAR:header=prefix
        elif TOKENS[token].startswith('POS_'):header=prefix+[BAR]
        else:continue
        if len(header)+len(ids)-start<=budget:
            result=header+list(ids[start:])
            positions=[None]*len(header)+list(range(start,len(ids)))
            return (result,positions) if return_positions else result
    raise ValueError('No complete event boundary fits the model context')


def model_prompt(ids, context, mode):
    if mode=='legacy':return list(ids[-context:])
    if mode=='framed':return framed_context(ids,context)
    if mode=='framed_75':return framed_context(ids,context,budget=max(16,int(context*.75)))
    raise ValueError('Unknown context mode: '+str(mode))


def instrument_palette(counts, palette='original', max_programs=8):
    counts={int(k):int(v) for k,v in counts.items()}
    ranked=sorted(counts,key=lambda k:(-counts[k],k))
    if palette=='original':return ranked[:max_programs]
    if palette!='band':raise ValueError('Unknown instrument palette')
    groups={'guitar':[p for p in ranked if 24<=p<=31],
            'bass':[p for p in ranked if 32<=p<=39],
            'drums':[p for p in ranked if p==128]}
    missing=[name for name,values in groups.items() if not values]
    if missing:raise ValueError('Training data lacks band roles: '+', '.join(missing))
    return [groups[role][0] for role in ('guitar','bass','drums')]
