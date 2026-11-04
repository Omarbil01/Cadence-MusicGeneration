"""Standard MIDI 0/1 reader and writer using only Python's standard library.

Time is quarter-note beats. Program 128 represents channel-10 percussion.
Original channel/track, velocity, tempo and meter metadata are retained on read.
"""
from __future__ import annotations
from collections import defaultdict, deque, Counter
from dataclasses import dataclass, asdict
from pathlib import Path
import math
import struct


@dataclass(frozen=True)
class Note:
    start: float
    end: float
    pitch: int
    program: int
    velocity: int = 80
    channel: int = 0
    track: int = 0


def _vlq(data, pos, limit):
    value = 0
    for _ in range(4):
        if pos >= limit:
            raise ValueError('Truncated MIDI variable-length quantity')
        b = data[pos]; pos += 1
        value = (value << 7) | (b & 127)
        if b < 128:
            return value, pos
    raise ValueError('MIDI variable-length quantity exceeds four bytes')


def _pack_vlq(value):
    value = int(value)
    if not 0 <= value <= 0x0fffffff:
        raise ValueError('Invalid MIDI delta time')
    result = [value & 127]; value >>= 7
    while value:
        result.append((value & 127) | 128); value >>= 7
    return bytes(reversed(result))


def read_midi(path, *, sustain=True):
    data = Path(path).read_bytes()
    if len(data) < 14 or data[:4] != b'MThd':
        raise ValueError('Not a Standard MIDI file')
    header_len = struct.unpack('>I', data[4:8])[0]
    fmt, ntracks, ppq = struct.unpack('>HHH', data[8:14])
    if fmt not in (0, 1) or ppq == 0 or ppq & 0x8000:
        raise ValueError('Require synchronous format-0/1 MIDI with PPQ timing')
    if fmt == 0 and ntracks != 1:
        raise ValueError('Format-0 MIDI must have one track')
    pos = 8 + header_len
    events = []; tempos = [(0., 500000)]; meters = [(0., 4, 4)]; last_tick = 0
    for track in range(ntracks):
        if data[pos:pos+4] != b'MTrk' or pos+8 > len(data):
            raise ValueError('Missing MIDI track')
        size = struct.unpack('>I', data[pos+4:pos+8])[0]
        at = pos+8; limit = at+size
        if limit > len(data):
            raise ValueError('Truncated MIDI track')
        tick = 0; running = None; index = 0
        while at < limit:
            delta, at = _vlq(data, at, limit); tick += delta
            if at >= limit:
                raise ValueError('Missing MIDI event')
            status = data[at]
            if status >= 128:
                at += 1
            elif running is not None:
                status = running
            else:
                raise ValueError('MIDI running status has no prior channel message')
            if status == 255:
                if at >= limit:
                    raise ValueError('Truncated meta event')
                kind = data[at]; at += 1
                count, at = _vlq(data, at, limit)
                payload = data[at:at+count]; at += count
                if at > limit:
                    raise ValueError('Truncated meta payload')
                if kind == 81 and len(payload) == 3:
                    us = int.from_bytes(payload, 'big')
                    if us > 0: tempos.append((tick/ppq, us))
                elif kind == 88 and len(payload) >= 2:
                    meters.append((tick/ppq, payload[0], 2**payload[1]))
                # Running channel status is retained across meta events in SMF.
            elif status in (240, 247):
                count, at = _vlq(data, at, limit); at += count; running = None
                if at > limit: raise ValueError('Truncated SysEx event')
            elif 128 <= status < 240:
                running = status; kind = status >> 4; channel = status & 15
                count = 1 if kind in (12, 13) else 2
                payload = data[at:at+count]; at += count
                if len(payload) != count or any(x > 127 for x in payload):
                    raise ValueError('Invalid MIDI channel payload')
                events.append((tick, track, index, kind, channel, tuple(payload)))
            else:
                raise ValueError(f'Unsupported MIDI status {status}')
            index += 1
        last_tick = max(last_tick, tick); pos = limit
    if pos != len(data):
        raise ValueError('Unexpected bytes after MIDI tracks')
    # Later entries at the same beat replace defaults / earlier metadata.
    tempos = sorted(dict(tempos).items())
    meter_map = {beat: (n, d) for beat, n, d in meters}
    meters = [(beat, *nd) for beat, nd in sorted(meter_map.items())]
    programs = defaultdict(int); pedals = defaultdict(bool)
    opened = defaultdict(deque); released = defaultdict(list)
    notes = []; unmatched_off = 0; pedal_events = 0
    note_on_events = 0; note_off_events = 0; channel_mode_closures = 0
    controller_closed = Counter(); channel_mode_redundant_offs = 0

    def finish(start_record, end_tick):
        start, pitch, program, velocity, channel, track = start_record
        if end_tick > start:
            notes.append(Note(start/ppq, end_tick/ppq, pitch, program, velocity, channel, track))

    for tick, track, index, kind, channel, payload in sorted(events):
        if kind == 12:
            programs[channel] = payload[0]
        elif kind == 11 and payload[0] == 64:
            pedal_events += 1
            down = payload[1] >= 64
            if sustain and pedals[channel] and not down:
                for rec in released[channel]: finish(rec, tick)
                released[channel].clear()
            pedals[channel] = down
        elif kind == 11 and payload[0] in (120, 121, 123):
            controller = payload[0]
            # MIDI channel-mode messages are valid note-ending events.
            # CC123 releases keys and respects sustain; CC120 silences immediately.
            # CC121 resets sustain, but must not end keys that are still held down.
            if controller in (120, 123):
                for (ch, pitch), queue in opened.items():
                    if ch != channel: continue
                    while queue:
                        rec = queue.popleft(); channel_mode_closures += 1
                        controller_closed[ch,pitch] += 1
                        if controller == 123 and sustain and pedals[channel] and channel != 9:
                            released[channel].append(rec)
                        else:
                            finish(rec, tick)
            if controller in (120, 121):
                for rec in released[channel]: finish(rec, tick)
                released[channel].clear()
            if controller == 121: pedals[channel] = False
        elif kind == 9 and payload[1] > 0:
            note_on_events += 1
            pitch, velocity = payload
            # A reattack ends a deferred same-pitch sustain, preserving the attack.
            if sustain:
                keep = []
                for rec in released[channel]:
                    if rec[1] == pitch: finish(rec, tick)
                    else: keep.append(rec)
                released[channel] = keep
            program = 128 if channel == 9 else programs[channel]
            opened[channel, pitch].append((tick, pitch, program, velocity, channel, track))
        elif kind == 8 or (kind == 9 and payload[1] == 0):
            note_off_events += 1
            pitch = payload[0]
            if not opened[channel, pitch]:
                # A key may send its ordinary note-off after a channel-mode
                # message already ended it. Consume only a known closure.
                if controller_closed[channel,pitch]:
                    controller_closed[channel,pitch] -= 1
                    channel_mode_redundant_offs += 1
                else:
                    unmatched_off += 1
                continue
            rec = opened[channel, pitch].popleft()
            if sustain and pedals[channel] and channel != 9:
                released[channel].append(rec)
            else: finish(rec, tick)
    for pending in released.values():
        for rec in pending: finish(rec, last_tick)
    dangling = sum(len(q) for q in opened.values())
    event_count = note_on_events + note_off_events
    unmatched_ratio = (unmatched_off + dangling)/max(1, event_count)
    end_beat = last_tick/ppq
    notes.sort(key=lambda n: (n.start, n.program, n.pitch, n.end))
    return notes, {'format': fmt, 'tracks': ntracks, 'ppq': ppq,
        'duration_beats': end_beat, 'tempos': tempos, 'meters': meters,
        'duration_seconds': beat_to_seconds(end_beat, tempos),
        'unmatched_off': unmatched_off, 'dangling_on': dangling,
        'note_on_events': note_on_events, 'note_off_events': note_off_events,
        'note_event_count': event_count, 'unmatched_event_ratio': unmatched_ratio,
        'channel_mode_note_closures': channel_mode_closures,
        'channel_mode_redundant_note_offs': channel_mode_redundant_offs,
        'pedal_events': pedal_events, 'sustain_applied': sustain,
        'programs': sorted({n.program for n in notes})}


def beat_to_seconds(beat, tempos):
    total = 0.; cursor = 0.; tempo = 500000
    for at, us in tempos:
        if at > beat: break
        total += max(0., at-cursor)*tempo/1e6
        cursor = at; tempo = us
    return total + max(0., beat-cursor)*tempo/1e6


def bpm_at(beat, tempos):
    tempo = 500000
    for at, us in tempos:
        if at > beat: break
        tempo = us
    return 60000000/tempo


def activity(notes, start=0., end=None):
    end = max((n.end for n in notes), default=start) if end is None else end
    spans = sorted((max(start,n.start),min(end,n.end)) for n in notes
                   if n.end > start and n.start < end)
    cursor = start; active = 0.; gaps = []
    for a,b in spans:
        if a > cursor: gaps.append((cursor,a))
        if b > cursor: active += b-max(cursor,a); cursor = b
    if cursor < end: gaps.append((cursor,end))
    onsets = sorted({n.start for n in notes if start <= n.start < end})
    onset_gaps = [b-a for a,b in zip([start]+onsets,onsets+[end])]
    duration = max(0.,end-start)
    return {'notes':len(spans),'active_fraction':active/duration if duration else 0.,
        'silence_beats':sum(b-a for a,b in gaps),
        'longest_silence_beats':max((b-a for a,b in gaps),default=0.),
        'longest_onset_gap_beats':max(onset_gaps,default=duration),
        'onsets_per_beat':len(onsets)/duration if duration else 0.,
        'gaps':[{'start_beat':a,'end_beat':b} for a,b in gaps]}


def write_midi(path, notes, *, bpm=120., meter=(4,4), end_beat=None):
    if not 20 <= bpm <= 300: raise ValueError('BPM outside 20..300')
    programs = sorted({n.program for n in notes if n.program != 128})
    if len(programs) > 15: raise ValueError('MIDI output has more than 15 melodic programs')
    channels = dict(zip(programs,[c for c in range(16) if c != 9]))
    ppq = 480
    end = max((n.end for n in notes), default=0.) if end_beat is None else end_beat
    if any(n.end > end+1e-8 or n.start < 0 or n.end <= n.start for n in notes):
        raise ValueError('Invalid output note interval')
    def track_blob(events):
        prior = 0; content = bytearray()
        for tick, priority, payload in sorted(events,key=lambda e:(e[0],e[1],e[2])):
            content += _pack_vlq(tick-prior)+payload; prior = tick
        final = max(prior,round(end*ppq))
        content += _pack_vlq(final-prior)+b'\xff\x2f\x00'
        return b'MTrk'+struct.pack('>I',len(content))+content
    numerator, denominator = meter
    if denominator <= 0 or denominator & (denominator-1): raise ValueError('Invalid meter denominator')
    tempo = round(60000000/bpm)
    conductor = [(0,0,b'\xff\x51\x03'+tempo.to_bytes(3,'big')),
                 (0,1,b'\xff\x58\x04'+bytes([numerator,int(math.log2(denominator)),24,8]))]
    tracks = [track_blob(conductor)]
    for program in sorted({n.program for n in notes}):
        ch = 9 if program == 128 else channels[program]
        name = ('Drums' if program == 128 else f'Program {program}').encode()
        events = [(0,-3,b'\xff\x03'+_pack_vlq(len(name))+name)]
        if program != 128: events.append((0,-2,bytes([0xc0|ch,program])))
        for n in notes:
            if n.program != program: continue
            if not 0 <= n.pitch <= 127: raise ValueError('Invalid MIDI pitch')
            events.extend([(round(n.start*ppq),1,bytes([0x90|ch,n.pitch,max(1,min(127,n.velocity))])),
                           (round(n.end*ppq),0,bytes([0x80|ch,n.pitch,0]))])
        tracks.append(track_blob(events))
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(b'MThd'+struct.pack('>IHHH',6,1,len(tracks),ppq)+b''.join(tracks))
    return {'tracks':len(tracks),'programs':sorted({n.program for n in notes}),'notes':len(notes)}
