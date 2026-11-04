"""Preserve internal silence gaps on the continuous cropped audio timeline."""
import numpy as np


def alignment_to_phones(tier, sampling_rate, hop_length):
    intervals = list(tier._objects)
    silence = {'', 'sil', 'sp', 'spn'}
    speech = [i for i, item in enumerate(intervals) if item.text.strip() not in silence]
    if not speech:
        return [], [], 0.0, 0.0, {'status': 'no_speech'}
    intervals = intervals[speech[0]:speech[-1]+1]
    start, end = intervals[0].start_time, intervals[-1].end_time
    phones, spans = [], []
    cursor = start
    implicit_seconds = explicit_seconds = 0.0

    def append(phone, a, b):
        if b <= a:
            return
        if phones and phone == 'sp' and phones[-1] == 'sp':
            spans[-1] = (spans[-1][0], b)
        else:
            phones.append(phone)
            spans.append((a, b))

    for item in intervals:
        a, b = float(item.start_time), float(item.end_time)
        if a < cursor - 1e-7 or b < a:
            raise ValueError('Overlapping or reversed phone intervals')
        if a > cursor + 1e-7:
            implicit_seconds += a-cursor
            append('sp', cursor, a)
        phone = item.text.strip()
        if not phone:
            explicit_seconds += b-a
            phone = 'sp'
        append(phone, a, b)
        cursor = b
    frame = lambda t: int(np.round((t-start)*sampling_rate/hop_length))
    durations = [frame(b)-frame(a) for a, b in spans]
    expected = frame(end)
    if any(d < 0 for d in durations) or sum(durations) != expected:
        raise ValueError('Durations do not cover the cropped timeline')
    details = {'status':'passed', 'start_seconds':float(start), 'end_seconds':float(end),
               'timeline_seconds':float(end-start), 'frames':expected,
               'implicit_gap_seconds':implicit_seconds, 'blank_interval_seconds':explicit_seconds,
               'preserved_gap_seconds':implicit_seconds+explicit_seconds,
               'silence_tokens':phones.count('sp'),
               'duration_quantization':'rounded relative endpoints from cropped start'}
    return phones, durations, start, end, details
