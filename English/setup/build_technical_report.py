#!/usr/bin/env python3
"""Evidence-bound, print-readable technical report using ReportLab and HarfBuzz."""
from pathlib import Path
import copy
import html
import json
import math
import sys
from io import BytesIO

ROOT=Path(__file__).resolve().parents[2]


def main():
    from pypdf import PdfReader, PdfWriter
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak,Flowable,KeepTogether
    from reportlab.pdfgen.canvas import Canvas
    r=ROOT;d=json.loads((r/'English/artifacts/Assignment/analysis/measured-analysis.json').read_text())
    asr=json.loads((r/'English/artifacts/Assignment/analysis/english-asr.json').read_text())
    baselines=json.loads((r/'English/artifacts/Assignment/analysis/pause-baselines.json').read_text())
    coverage=json.loads((r/'English/artifacts/Assignment/analysis/english-speaker-coverage.json').read_text())
    listening=json.loads((r/'English/artifacts/Assignment/analysis/final-listening-review.json').read_text())
    listening_map={x['id']:x for x in listening['samples']}
    release=json.loads((r/'English/setup/checkpoint_manifest.json').read_text())['release_tag']
    gen=json.loads((r/'English/artifacts/Assignment/generation.json').read_text());info={s['id']:s for s in gen['samples']}
    audits=d['dataset_audits'];runs=d['training_runs'];samples=d['samples']
    assert set(listening_map)==set(samples)
    for sid,sample in samples.items():
        assert listening_map[sid]['audio_sha256']==sample['audio_sha256']
    selected=json.loads((r/'English/artifacts/Assignment/checkpoint-selection.json').read_text())
    fonts=r/'Report/fonts'
    for name,file in [('Body','DejaVuSans.ttf'),('Bold','DejaVuSans-Bold.ttf'),('Mono','DejaVuSansMono.ttf')]:
        pdfmetrics.registerFont(TTFont(name,str(fonts/file),shapable=False))
    pdfmetrics.registerFont(TTFont('Telugu',str(r/'Report/fonts/NotoSansTelugu-Regular.ttf'),shapable=True))
    pdfmetrics.registerFontFamily('Body',normal='Body',bold='Bold',italic='Body',boldItalic='Bold')
    C=colors.HexColor;navy=C('#172b44');teal=C('#087b78');gray=C('#546273');pale=C('#edf3f7')
    W=A4[0]-96
    styles={
      'body':ParagraphStyle('body',fontName='Body',fontSize=9.2,leading=13.1,spaceAfter=7,textColor=navy),
      'small':ParagraphStyle('small',fontName='Body',fontSize=7.7,leading=10.5,spaceAfter=5,textColor=gray),
      'table':ParagraphStyle('table',fontName='Body',fontSize=8,leading=11,textColor=navy),
      'th':ParagraphStyle('th',fontName='Bold',fontSize=8,leading=11,textColor=colors.white),
      'h1':ParagraphStyle('h1',fontName='Bold',fontSize=18,leading=23,spaceAfter=12,textColor=navy),
      'h2':ParagraphStyle('h2',fontName='Bold',fontSize=11.4,leading=15,spaceBefore=5,spaceAfter=6,textColor=teal),
      'title':ParagraphStyle('title',fontName='Bold',fontSize=27,leading=33,spaceAfter=14,textColor=navy),
      'source':ParagraphStyle('source',fontName='Body',fontSize=10.2,leading=14.2,spaceAfter=9,textColor=navy),
      'telugu':ParagraphStyle('telugu',fontName='Telugu',fontSize=12,leading=19,spaceAfter=9,shaping=1,textColor=navy),
      'code':ParagraphStyle('code',fontName='Mono',fontSize=7.2,leading=10,spaceAfter=5,textColor=navy,splitLongWords=True),
    }
    def esc(s):return html.escape(str(s)).replace('\u2011','-').replace('\u2013','-').replace('\u2014','-')
    class SourceParagraph(Paragraph):
        """Preserve original Unicode as accessible text around shaped glyphs."""
        def __init__(self, text, style, actual_text):
            super().__init__(text, style)
            self.actual_text=actual_text
        def draw(self):
            payload=('\ufeff'+self.actual_text).encode('utf-16-be').hex().upper()
            self.canv._code.append('/Span << /ActualText <'+payload+'> >> BDC')
            try:super().draw()
            finally:self.canv._code.append('EMC')
    def P(s,style='body',markup=False):
        text=s if markup else esc(s)
        return SourceParagraph(text,styles[style],str(s)) if style=='telugu' else Paragraph(text,styles[style])
    story=[];pages=[]
    def page(title):
        if story:story.append(PageBreak())
        pages.append(title);story.append(P(title,'h1'))
    def para(s,style='body'):story.append(P(s,style))
    def sub(s):story.append(P(s,'h2'))
    def table(headers,rows,widths=None):
        cells=[[P(s,'th') for s in headers]]+[[P(x,'table') for x in row] for row in rows]
        t=Table(cells,colWidths=widths or [W/len(headers)]*len(headers),repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),navy),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,pale]),
          ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),
          ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),('LINEBELOW',(0,-1),(-1,-1),.4,C('#c5d2dc'))]))
        story.extend([t,Spacer(1,9)])
    def fig(name,caption,width=W):
        from PIL import Image as PI
        path=r/'Report/figures'/name
        with PI.open(path) as im:height=width*im.height/im.width
        story.extend([Image(str(path),width=width,height=height),P(caption,'small')])
    def code(s):story.append(P(s,'code'))
    def run(key):return runs[key]
    en=run('gap-fixed-v2-20261103/english');te=run('gap-fixed-v2-20261103/telugu')
    ft=run('english-finetune-pilot-20261104/english');lp=run('english-learned-pauses-pilot-20261104/english')
    def pct(v):return f'{100*v:.1f}%'
    def statsline(s):return f"{s['duration_seconds']:.3f} s | {s['mel_frames']} frames | {s['lexical_phone_count']} lexical phones | RMS {s['rms_dbfs']:.1f} dBFS"

    class Pipeline(Flowable):
        def __init__(self):Flowable.__init__(self);self.width=W;self.height=100
        def draw(self):
            c=self.canv;labels=['Phones +\nposition','FFT\nencoder','Variance\nadaptor','FFT +\npostnet','HiFi-GAN\nwaveform'];boxw=86;gap=(W-5*boxw)/4
            for i,text in enumerate(labels):
                x=i*(boxw+gap);c.setFillColor(pale);c.setStrokeColor(teal);c.roundRect(x,35,boxw,48,6,fill=1,stroke=1)
                c.setFillColor(navy);c.setFont('Body',8.5)
                for j,line in enumerate(text.split('\n')):c.drawCentredString(x+boxw/2,63-j*12,line)
                if i<4:
                    c.setStrokeColor(gray);c.line(x+boxw+2,59,x+boxw+gap-2,59)
            c.setFont('Body',7.8);c.setFillColor(gray);c.drawString(0,13,'Separate acoustic models. Speaker conditioning; English adds text-derived boundary embeddings and pause heads.')

    # 1 - problem, approach and development sequence precede detailed results.
    pages.append('Abstract and implementation overview');story.append(Spacer(1,22))
    story.append(P('FastSpeech 2<br/>English and Telugu','title',True))
    story.append(P('Implementation, training and test-sentence analysis','h2'))
    para('Parichay Mago | Technical report','small')
    sub('Abstract')
    para('This work develops two independent FastSpeech 2 text-to-speech systems for English and Telugu using Svarah and IndicTTS Telugu. We began by examining the reference architecture, checking dataset access and transcript quality, and selecting approximately five hours of training audio and one hour of validation audio per language. We then implemented language-specific text processing, phoneme conversion, forced alignment and acoustic-feature extraction. Separate acoustic models predict mel spectrograms, which a pretrained HiFi-GAN converts into speech [1-5]. The experiments investigate training convergence, alignment quality and English prosody through baseline training, feature refinement and targeted fine-tuning. Evaluation covers all ten assignment sentences and their predicted duration, pitch and energy trajectories.')
    sub('Implementation overview')
    para('The first implementation stage established the data pipeline. We checked usable audio duration, built reproducible train/validation splits and handled speaker conditioning. English uses estimated acoustic speaker labels. Telugu uses a pronunciation lexicon and a stable phone mapping compatible with the reference implementation.')
    para('We trained separate English and Telugu models using aligned duration, pitch and energy targets. We reviewed the alignment, updated feature extraction to preserve internal silence, rebuilt both feature sets and trained both models from scratch.')
    para('The English experiments continued with low-rate fine-tuning and a model extension with word-boundary and punctuation embeddings, plus predictors for pause presence and duration. Finally, we selected checkpoints using complete validation evaluations, generated the required test audios and analysed their variance predictions alongside the loss curves.')
    para('The report follows an introduction-method-experiments-discussion structure, informed by the Seed-TTS technical report [11]. Methods and data appear on pp. 2-5, experimental results on pp. 6-9, sentence-level analysis on pp. 10-19, and discussion, reproducibility, software verification, speaker coverage and references on pp. 20-26.', 'small')

    # 2
    page('1. Introduction and dataset preparation')
    para('Adapting a text-to-speech system to two languages requires a consistent path from written text to pronunciation, acoustic timing and waveform generation. We adapted speaker conditioning for English, phoneme representation for Telugu and alignment-based supervision for both. The objective was to build and evaluate two separate systems from the reference FastSpeech 2 implementation [1,2].')
    para('The assignment specifies two separate models, approximately five hours of training audio and one hour of validation per language, the exact five test sentences per language, loss curves, and duration/F0/energy analysis for every test. E2a/E2b and T2a/T2b count separately, giving ten required audios and thirty variance figures.')
    sub('1.1 Datasets and usable duration')
    rows=[]
    for language in ('english','telugu'):
        raw=d['data_splits']['datasets'][language]['partitions'];feat=d['feature_rebuild']['languages'][language]['rebuild']
        rows.append([language.title(),f"{raw['train']['rows']:,} / {raw['validation']['rows']:,}",f"{raw['train']['hours']:.5f} / {raw['validation']['hours']:.5f}",f"{feat['split_rows']['train']:,} / {feat['split_rows']['validation']:,}",f"{feat['feature_hours']['train']:.4f} / {feat['feature_hours']['validation']:.4f}"])
    table(['Language','Source clips train/val','Source hours train/val','Usable clips train/val','Feature hours train/val'],rows,[65,110,105,110,W-390])
    para('Source duration satisfies the approximately 5 h / 1 h selection target. The table also gives the number of clips and feature duration after alignment and edge-silence cropping.')
    sub('1.2 Subsampling and split separation')
    para('Seed 20261102 determines an ordering of atomic groups. Validation is selected first, then training, greedily approaching each duration target. Groups connect repeated normalized transcripts, identical audio hashes and English chunks from the same recording. Each group belongs to a single split. The selected Telugu subset uses the source male label. The splits hold out utterances and recording groups.')
    sub('1.3 Speaker conditioning')
    para('Training-only average-linkage cosine clustering of pretrained SpeechBrain ECAPA embeddings uses a distance threshold of 0.30 and produces 254 estimated labels. Validation uses frozen training centroids. All final English tests retain en_est_0155 from earlier diagnostics. Appendix D gives label coverage and selection details. Telugu uses the source label te_male [9].')
    sub('1.4 Dataset revisions')
    for language in ('english','telugu'):
        para(d['data_splits']['datasets'][language]['dataset'],'small');code(d['data_splits']['datasets'][language]['revision'])
    para('Supporting records are English/artifacts/DataSplits/summary.json, verification.json, English/artifacts/DatasetAudit/*-audit.json and English/artifacts/Preprocessing/gap-fixed-v2/rebuild-summary.json.', 'small')

    # 3
    page('2. Method: architecture and language support')
    story.append(Pipeline())
    para('The reference variant encodes phones with feed-forward Transformer blocks, predicts duration, pitch and energy, length-regulates the sequence, and decodes mel spectrograms. A convolutional postnet refines mels. This project uses direct phoneme-level F0 and an external HiFi-GAN [1-3].')
    table(['Component','Executed configuration'],[
      ['Encoder / decoder','4 / 6 FFT blocks; 2 attention heads; hidden size 256'],
      ['FFT feed-forward','Filter size 1024; kernel sizes 9 and 1; dropout 0.2'],
      ['Variance predictors','Filter size 256; kernel 3; dropout 0.5'],
      ['Variance embeddings','256 linearly quantized bins each for pitch and energy'],
      ['Length and speakers','Maximum sequence 2048; English 254 estimated labels; Telugu 1 label'],
      ['Acoustic targets','80-bin log mels; 22,050 Hz; FFT/window 1,024; hop 256; 0-8,000 Hz'],
      ['Waveform synthesis','Frozen universal HiFi-GAN; mono PCM16 output; no vocoder retraining']], [120,W-120])
    sub('English learned-pause extension')
    para('Each word has a boundary slot, including words with zero measured pause. A seven-entry boundary embedding conditions the encoder; two new variance predictors estimate pause presence and positive pause length. Punctuation categories distinguish unpunctuated, comma, period, question, exclamation and clause boundaries. The parent acoustic weights are loaded exactly; new heads receive a larger learning rate. The existing pitch and energy heads also receive the new encoder context.')
    para('At inference a sigmoid threshold of 0.5 selects a positive pause. The predicted log duration becomes rounded frames, and the length regulator generates the acoustic span. Terminal boundaries supply sentence context and receive zero frames. The cap of 354 frames is the training-observed maximum. Lexical durations have a one-frame floor.')
    sub('Telugu adaptation')
    para('The Telugu path uses a bijective IPA-to-ASCII phone map, a language-specific pronunciation lexicon and aligner, its own statistics and checkpoint, and the base variance adaptor. The acoustic models and optimizers are separate, with a shared frozen vocoder.')

    # 4
    page('2.1 Preprocessing and alignment')
    sub('Text, audio and alignment')
    para('English normalizes text through the reference cleaners and g2p-en ARPAbet conversion. Telugu uses NFC, number verbalization and eSpeak-NG Telugu pronunciations, mapped to stable lowercase telph identifiers. English training pronunciations cover 6,470 words; Telugu covers 13,848. English symbol IDs remain stable when Telugu symbols are appended [7,8].')
    para('Original English source is 16 kHz mono; the selected Telugu source is 48 kHz mono. scipy resample_poly creates 22,050 Hz PCM16 copies. Original WAVs and raw transcripts remain unchanged. MFA 3.3.9 supplies alignments: English uses english_us_arpa with speaker adaptation disabled; Telugu fits an acoustic aligner using training audio only, then freezes it for validation. Telugu alignment is not the TTS model itself [6].')
    sub('Internal silence in aligned targets')
    para('The gap-fixed-v2 feature extractor represents internal silence with explicit tokens, preserves the full cropped timeline and enforces sum(duration frames)=mel length. Both language models were trained from scratch on these features.')
    table(['Alignment measure','English','Telugu'],[
      ['Utterances with preserved internal silence','2,965','2,451'],
      ['Internal silence duration (train+val)','0.8870 h','0.7228 h'],
      ['Accepted train / validation','3,037 / 617','2,153 / 428'],
      ['Largest mel length','1,674 frames','1,698 frames']], [W*.52,W*.24,W*.24])
    sub('Feature extraction and normalization')
    para('WORLD DIO plus StoneMask estimates F0. The extractor averages interpolated F0 and frame energy over phones. Spectral energy is the linear L2 norm. Pitch and energy means, standard deviations and bins are fitted on training features using per-example IQR filtering. Validation uses frozen parameters. The extractor crops edge silence, retains internal silence and verifies that arrays are finite, use known phones and have consistent lengths.')
    para('The English pause pilot uses all 3,654 clips from the aligned feature set. Each internal word boundary receives its measured duration in frames, including zero where the alignment shows no silence. The pilot preserves lexical phone durations, pitch and energy, source mel bytes, split membership, speaker mapping and normalization statistics.')
    para('Supporting records are Report/Alignment_Gap_Fix.txt, English/artifacts/Preprocessing/gap-fixed-v2/ and English/artifacts/Preprocessing/learned-pauses-v1/. Normalization statistics, speaker centroids and the Telugu alignment model are fitted on training data.', 'small')

    # 5
    page('3. Experimental setup and loss functions')
    table(['Setting','Corrected paired run','English fine-tune','English pause pilot'],[
      ['Initialization','Random, separate models','English corrected best 12k','Fine-tune best +2,500'],
      ['Batch / gradient clipping','8 / norm 1.0','8 / norm 1.0','8 / norm 1.0'],
      ['Optimizer','Independent Adam','New Adam state','New Adam state'],
      ['Betas / epsilon','0.9,0.98 / 1e-9','Same','Same'],
      ['Learning rate','4000-step warmup, then inverse square root','Fixed 2e-5','Base 2e-5; new heads 2e-4'],
      ['Updates reached','30k per language','5,000 additional','5,000 additional'],
      ['Validation / save','Every 500 / 100 updates','Same','Same'],
      ['Stopping','Step limit','5 non-improving validations','8 non-improving validations']], [93,137,130,W-360])
    para('Executed runner arguments override the upstream batch 16 / 900k-step template. Training GPUs were RTX PRO 4500 Blackwell for the corrected paired run, A40 for the fine-tune, and RTX 3090 for the pause pilot; PyTorch 2.8.0 with CUDA 12.8 and Python 3.10 were used. Final inference and this analysis run locally on CPU.')
    sub('Base objective: five loss terms')
    code('Lbase = Lmel + Lpostnet + Lpitch + Lenergy + Llog_duration')
    para('Mel terms use L1 error; pitch, energy and log(1+duration) use MSE. Six numbers are logged: total followed by the five component losses. No claim of six separate component terms is intended.')
    sub('Learned-pause objective: seven loss terms')
    code('Lpause = Lmel + Lpostnet + Lpitch + Lenergy + Lspeech_duration')
    code('         + 0.25 * Lpresence + 0.25 * Lpositive_pause_duration')
    para('Pause presence uses weighted BCE, with training-only positive weight 2.9192; positive pause length uses log-duration MSE. Speech-duration loss excludes boundary slots; pitch/energy losses exclude tokens without target frames. Eight numbers are logged including total. Because both terms and masks change, base and pause totals are not directly comparable.')
    sub('Validation interpretation')
    para('Checkpoint selection uses complete held-out evaluations. Batch losses are averaged with example-count weights. Validation supplies target duration, pitch and energy to the mel decoder and records predictor errors alongside mel losses. Training minibatches use dropout.')

    # 6
    page('4. Results: baseline model training')
    fig('corrected_training.png','Figure 1. Original-objective losses. Top: raw minibatches, moving means and complete validation. Bottom: validation detail. Gold marks the selected minimum; each language has its own dataset and target distribution.')
    table(['Run','Best validation','Final validation','Final above best'],[
      ['English corrected',f"{en['best']['losses'][0]:.4f} at 12,000",f"{en['last']['losses'][0]:.4f}",f"{en['final_minus_best_validation_percent']:.2f}%"],
      ['Telugu corrected',f"{te['best']['losses'][0]:.4f} at 18,000",f"{te['last']['losses'][0]:.4f}",f"{te['final_minus_best_validation_percent']:.2f}%"]], [110,140,120,W-370])
    para('Both training losses continue downward through 30,000 updates. Validation reaches its minimum at step 12,000 for English and step 18,000 for Telugu. At the end, English has a minibatch loss of 1.1430 and validation loss of 2.6560. Telugu has a minibatch loss of 0.8414 and validation loss of 2.1311. Each run selects its checkpoint at the validation minimum.')
    para('The two runs use different languages, speaker compositions and normalized target distributions. Loss trends are interpreted within each run.')

    # 7
    page('4.1 English fine-tuning experiments')
    fig('english_pilots.png','Figure 2. Separate English pilot axes. Step 0 is the re-evaluated parent before updates. Left retains the base objective; right adds pause penalties and changes masks. Curves must not be concatenated into one comparable loss scale.')
    table(['Pilot','Own baseline','Best','Final'],[
      ['Fixed-rate fine-tune','2.6051','2.5788 at +2,500','2.5879'],
      ['Learned-pause','3.3578','3.0544 at +1,000','3.0984']], [W*.32,W*.20,W*.27,W*.21])
    para('The fine-tune reduces its original objective by 1.01%, then early-stops. The pause pilot reduces its own baseline by 9.03%, then plateaus for eight evaluations. Its best total is 2.5444 from the five non-pause components plus 0.5100 from the two weighted pause penalties. Even the 2.5444 remainder is not an exact old-objective comparison because masks and input sequences changed.')
    para('Postnet mel loss is 0.7462 for the parent and 0.7482 for the selected pause model. The final English checkpoint is the pause model at the minimum complete validation loss for that run.')

    # 8
    page('4.2 Learned pause evaluation')
    fig('pause_validation.png','Figure 3. Held-out pause-head metrics. F1 starts at zero. Duration MAE covers all 1,751 ground-truth positive-duration boundaries.')
    pm=lp['best']['pause_metrics'];basepm=lp['first']['pause_metrics']
    table(['At selected step +1,000','Value'],[
      ['Pause precision / recall / F1',f"{pct(pm['precision'])} / {pct(pm['recall'])} / {pct(pm['f1'])}"],
      ['TP / FP / FN',f"{pm['tp']} / {pm['fp']} / {pm['fn']}"],
      ['Positive-pause duration MAE',f"{pm['positive_pause_mae_seconds']*1000:.1f} ms (baseline {basepm['positive_pause_mae_seconds']*1000:.1f} ms)"],
      ['Positive / all boundaries: train; validation','8,994 / 35,249; 1,751 / 7,348']], [W*.53,W*.47])
    baseline_rows=[]
    for name,label in [('every_internal_boundary','Every internal boundary'),('punctuation_only','Punctuation only')]:
        v=baselines['baselines'][name];baseline_rows.append([label,pct(v['precision']),pct(v['recall']),pct(v['f1'])])
    v=baselines['selected_model'];baseline_rows.append(['Selected model (+1,000)',pct(v['precision']),pct(v['recall']),pct(v['f1'])])
    table(['Held-out occurrence policy','Precision','Recall','F1'],baseline_rows,[W*.43,W*.19,W*.19,W*.19])
    para('The baselines use the same 7,348 internal validation boundaries, with 1,751 positive targets; terminal slots are excluded. Every-boundary prediction gives TP/FP/FN 1,751/5,597/0; punctuation-only gives 214/75/1,537. They are evaluation comparators and do not alter synthesized audio. Model counts come from the saved complete validation, not a new GPU evaluation.','small')
    para('The selected model improves F1 over the every-boundary baseline by 7.2 percentage points. E4 assigns zero frames at its comma. The final English checkpoint uses the punctuation-aware architecture and the minimum validation loss within its run.')
    para('Training counts of positive pauses and internal markers are 1,050 of 1,445 for commas, 12 of 12 for question marks and 1 of 2 for clause markers. The internal exclamation count is zero. Most period and question markers are terminal. The preprocessing summary gives the full counts.', 'small')

    # 9
    page('4.3 Generated speech evaluation')
    para('All outputs are mono 22,050 Hz PCM16, from identical frozen vocoder weights, unity pitch/energy/duration controls and a single utterance inference. No target prosody, denoising, fades, inserted silence or sentence concatenation is applied. Waveform lengths equal predicted mel frames times 256. No output has full-scale clipping.')
    rows=[]
    asrmap={x['id']:x for x in asr['generated']}
    for sid,s in samples.items():
        rows.append([sid,f"{s['duration_seconds']:.3f}",str(s['mel_frames']),f"{s['rms_dbfs']:.1f}",
                     f"{s['lexical_F0_hz_min_max'][0]:.0f}-{s['lexical_F0_hz_min_max'][1]:.0f}",
                     pct(asrmap[sid]['WER']) if sid in asrmap else 'Not scored'])
    table(['ID','Seconds','Frames','RMS dBFS','Predicted F0 Hz','ASR disagreement'],rows,[43,63,60,73,120,W-359])
    sub('English recognition check')
    para(f"Whisper-small is run locally with language en, temperature 0, no prompt and no previous-text conditioning [10]. Clock/date and ordinary number normalization are applied equally to reference and hypothesis before edit alignment. Generated disagreement is {pct(asr['generated_aggregate']['WER'])} ({asr['generated_aggregate']['errors']}/{asr['generated_aggregate']['reference_words']} edits/words). Three natural validation controls from the same estimated cluster give {pct(asr['natural_controls_aggregate']['WER'])} (1/18). The controls use separate, shorter source sentences.")
    para('Recognition reproduces the normalized E2b text. The E2a difference is in possessive spelling. The table reports English recognition results alongside acoustic measurements for both languages.')
    para('The F0 ranges describe model conditioning. Spectral high-band energy and low-level frame fractions were also computed.', 'small')

    # One complete source + all three compact curves + individual analysis per test.
    challenges={'E1':'Long English word','E2a':'Conduct as a noun','E2b':'Conduct as a verb','E3':'Visible split word','E4':'Date, clock and code-mixing',
                'T1':'Long Telugu technical term','T2a':'Pattu in the grip context','T2b':'Pattu in the silk context','T3':'Visible split Telugu word','T4':'Telugu date and clock'}
    comments={
     'E1':[
      'The frontend uses the full 28-phone CMU entry. Its predicted span is 1.881 s, about one third of the 5.666 s utterance.',
      'The target region has weighted F0 conditioning of 235.3 Hz and local duration variation. F0 conditioning falls about 48 Hz between the first and last lexical-phone quintiles, with energy falling near the end.',],

     'E2a':[
      'The frontend identifies conduct as a noun and supplies K AA1 N D AH0 K T. The seven phones occupy 0.476 s; their F0 conditioning averages 214.2 Hz and spectral energy 23.6.',
      'E2a has zero internal pause frames. Its energy curve varies across phones and declines at the end.',
      'The 9.1% recognition edit rate comes from a single possessive spelling difference.'],
     'E2b':[
      'Following will, conduct is tagged as a verb and encoded K AH0 N D AH1 K T. It occupies 0.430 s versus 0.476 s in E2a. Weighted F0 is 217.0 Hz and spectral energy 34.4; these are contextual outputs under different phone stress and sentence positions.',
      'The learned model predicts one short boundary pause. The noun/verb stress contrast comes from the text frontend.',
      'Recognition reproduces all normalized words, giving 0/11 edits.'],
     'E3':[
      'The frontend preserves the supplied rhyth m as two tokens. The fallback maps rhyth to R AY1 DH and m to EH1 M. Each part is predicted for 0.197 s.',
      'The boundary predictor assigns zero internal pause frames. The duration-weighted spectral energies of rhyth and m are 50.7 and 41.1. F0 conditioning ranges from 181 to 231 Hz.',
      'The recognizer returns right end in the corresponding region. The normalized disagreement is 25.0%.'],
     'E4':[
      'Spoken normalization expands 10:45 AM, the ordinal day, and 2026; the Latin andaroo token uses a documented English-phone approximation. The output contains 22 normalized spoken tokens in 6.014 s (219.5 tokens/min), faster than the other English test cases under this tokenization.',
      'The comma after the date receives zero pause frames. Two other boundaries contribute 0.163 s total. The andaroo span occupies 0.360 s with weighted F0 conditioning of 214.1 Hz and spectral energy of 40.5, compared with 21.5 for twelfth.',
      'The normalized recognition edit rate is 54.5%. The decoded text below shows how the recognizer rendered the clock and date.'],
     'T1':[
      'Telugu eSpeak supplies 19 phones for the opening technical term, all mapped into the frozen inventory. Its span is 1.463 s of the 4.540 s output.',
      'The target region reaches approximately 277 Hz F0 conditioning, then the utterance declines toward 119 Hz. F0 conditioning falls by approximately 74 Hz from the first to the last lexical-phone quintile, with alternating spectral energy.'],

     'T2a':[
      'The shared target pattu is encoded /p a retroflex-t retroflex-t u/ using the same five trained phone IDs in both T2 cases. Here its total is 23 frames (0.267 s); weighted F0 conditioning is 201.4 Hz and spectral energy 19.6.',
      'The paired retroflex tokens receive 5+5 frames. The curves show local context effects and a final F0 descent.',
      'The source uses pattu in the sense of grip or hold.'],
     'T2b':[
      'The same pattu phones occupy 25 frames (0.290 s), 23 ms longer than in T2a. Weighted F0 conditioning is 197.1 Hz and spectral energy 22.5. The paired retroflex tokens receive 6+4 frames.',
      'The model predicts different prosody for the word in the silk-sari and grip contexts, which differ in word position and neighboring phones.',
      'The decoded waveform stays within the PCM16 range.'],
     'T3':[
      'The visible source split ada/vilo is preserved as two input words. Their predicted spans are 0.186 s and 0.209 s, with weighted spectral energies of 61.1 and 54.4. Both use automatic Telugu pronunciations.',
      'This is the shortest Telugu test at 1.753 s. F0 conditioning falls about 65 Hz from the first to the last lexical-phone quintile. Its high-band power fraction of 1.02% is the largest in this set.'],

     'T4':[
      'The clock phrase is verbalized with hour and minute units, 12va becomes the inflected date ordinal, and 2026 becomes a cardinal year. The .txt and JSON files preserve the source text.',
      'All 129 input phones receive frames in the 7.233 s output. F0 conditioning ranges from 99 to 233 Hz and ends lower. Energy also falls in the last phones.',
      'Trained phones represent the complete date and time sequence.']}
    for case_number,(sid,s) in enumerate(samples.items(),start=1):
        case_title=f'5.{case_number} '+sid+' | '+challenges[sid]
        if case_number==1:
            page('5. Test-sentence analysis')
            sub(case_title)
        else:page(case_title)
        story.append(P(s['source_text'],'telugu' if s['language']=='telugu' else 'source'))
        para(statsline(s),'small')
        fig(sid+'_variance.png','All model predictions are plotted by token index. Green spans highlight the test word(s); faint orange marks English boundary slots. F0 and energy omit zero-duration tokens. Full phone-labeled originals and CSVs remain in Results/.')
        for text in comments[sid]:para(text)
        if sid in asrmap:
            para('ASR diagnostic: '+asrmap[sid]['ASR_text'].strip(),'small')
        item=listening_map[sid]

        para('Author listening observation: '+item['overall_status'].title()+'.','small')
        para('WAV: Results/'+s['language'].title()+'/test_generated_audios/'+sid+'.wav. Text, metadata, predictions and CSV: '+s['language'].title()+'/artifacts/test_generated_audios/.','small')

    # 20
    page('6. Discussion and conclusions')
    table(['Aspect','English final model','Telugu final model'],[
      ['Data / voice conditioning','Svarah; 254 estimated labels; one cluster used for tests','IndicTTS male subset; one source label'],
      ['Final architecture','FastSpeech2 plus learned boundaries and pause heads','Base FastSpeech2 with Telugu phone map'],
      ['Validation result','3.0544 under new seven-term objective','2.1065 under base five-term objective'],
      ['Test cases','Long word, split word, mixed date/time and conduct stress','Technical term, split word, pattu contexts and date/time'],
      ['Timing model','Learned boundary predictions with pause detection F1 of 45.6%','Phone durations from the base variance adaptor'],
      ['Author overall acceptance','5/5 recordings marked Good','5/5 recordings marked Good'],
      ['Evaluation evidence','Author ratings, acoustic measurements and five ASR diagnostics','Author ratings and acoustic measurements']], [105,197,W-302])
    para('Each language uses its own loss terms, masking, dataset, speaker composition and normalization. The comparison covers validation, variance predictions, waveform measurements and author ratings.')
    sub('Conclusions')
    para('The system preserves the source text and reproduces all ten test outputs. It predicts varying phoneme-level duration, pitch and energy, uses frontend-supplied conduct stress and changes pattu predictions across contexts. Complete validation selects the final checkpoints. English pause F1 is 45.6% compared with 38.5% and 21.0% for the two occurrence baselines.')

    para('The author rated all ten recordings Good and reported both language sets acceptable, with no specific pronunciation issues noticed.')


    # 21
    page('Appendix A. Reproducibility')
    para('Run commands from the repository root. Use an isolated Python 3.10 environment. English/setup/requirements-inference.txt pins inference packages; root requirements.txt combines preprocessing, training and inference dependencies. Recorded GPU training used PyTorch 2.8.0+cu128. The short README covers inference; English/docs/REPRODUCIBILITY.md gives CUDA wheels, MFA setup, preprocessing, training and verification commands.')
    for command in ['uv venv --python 3.10 .venvs/final-inference',
       'uv pip install --python .venvs/final-inference/bin/python -r English/setup/requirements-inference.txt',
       'python3 English/setup/download_checkpoints.py',
       '.venvs/final-inference/bin/python English/setup/prepare_final_inference.py --nltk-only',
       '.venvs/final-inference/bin/python English/inference.py --test-id E1',
       '.venvs/final-inference/bin/python Telugu/inference.py --test-id T1',
       '.venvs/final-inference/bin/python English/setup/generate_assignment_outputs.py',
       '.venvs/final-inference/bin/python English/setup/verify_assignment_outputs.py']:
        code(command)
    sub('Preprocessing in a clean reconstruction')
    para('The reproduction guide lists pinned dataset-audit commands, English/setup/restore_assignment_splits.py and export_split_audio.py to restore source membership and frozen speaker labels after authorized dataset access. Linux alignment expects MFA 3.3.9 at /workspace/.venvs/mfa and english_us_arpa.zip under /workspace/.cache/mfa/pretrained_models/acoustic/. These wrappers build corpora, alignments and corrected features, then check a real batch:')
    code('python English/preprocess.py --stage all --device cpu --feature-version gap-fixed-v2')
    code('python Telugu/preprocess.py --stage all --device cpu --feature-version gap-fixed-v2')
    code('python English/setup/build_learned_pause_features.py')
    para('The last command builds measured pause features and the three training YAMLs. For existing features, add --configs-only to reconstruct the YAMLs without rewriting targets. The committed English/config/learned-pauses-v1/English/ snapshot records the original experiment. Dataset access follows source license/gate conditions.','small')
    sub('Training reconstruction')
    code('python English/train.py --run-id english-new --fresh --minutes 120 --max-steps 30000 --batch-size 8')
    code('python Telugu/train.py --run-id telugu-new --fresh --minutes 120 --max-steps 30000 --batch-size 8')
    para('These commands start new CUDA runs. Fine-tuning adds --initialize-from, its matching --initialize-sha256, --learning-rate 2e-5 and patience 5. Pause training adds --learned-pauses, its generated --config-root, --new-head-learning-rate 2e-4 and patience 8. Resume with --resume and the run ID, without initialization flags. A retained session/STOP is rejected before writes; remove only that run\'s marker after explicitly deciding to continue. Policy, optimizer and best selection are restored. Explicit --device cpu supports local tests.')

    # 22
    page('Appendix B. Checkpoints and artifacts')
    para('The final English weights are learned-pause step +1,000 on fine-tune step +2,500, itself initialized from corrected step 12,000. Telugu uses corrected step 18,000. Parent checkpoints remain separate. SHA256 fingerprints identify the exact selected local files:')
    for language in ('english','telugu'):
        para(selected[language]['path'],'small')
        code(selected[language]['sha256'])
    table(['Artifact','Location'],[
      ['Frozen weights / metadata','English/checkpoints/ and Telugu/checkpoints/'],
      ['Inference configs/resources','English/config.yaml, Telugu/config.yaml; resources/ in each'],
      ['Audios','Results/{English,Telugu}/test_generated_audios/'],
      ['Text, metadata and predictions','{English,Telugu}/artifacts/test_generated_audios/'],
      ['Thirty full-label variance figures','Results/{English,Telugu}/variance_adaptor_plots/'],
      ['Final selected-run loss plots','Results/{English,Telugu}/{training,validation}_loss.png'],
      ['Interactive local review / output ZIP','English/artifacts/Assignment/index.html; assignment_test_outputs.zip'],
      ['Measured analyses / recognition','English/artifacts/Assignment/analysis/'],
      ['Report source and figure scripts','English/setup/build_technical_report.py; create_report_figures.py']], [W*.40,W*.60])
    para('The repository root is the assignment\'s project/ directory: English/, Telugu/, Results/, Report/, README.md and requirements.txt. Each language folder contains its own model, preprocessing, training and inference implementation and frozen vocoder. Reproduction tools and combined experiment records live under English/. Original upstream demos are isolated in English/reference/. Results/ contains only required loss PNGs, WAVs and variance PNGs. Combined supporting records are in English/artifacts/; language-specific inference metadata is under each language folder. The repository includes a map of the previous paths. Historical run records retain recorded paths and hashes. Current commands above use the reorganized source paths.','small')
    para('The universal vocoder is copied into each language/hifigan/ folder. Both files retain the same frozen SHA256; the layout change does not retrain the vocoder or alter waveform synthesis.','small')
    para('The assignment release contains a complete project/ archive with separate English/Telugu model-only checkpoints and the frozen vocoder. SHA256SUMS verifies the archive; English/setup/download_checkpoints.py verifies each restored model. Per-language training wrappers, updated requirements and the README are included. The repository URL remains to be submitted through the assignment form.','small')
    story.append(P('<link href="https://github.com/ParichayMago/fastspeech2-english-telugu/releases/tag/'+release+'" color="#087b78">Complete project and checkpoint download</link>','small',True))
    story.append(P('<link href="https://github.com/ParichayMago/fastspeech2-english-telugu" color="#087b78">GitHub repository: ParichayMago/fastspeech2-english-telugu</link>','small',True))
    sub('Recreate the report without cloud compute')
    code('.venvs/inference/bin/python English/setup/analyze_assignment_outputs.py')
    code('.venvs/inference/bin/python English/setup/create_report_figures.py')
    code('.venvs/report/bin/python English/setup/build_technical_report.py')
    para('The report builder consumes the saved English recognition cache. Optional analysis flag --asr computes missing or changed Whisper-small results and requires openai-whisper==20250625. Report dependencies are pinned in Report/requirements-report.txt. The included Noto Sans Telugu font is shaped with HarfBuzz and distributed under the accompanying SIL Open Font License.','small')

    # 23 - distinguish implementation corrections from historical model training.
    page('Appendix C. Software verification')
    para('The submission includes checkpoint recovery, device portability, final-recording discovery and training stop handling. It retains the training logs, weights, ten test WAVs and prediction arrays.')
    table(['Feature','Verified behavior'],[
      ['Best-checkpoint preservation','Resume restores the best loss, checkpoint hash and early-stopping counter. Complete validation selects the checkpoint with the lowest validation loss.'],
      ['Base and fine-tune recovery','Resume restores the saved model, configuration and Adam state, including fixed learning-rate groups and early stopping.'],
      ['Pause configurations','Feature construction writes all three YAMLs from measured training statistics. --configs-only reconstructs them; original configs are also distributed.'],
      ['Tensor device allocation','Masks use lengths.device and length-regulator output uses x.device, independent of CUDA availability.'],
      ['Energy control','Both phoneme-level and frame-level energy paths receive e_control. Unity-control generation remains identical.'],
      ['Review interface','Discover all ten final WAV/JSON pairs, original transcripts and current hashes; export all reviewed statuses. Saved judgments remain local.'],
      ['Retained stop request','A saved session/STOP marker blocks resume before writes. A stop request during a run saves its current progress.']], [W*.31,W*.69])
    sub('Local validation')
    para('Each language-local setup suite passes 41 regression checks. The review server passes nine, giving 91 checks across the two implementations and the server. Synthetic CPU fixtures cover base, fixed-rate and pause recovery, legacy checkpoints, Adam continuation, historical best/config restoration and stop handling. Real-data pause integration also passes locally. Browser checks covered audio playback, save/reload/export, draft retention and narrow-screen layout.')
    para('All ten frozen sentences regenerate with identical WAV SHA256 values and exact equality of every exported array, including mels and pause-head outputs. CPU device tests simulate CUDA availability and include meta-device allocation.')
    para('For the language-local layout, preprocessing/training help and real CPU inference also pass with the other language directory absent. A real-data CPU update and validation pass for each language using throwaway models; released weights are unchanged. The language layout verification and preprocessing records contain the details.','small')
    sub('Reproduce the software checks')
    code('python -m unittest discover -s English/setup -p test_*.py -v')
    code('python -m unittest discover -s Telugu/setup -p test_*.py -v')
    code('python -m unittest discover -s English/audio_review -p test_server.py -v')
    code('python English/setup/verify_frozen_regeneration.py')
    code('python English/setup/evaluate_pause_baselines.py')
    para('Recovery tests use the full requirements.txt stack, including TensorBoard. The baseline command uses local held-out pause targets. The saved baseline results are distributed with the project.', 'small')

    # 24
    page('Appendix D. Speaker coverage')
    selected_cov=coverage['selected_training_source'];accepted_cov=coverage['accepted_training_source']
    sub('Estimated English label coverage')
    table(['Measure','Selected source','Accepted source'],[
      ['Training clips / estimated labels',f"{selected_cov['clips']:,} / {selected_cov['labels']}",f"{accepted_cov['clips']:,} / {accepted_cov['labels']}"],
      ['Labels with 1-2 clips',f"{selected_cov['labels_with_one_or_two_clips']} (30.3%)",f"{accepted_cov['labels_with_one_or_two_clips']} (30.3%)"],
      ['Median clips / source seconds per label',f"7 / {selected_cov['median_source_seconds_per_label']:.1f} s",f"7 / {accepted_cov['median_source_seconds_per_label']:.1f} s"],
      ['Final label en_est_0155: clips',str(selected_cov['final_label']['clips']),str(accepted_cov['final_label']['clips'])],
      ['Final label: source duration',f"{selected_cov['final_label']['source_minutes']:.3f} min",f"{accepted_cov['final_label']['source_minutes']:.3f} min"]],[W*.54,W*.23,W*.23])
    para('The table gives source WAV durations, including edges. Shared English parameters learn from the full approximately five-hour selection across all labels. The accepted aligned features total 4.6536 hours, as shown in Section 1.1.', 'small')
    para('ECAPA clustering uses training recordings with at least 3 s of speech, average-linkage cosine distance and a threshold of 0.30. Short training and validation recordings use frozen training centroids. The source dataset reports 117 speakers [4]. The training pipeline produces 254 estimated labels.')
    para('The final label is third by selected source duration and is retained for continuity with earlier previews.')



    code('python English/setup/analyze_speaker_coverage.py')
    para('The script recomputes coverage from selection-membership.jsonl, the feature-selection records and saved clustering metadata. Full per-label counts, durations and input SHA256 values are in English/artifacts/Assignment/analysis/english-speaker-coverage.json.', 'small')

    # 25
    page('Appendix E. Additional software checks')
    para('The following checks cover the two language implementations and the shared artifact review page.')
    table(['Feature','Verified behavior','Status'],[

      ['Twenty metadata links','Exact source text and Generation details now target each language/artifacts/test_generated_audios/. Every local href/src in the generated page is checked by regression tests. Audio and plot paths remain unchanged.','Verified'],
      ['Telugu decimal meaning','The frontend preserves the decimal point, each fractional digit, trailing zeros and signs in simple decimal expressions.','Verified'],
      ['Training source fingerprints','Future run/resume records hash the language-local dataset.py alongside model/runner helpers. A regression verifies inclusion and sensitivity to changed dataset bytes. Historical run records are preserved.','Verified']], [W*.22,W*.65,W*.13])
    sub('Decimal example and scope')
    story.append(P('ధర 3.14 : ధర మూడు బిందువు ఒకటి నాలుగు','telugu'))
    para('The separator follows the pinned num2words Telugu pointword; the fraction is expanded digit by digit without float conversion. This preserves 3.140 as one-four-zero after the point. Simple positive, negative, leading-point and Telugu-digit forms are covered. Comma-grouped decimals and malformed numeric chains must be spelled out. The source text remains unchanged in generation metadata.')
    sub('Validation results')
    para('The expanded suites pass 41 checks per language plus nine review-server checks. Targeted regressions cover all twenty sidecar links, decimal cases and dataset fingerprints. All ten frozen sentences regenerate with identical WAV hashes and prediction arrays from both code copies. A custom Telugu decimal passes end-to-end CPU inference for trained-phone coverage and waveform generation.')
    para('The assignment artifacts include the software verification record.', 'small')

    page('References')
    refs=[
      ('1','Ren et al. FastSpeech 2: Fast and High-Quality End-to-End Text to Speech. ICLR 2021.','https://arxiv.org/abs/2006.04558'),
      ('2','ming024. FastSpeech2 PyTorch implementation; base checkout d4e79eb52e8b01d24703b2dfc0385544092958f3.','https://github.com/ming024/FastSpeech2'),
      ('3','Kong, Kim and Bae. HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis. NeurIPS 2020.','https://arxiv.org/abs/2010.05646'),
      ('4','AI4Bharat. Svarah dataset card and access conditions.','https://huggingface.co/datasets/ai4bharat/Svarah'),
      ('5','SPRINGLab. IndicTTS Telugu dataset card.','https://huggingface.co/datasets/SPRINGLab/IndicTTS_Telugu'),
      ('6','Montreal Forced Aligner 3.3.9 documentation.','https://montreal-forced-aligner.readthedocs.io/en/v3.3.9/'),
      ('7','Kyubyong/g2p. g2p-en text, homograph and pronunciation implementation.','https://github.com/Kyubyong/g2p'),
      ('8','Phonemizer and eSpeak-NG loader documentation.','https://pypi.org/project/espeakng-loader/'),
      ('9','SpeechBrain. Pretrained ECAPA-TDNN speaker embedding model.','https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb'),
      ('10','Radford et al. Robust Speech Recognition via Large-Scale Weak Supervision.','https://arxiv.org/abs/2212.04356'),
      ('11','Seed-TTS (2024). A Family of High-Quality Versatile Speech Generation Models. Technical report; reference for report organization.','https://arxiv.org/abs/2406.02430')]
    for number,title,url in refs:
        para('['+number+'] '+title,'small')
        story.append(P('<link href="'+url+'" color="#087b78">'+esc(url)+'</link>','small',True))
    sub('Project records')
    for text in [
      'Dataset revisions, split integrity and hours: English/artifacts/DatasetAudit/; English/artifacts/DataSplits/summary.json and verification.json.',
      'Aligned feature records: English/artifacts/Preprocessing/gap-fixed-v2/rebuild-summary.json.',
      'Optimizer logs: English/artifacts/Training/{gap-fixed-v2-20261103,english-finetune-pilot-20261104,english-learned-pauses-pilot-20261104}/<language>/metrics.jsonl; copied unchanged from local output/runs/ backups.',
      'Exact source strings: English/setup/assignment_test_sentences.json; source PDF SHA256 a88e379253c50a018ff6319c05a372a57a0d416cf7c4dc20807d7752e67e323f.',
      'Final output hashes and software checks: English/artifacts/Assignment/verification.json; checkpoint-selection.json; CLI smoke verification.',
      'Computed analysis and ASR hypotheses: English/artifacts/Assignment/analysis/measured-analysis.json and english-asr.json. All numerical tables and report plots derive from these saved records.',
      'Telugu recognition records: English/artifacts/InitialSpeechReview/review.json. Final-recording author ratings: English/artifacts/Assignment/analysis/final-listening-review.json.'
    ]:para(text,'small')

    class NumberedCanvas(Canvas):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self._states=[]
        def showPage(self):self._states.append(dict(self.__dict__));self._startPage()
        def save(self):
            count=len(self._states)
            for state in self._states:
                self.__dict__.update(state)
                self.setFont('Body',7.5);self.setFillColor(gray)
                self.drawString(48,28,'FastSpeech 2 | English and Telugu')
                self.drawRightString(A4[0]-48,28,f'{self._pageNumber} / {count}')
                Canvas.showPage(self)
            Canvas.save(self)
    def header(canvas,doc):
        if doc.page==1:return
        canvas.saveState();canvas.setFillColor(teal);canvas.setFont('Bold',7.6)
        canvas.drawString(48,A4[1]-29,'IMPLEMENTATION AND EVALUATION')
        canvas.setStrokeColor(C('#cad7df'));canvas.setLineWidth(.5);canvas.line(48,A4[1]-36,A4[0]-48,A4[1]-36)
        canvas.restoreState()
    destination=r/'Report/Technical_Report.pdf'
    buffer=BytesIO()
    doc=SimpleDocTemplate(buffer,pagesize=A4,rightMargin=48,leftMargin=48,topMargin=54,bottomMargin=45,
           title='FastSpeech 2 - English and Telugu: Technical Report',author='Parichay Mago',
           subject='Assignment implementation, training losses, generated speech and variance analysis')
    doc.build(story,onFirstPage=header,onLaterPages=header,canvasmaker=NumberedCanvas)
    buffer.seek(0)
    reader=PdfReader(buffer)
    writer=PdfWriter(clone_from=reader)
    writer.metadata={key:value for key,value in reader.metadata.items()
                     if key not in ('/CreationDate','/ModDate')}
    with destination.open('wb') as stream:
        writer.write(stream)
    (r/'Report/report-section-plan.json').write_text(json.dumps({'planned_pages':pages,'planned_page_count':len(pages)},indent=2)+'\n')
    print(json.dumps({'report':str(destination),'planned_pages':len(pages),'bytes':destination.stat().st_size}))


if __name__=='__main__':main()
