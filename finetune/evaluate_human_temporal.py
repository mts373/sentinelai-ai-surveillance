import json, gc, re
from pathlib import Path
from collections import Counter
import cv2
import numpy as np
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel

ROOT=Path(r'C:\SentinelAI_Qwen')
MODEL='Qwen/Qwen2.5-VL-7B-Instruct'
ADAPTER=ROOT/'models'/'sentinelai_qwen25vl_lora'
LABELS=ROOT/'dataset'/'test_evaluation'/'human_temporal_labels.json'
OUT=ROOT/'dataset'/'test_evaluation'/'human_temporal_evaluation.json'
CLASSES=['Normal','Fire','Fight','Road Accident']
MIN_PIXELS=200704; MAX_PIXELS=602112; SAMPLE_FPS=2.0; MAX_NEW_TOKENS=150
PROMPT=('Analyze this surveillance video clip. Classify the scene as exactly one of: '
        'Normal, Fire, Fight, or Road Accident. Return ONLY a JSON object with the fields '
        'classification, evidence, and incident_summary.')

def load_labels():
    if not LABELS.exists(): raise FileNotFoundError(f'Human labels not found:\n{LABELS}')
    data=json.loads(LABELS.read_text(encoding='utf-8'))
    if not isinstance(data,list): raise ValueError('human_temporal_labels.json must be a JSON list.')
    valid=[]; skipped=0
    for i,x in enumerate(data):
        # CRITICAL: human_label is the ground truth. source_label is NOT.
        label=x.get('human_label')
        if label in CLASSES and all(k in x for k in ('video_path','start','end')): valid.append(x)
        else: skipped+=1
    print(f'Human annotations loaded: {len(data)}')
    print(f'Valid annotations: {len(valid)}')
    print(f'Skipped/invalid annotations: {skipped}')
    if not valid: raise RuntimeError('No valid human annotations found.')
    print('Human-label distribution:')
    c=Counter(x['human_label'] for x in valid)
    for k in CLASSES: print(f'  {k:<15}{c[k]}')
    return valid

def load_model():
    if not ADAPTER.exists(): raise FileNotFoundError(f'LoRA adapter not found:\n{ADAPTER}')
    bnb=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
    print('Loading Qwen2.5-VL...')
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL,quantization_config=bnb,device_map={'':0},torch_dtype=torch.bfloat16,low_cpu_mem_usage=True)
    print('Base model: OK')
    model=PeftModel.from_pretrained(model,str(ADAPTER),is_trainable=False)
    model.eval(); print('LoRA adapter: OK')
    return model

def load_processor():
    p=AutoProcessor.from_pretrained(MODEL,min_pixels=MIN_PIXELS,max_pixels=MAX_PIXELS)
    print('Processor: OK'); return p

def read_window(path,start,end):
    cap=cv2.VideoCapture(str(path))
    if not cap.isOpened(): raise RuntimeError(f'Cannot open video: {path}')
    fps=cap.get(cv2.CAP_PROP_FPS) or 30.0
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration=total/fps if total else 0
    start=max(0.,float(start)); end=min(float(end),duration)
    if end<=start: cap.release(); raise RuntimeError(f'Invalid window {start}->{end}')
    # Sequential read: avoids unreliable UCF-Crime seeking.
    a=int(np.floor(start*fps)); b=int(np.ceil(end*fps)); frames=[]; i=0
    while i<b:
        ok,frame=cap.read()
        if not ok: break
        if i>=a: frames.append(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        i+=1
    cap.release()
    if not frames: raise RuntimeError(f'No frames in {path} {start}->{end}')
    n=max(1,int(round((end-start)*SAMPLE_FPS)))
    if len(frames)>n:
        idx=np.linspace(0,len(frames)-1,n).astype(int); frames=[frames[j] for j in idx]
    t=torch.from_numpy(np.stack(frames)).permute(0,3,1,2).contiguous()
    return t,fps,duration

def parse(text):
    text=re.sub(r'```(?:json)?','',text,flags=re.I).strip()
    try:
        x=json.loads(text); y=x.get('classification')
        if y in CLASSES: return y
    except Exception: pass
    m=re.search(r'\{.*?\}',text,re.S)
    if m:
        try:
            y=json.loads(m.group()).get('classification')
            if y in CLASSES: return y
        except Exception: pass
    for y in CLASSES:
        if re.search(rf'\b{re.escape(y)}\b',text,re.I): return y
    return None

@torch.inference_mode()
def predict(model,processor,video):
    messages=[{'role':'user','content':[{'type':'video','video':'temporal_window.mp4'},{'type':'text','text':PROMPT}]}]
    text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    inputs=processor(text=[text],videos=[video],padding=True,return_tensors='pt')
    inputs={k:(v.to('cuda') if torch.is_tensor(v) else v) for k,v in inputs.items()}
    ids=model.generate(**inputs,max_new_tokens=MAX_NEW_TOKENS,do_sample=False)
    n=inputs['input_ids'].shape[-1]
    raw=processor.batch_decode(ids[:,n:],skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
    return parse(raw),raw

def matrix(results):
    m={a:{b:0 for b in CLASSES} for a in CLASSES}
    for x in results:
        if x['ground_truth'] in CLASSES and x['prediction'] in CLASSES: m[x['ground_truth']][x['prediction']]+=1
    return m

def main():
    print('\n'+'='*70+'\nSENTINELAI - HUMAN TEMPORAL EVALUATION\n'+'='*70)
    if not torch.cuda.is_available(): raise RuntimeError('CUDA is not available.')
    print('CUDA: True'); print('GPU:',torch.cuda.get_device_name(0)); print('GPU memory:',f"{torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB")
    labels=load_labels()
    print('\nLoading LoRA adapter...'); model=load_model()
    print('\nLoading processor...'); processor=load_processor()
    print('\n'+'='*70+'\nSTARTING EVALUATION\n'+'='*70); print('Windows to evaluate:',len(labels))
    results=[]
    for no,item in enumerate(labels,1):
        path=Path(item['video_path']); start=float(item['start']); end=float(item['end']); truth=item['human_label']
        print(f'\n[{no}/{len(labels)}] {path.name}'); print(f'Window: {start:.2f}s → {end:.2f}s'); print('Human ground truth:',truth)
        rec={'video_path':str(path),'source_label':item.get('source_label'),'start':start,'end':end,'duration':item.get('duration'),'ground_truth':truth,'prediction':None,'raw_output':None}
        try:
            if not path.exists(): raise FileNotFoundError(f'Video not found: {path}')
            video,fps,dur=read_window(path,start,end); print(f'Source FPS: {fps:.3f}'); print('Frames sent to Qwen:',video.shape[0])
            pred,raw=predict(model,processor,video); rec['prediction']=pred; rec['raw_output']=raw
            print('Predicted:',pred); print('Raw output:'); print(raw); del video; torch.cuda.empty_cache()
        except Exception as e:
            rec['error']=repr(e); print('ERROR:',repr(e)); torch.cuda.empty_cache()
        results.append(rec)
    evaluated=[x for x in results if x['prediction'] in CLASSES]; correct=sum(x['prediction']==x['ground_truth'] for x in evaluated); acc=correct/len(evaluated) if evaluated else 0
    m=matrix(evaluated)
    print('\n'+'='*70+'\nHUMAN-GROUND-TRUTH CONFUSION MATRIX\n'+'='*70)
    print(f"{'Actual':<20}{'Normal':>10}{'Fire':>10}{'Fight':>10}{'Road Accident':>16}")
    for a in CLASSES: print(f"{a:<20}{m[a]['Normal']:>10}{m[a]['Fire']:>10}{m[a]['Fight']:>10}{m[a]['Road Accident']:>16}")
    print('\n'+'='*70+'\nWINDOW RESULTS\n'+'='*70); print(f'Accuracy: {acc:.4f}'); print(f'Accuracy percentage: {acc*100:.2f}%'); print('Correct:',correct); print('Evaluated:',len(evaluated)); print('Unparsed/errors:',len(results)-len(evaluated))
    print('\n'+'='*70+'\nPER-CLASS PERFORMANCE\n'+'='*70)
    for a in CLASSES:
        items=[x for x in evaluated if x['ground_truth']==a]; ok=sum(x['prediction']==a for x in items); ca=ok/len(items) if items else 0
        print(f'{a:<20}{len(items):>5} samples   {ok:>5} correct   {ca*100:>6.2f}%')
    output={'model':MODEL,'adapter':str(ADAPTER),'human_labels_file':str(LABELS),'total_human_annotations':len(labels),'evaluated':len(evaluated),'unparsed_or_errors':len(results)-len(evaluated),'correct':correct,'accuracy':acc,'accuracy_percent':acc*100,'confusion_matrix':m,'results':results}
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(output,indent=2,ensure_ascii=False),encoding='utf-8')
    print('\nResults saved:'); print(OUT); print('\nDO NOT RETRAIN YET.')
    del model,processor; gc.collect(); torch.cuda.empty_cache()

if __name__=='__main__': main()