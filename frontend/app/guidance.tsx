'use client';
import { useEffect, useState } from 'react';
import { Language, translate, number } from './i18n';
import { Advice, LeafResult, Controls, createReport, sourceLabel } from './advisory-report';

const defaults:Controls={mode:'consensus',provider:'gemini',min_confidence:.8,min_consensus:2};
const api=process.env.NEXT_PUBLIC_API_URL||(process.env.NODE_ENV==='development'?'http://localhost:8000':'');
export default function Guidance({result,language}:{result:LeafResult;language:Language}) {
 const t=(text:string)=>translate(language,text);
 const [draft,setDraft]=useState<Controls>(defaults);
 const [controls,setControls]=useState<Controls>(defaults);
 const [providers,setProviders]=useState<{id:string;configured:boolean;model:string|null}[]|null>(null);
 const [retry,setRetry]=useState(0);
 const key=JSON.stringify([result,language,controls]);
 const [state,setState]=useState<{key:string;data?:Advice;error?:boolean}>({key:''});
 const [report,setReport]=useState<{key:string;url:string}|null>(null);
 const [exporting,setExporting]=useState(false);
 const [reportError,setReportError]=useState(false);
 useEffect(()=>{const controller=new AbortController();
  fetch(`${api}/api/advice/config`,{signal:controller.signal}).then(r=>{if(!r.ok)throw new Error();return r.json();}).then(d=>{if(Array.isArray(d.providers))setProviders(d.providers);}).catch(()=>{});
  return ()=>controller.abort();
 },[]);
 useEffect(()=>()=>{if(report)URL.revokeObjectURL(report.url);},[report]);
 useEffect(()=>{
  const controller=new AbortController();setState({key});setReportError(false);
  async function load(){try{
   if(!api)throw new Error('Missing API');
   const response=await fetch(`${api}/api/advice`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({prediction_id:result.prediction_id,prediction:result.prediction,confidence:result.confidence,probabilities:result.probabilities,language,...controls}),signal:AbortSignal.any([controller.signal,AbortSignal.timeout(60000)])});
   if(!response.ok)throw new Error('Advice unavailable');
   const data=await response.json();
   if(typeof data.text!=='string'||data.language!==language)throw new Error('Invalid advice');
   if(!controller.signal.aborted)setState({key,data});
  }catch{if(!controller.signal.aborted)setState({key,error:true});}}
  void load();return ()=>controller.abort();
 },[key,result,language,controls,retry]);
 const current=state.key===key?state:null;
 const advice=current?.data;
 async function exportReport(){
  if(!advice)return;setExporting(true);setReportError(false);
  try{const blob=await createReport(result,advice,controls,language);setReport({key,url:URL.createObjectURL(blob)});}
  catch{setReportError(true);}finally{setExporting(false);}
 }
 const fallback=advice?.fallback_reason;
 const status=fallback==='insufficient_models'?'Not enough configured models; showing protocol guidance.':fallback==='rules_selected'?'Rule-based mode selected.':fallback?'Models did not reach the requested agreement or were unavailable; showing protocol guidance.':'';
 return <section className="guidance" aria-live="polite">
  <p className="eyebrow">{t('What to do next')}</p><h3>{t('Crop advisory')}</h3>
  <details className="research-controls"><summary>{t('Research controls')}</summary>
   <p className="small-muted">{t('Choose how advice is selected. Changes apply when you press Update guidance.')}</p>
   <div className="control-grid">
    <label>{t('Advisory mode')}<select value={draft.mode} onChange={e=>setDraft({...draft,mode:e.target.value as Controls['mode']})}>
     <option value="consensus">{t('Multi-model consensus')}</option><option value="single">{t('Single model (research)')}</option><option value="rules">{t('Rules only')}</option>
    </select></label>
    {draft.mode==='single'&&<label>{t('LLM provider')}<select value={draft.provider} onChange={e=>setDraft({...draft,provider:e.target.value})}>{['gemini','groq','openrouter'].map(p=><option key={p} value={p}>{p}</option>)}</select></label>}
    <label>{t('Confidence threshold')}<select value={draft.min_confidence} onChange={e=>setDraft({...draft,min_confidence:Number(e.target.value)})}>{[.5,.6,.7,.8,.9,.95,.99].map(v=><option key={v} value={v}>{number(language,v*100)}%</option>)}</select></label>
    {draft.mode==='consensus'&&<label>{t('Required votes')}<select value={draft.min_consensus} onChange={e=>setDraft({...draft,min_consensus:Number(e.target.value)})}><option value={2}>{number(language,2)}</option><option value={3}>{number(language,3)}</option></select></label>}
   </div>
   <p className="small-muted">{t('Chemical advice always requires at least 80% confidence and no mixed-infection flag. A passed threshold is not a confirmed diagnosis.')}</p>
   <button className="browse" onClick={()=>{setControls({...draft});setRetry(v=>v+1);}}>{t('Update guidance')}</button>
   <div className="provider-status">{providers?providers.map(p=><p key={p.id}><strong>{p.id}</strong> · {t(p.configured?'Configured':'Not configured')}{p.model&&<span> — {p.model}</span>}</p>):<p>{t('Provider configuration unavailable')}</p>}</div>
  </details>
  {advice?.mixed_infection&&<p className="notice" role="status"><strong>{t('Possible mixed infection / ambiguous scores')}</strong><br/>{t('Two disease scores are close. This is not proof of multiple infections. Retake a clear photo and seek field confirmation.')}</p>}
  {current?.error?<><p role="status">{t('AI guidance is unavailable right now. Your prediction is still available. Consult a local agricultural extension officer if symptoms persist.')}</p><button className="text-button" onClick={()=>setRetry(v=>v+1)}>{t('Try guidance again')}</button></>:advice?<>
   <p className="source-badge">{t(sourceLabel(advice.source))}</p>
   {status&&<p className="small-muted">{t(status)}</p>}
   <p className="small-muted">{t('Validation')}: {t(advice.validated?'Threshold passed':'Needs field confirmation')}</p>
   <p className="advice-text">{advice.text}</p>
   {!!advice.llm_models?.length&&<details className="vote-details"><summary>{t('Model agreement details')}</summary><p>{advice.llm_models.join(' · ')}</p>{Object.entries(advice.votes||{}).map(([category,items])=><div key={category}>{Object.entries(items).map(([item,votes])=><p key={item}>{number(language,votes)} {t('votes')} — {item}</p>)}</div>)}<p className="small-muted">{t('Protocol strings are shown in their original English for research comparison.')}</p></details>}
   <button className="browse report-button" disabled={exporting} onClick={exportReport}>{t(exporting?'Creating report…':'Create advisory report')}</button>
   {reportError&&<p role="alert">{t('Report export failed. Check your connection so the Bengali font can load, then retry.')}</p>}
   {report?.key===key&&<div className="advisory-report"><a className="browse" href={report.url} download={`maize-cure-${result.prediction}-${language}.png`}>{t('Download PNG report')}</a><img src={report.url} alt={t('Disease Advisory Report')}/></div>}
  </>:<p role="status">{t('Preparing guidance…')}</p>}
  <p className="small-muted">{t('General guidance; confirm symptoms in the field before taking action.')}</p>
  <p className="small-muted">{t('Only the predicted class and confidence category are sent to the AI service; your photo is not sent.')}</p>
 </section>;
}
