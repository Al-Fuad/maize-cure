'use client';
import { useEffect, useState } from 'react';
import { Language, translate, healthyCare, uncertainHealthy } from './i18n';

type Result = {prediction:string;confidence:number};
export default function Guidance({result,language}:{result:Result;language:Language}) {
 const t=(text:string)=>translate(language,text);
 const key=JSON.stringify([result.prediction,result.confidence,language]);
 const [retry,setRetry]=useState(0);
 const [state,setState]=useState<{key:string;text?:string;error?:boolean}>({key:''});
 useEffect(()=>{
  if(result.prediction==='Healthy')return;
  const controller=new AbortController();
  setState({key});
  const api=process.env.NEXT_PUBLIC_API_URL||(process.env.NODE_ENV==='development'?'http://localhost:8000':'');
  async function load(){
   try{
    if(!api)throw new Error('Missing API');
    const response=await fetch(`${api}/api/advice`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({prediction:result.prediction,confidence:result.confidence,language}),signal:AbortSignal.any([controller.signal,AbortSignal.timeout(60000)])});
    if(!response.ok)throw new Error('Advice unavailable');
    const data=await response.json();
    if(typeof data.text!=='string'||data.language!==language)throw new Error('Invalid advice');
    if(!controller.signal.aborted)setState({key,text:data.text});
   }catch{if(!controller.signal.aborted)setState({key,error:true});}
  }
  void load();
  return ()=>controller.abort();
 },[key,result.prediction,result.confidence,language,retry]);
 const healthy=result.prediction==='Healthy';
 const current=state.key===key?state:null;
 return <section className="guidance" aria-live="polite"><p className="eyebrow">{t('What to do next')}</p><h3>{t(healthy?'Healthy leaf care':'AI crop guidance')}</h3>
 {healthy?<p>{(result.confidence<.8?uncertainHealthy[language]:'')+healthyCare[language]}</p>:current?.error?<><p role="status">{t('AI guidance is unavailable right now. Your prediction is still available. Consult a local agricultural extension officer if symptoms persist.')}</p><button className="text-button" onClick={()=>setRetry(v=>v+1)}>{t('Try guidance again')}</button></>:current?.text?<p className="advice-text">{current.text}</p>:<p role="status">{t('Preparing guidance…')}</p>}
 <p className="small-muted">{t('General guidance; confirm symptoms in the field before taking action.')}</p>
 {!healthy&&<p className="small-muted">{t('Only the predicted class and confidence category are sent to the AI service; your photo is not sent.')}</p>}
 </section>;
}
