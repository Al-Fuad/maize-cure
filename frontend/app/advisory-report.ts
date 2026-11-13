import { Language, number, translate } from './i18n';

export type Advice = {
 text:string; language:Language; source:'static'|'rule_based'|'llm_single'|'llm_consensus';
 validated:boolean; mixed_infection:boolean; llm_models:string[]; consensus_count:number;
 votes:Record<string,Record<string,number>>; fallback_reason?:string;
};
export type LeafResult = {prediction_id?:string;models_used?:string[];classification_mode?:string;cascade_threshold?:number;model_id:string;prediction:string;confidence:number;inference_ms:number;probabilities:{class_name:string;probability:number}[]};
export type Controls = {mode:'consensus'|'single'|'rules';provider:string;min_confidence:number;min_consensus:number};
export const sourceLabel = (source:string) => ({static:'Healthy leaf care',rule_based:'Rule-based guidance',llm_single:'Single-model selection',llm_consensus:'Multi-model consensus'}[source] || source);

export async function createReport(result:LeafResult, advice:Advice, controls:Controls, language:Language):Promise<Blob> {
 const bn=language==='bn';
 const t=(s:string)=>translate(language,s);
 const family=bn?'Noto Sans Bengali':'DM Sans';
 const loaded=await document.fonts.load(`20px "${family}"`);
 await document.fonts.ready;
 if(bn&&!loaded.length)throw new Error('Bengali font unavailable');
 const canvas=document.createElement('canvas');
 canvas.width=1200;
 const ctx=canvas.getContext('2d');
 if(!ctx)throw new Error('Canvas unavailable');
 const font=`24px "${family}", sans-serif`;
 ctx.font=font;
 const lines:string[]=[];
 const wrap=(text:string)=>{
  for(const paragraph of text.split('\n')){
   let line='';
   for(const word of paragraph.split(/\s+/)){
    const next=line?`${line} ${word}`:word;
    if(ctx.measureText(next).width>1040&&line){lines.push(line);line=word;}else line=next;
   }
   lines.push(line);
  }
 };
 const date=new Intl.DateTimeFormat(bn?'bn-BD':'en-GB',{dateStyle:'medium',timeStyle:'short'}).format(new Date());
 wrap(`${t('Prediction')}: ${t(result.prediction.replaceAll('_',' '))} · ${number(language,result.confidence*100,2)}%`);
 wrap(`${t('Prediction model')}: ${result.model_id} · ${number(language,result.inference_ms)} ${t('ms')}`);
 if(result.models_used)wrap(`${t('Models used')}: ${result.models_used.join(' + ')}`);
 if(result.classification_mode==='cascade')wrap(`${t('Cascade threshold')}: ${number(language,(result.cascade_threshold??.8)*100)}%`);
 wrap(`${t('Advice source')}: ${t(sourceLabel(advice.source))}`);
 wrap(`${t('Confidence threshold')}: ${number(language,controls.min_confidence*100)}%`);
 if(controls.mode==='consensus')wrap(`${t('Required votes')}: ${number(language,controls.min_consensus)}`);
 wrap(`${t('Validation')}: ${t(advice.validated?'Threshold passed':'Needs field confirmation')}`);
 if(advice.mixed_infection)wrap(t('Possible mixed infection / ambiguous scores'));
 if(advice.llm_models.length)wrap(`${t('Responding models')}: ${advice.llm_models.join(', ')}`);
 wrap('');
 for(const score of result.probabilities)wrap(`${t(score.class_name.replaceAll('_',' '))}: ${number(language,score.probability*100,2)}%`);
 wrap('\n'+advice.text);
 wrap('\n'+t('General guidance; confirm symptoms in the field before taking action.'));
 canvas.height=240+lines.length*42+80;
 ctx.fillStyle='#f8f9f4';ctx.fillRect(0,0,canvas.width,canvas.height);
 ctx.fillStyle='#416348';ctx.fillRect(0,0,canvas.width,12);
 ctx.fillStyle='#27392c';ctx.font=`bold 38px "${family}", sans-serif`;
 ctx.fillText(t('Disease Advisory Report'),80,90);
 ctx.font=font;ctx.fillStyle='#63725e';ctx.fillText(`Maize Cure · ${date}`,80,145);
 ctx.fillStyle='#27392c';lines.forEach((line,index)=>ctx.fillText(line,80,220+index*42));
 return new Promise((resolve,reject)=>canvas.toBlob(blob=>blob?resolve(blob):reject(new Error('Export failed')),'image/png'));
}
