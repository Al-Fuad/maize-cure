'use client';
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { Language, translate, number } from './i18n';
export default function LossChart({data,language}:{data:{epoch:number;Training:number;Validation?:number}[];language:Language}) {
 const t=(text:string)=>translate(language,text);
 return <ResponsiveContainer width="100%" height="100%"><LineChart data={data}><CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e5e9e0"/><XAxis dataKey="epoch" tickLine={false} axisLine={false} minTickGap={35} tickFormatter={v=>number(language,Number(v))}/><YAxis tickLine={false} axisLine={false} width={42} tickFormatter={v=>number(language,Number(v),2)}/><Tooltip labelFormatter={v=>`${t('Epoch')} ${number(language,Number(v))}`} formatter={v=>number(language,Number(v),4)}/><Legend/><Line name={t('Training')} type="monotone" dataKey="Training" stroke="#436847" dot={false} strokeWidth={2}/><Line name={t('Validation')} type="monotone" dataKey="Validation" stroke="#c79743" dot={false} strokeWidth={2}/></LineChart></ResponsiveContainer>;
}
