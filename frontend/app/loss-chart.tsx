'use client';
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
export default function LossChart({data}:{data:{epoch:number;Training:number;Validation?:number}[]}) {
 return <ResponsiveContainer width="100%" height="100%"><LineChart data={data}><CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e5e9e0"/><XAxis dataKey="epoch" tickLine={false} axisLine={false} minTickGap={35}/><YAxis tickLine={false} axisLine={false} width={42}/><Tooltip/><Legend/><Line type="monotone" dataKey="Training" stroke="#436847" dot={false} strokeWidth={2}/><Line type="monotone" dataKey="Validation" stroke="#c79743" dot={false} strokeWidth={2}/></LineChart></ResponsiveContainer>;
}
