import bn from './bn.json';
export type Language = 'en' | 'bn';
export function translate(language: Language, text: string) {
 return language === 'bn' ? (bn as Record<string,string>)[text] ?? text : text;
}
export function number(language:Language, value:number, digits=0) {
 return new Intl.NumberFormat(language==='bn'?'bn-BD':'en-US',{maximumFractionDigits:digits,minimumFractionDigits:digits}).format(value);
}
export const healthyCare = {
 en:'The model suggests a healthy leaf. Keep checking leaves regularly, maintain good drainage and balanced nutrition, and avoid unnecessary pesticide use. If you notice spreading spots, wilting, or other symptoms, take a clearer photo and consult a local agricultural extension officer.',
 bn:'মডেলের ফল অনুযায়ী পাতাটি সুস্থ মনে হচ্ছে। নিয়মিত পাতা পর্যবেক্ষণ করুন, পানি নিষ্কাশন ভালো রাখুন এবং সুষম পুষ্টি নিশ্চিত করুন। অপ্রয়োজনে কীটনাশক ব্যবহার করবেন না। দাগ ছড়িয়ে পড়া, ঢলে পড়া বা অন্য উপসর্গ দেখা গেলে পরিষ্কার ছবি তুলুন এবং স্থানীয় কৃষি কর্মকর্তার পরামর্শ নিন।'
};
export const uncertainHealthy = {en:'The model is uncertain; this result does not confirm that the plant is healthy. Retake a clear photo and check symptoms in the field. ',bn:'মডেল এই ফলে অনিশ্চিত; গাছটি সুস্থ বলে নিশ্চিত হওয়া যায় না। পরিষ্কার ছবি তুলে আবার পরীক্ষা করুন এবং মাঠে উপসর্গ মিলিয়ে দেখুন। '};
