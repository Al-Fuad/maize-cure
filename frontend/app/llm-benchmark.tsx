import { ArrowUpRight, CheckCircle2, FlaskConical } from 'lucide-react';
// Published snapshot: backend/evaluation/results/live-2026-11-13/verified-metrics.json
import metrics from './benchmark-metrics.json';
import { Language, number } from './i18n';

// Pin evidence to the published run so future runs cannot silently change these claims.
const evidence = 'https://github.com/Al-Fuad/maize-cure/blob/867c39e/backend/evaluation/';
const models: Record<string, string> = {gemini: 'Gemini Flash Lite', groq: 'GPT-OSS 120B', openrouter: 'Nemotron Lightning'};

export default function LLMBenchmark({ language }: { language: Language }) {
  const text = (en: string, bn: string) => language === 'bn' ? bn : en;
  const n = (value: number, digits = 0) => number(language, value, digits);
  const ratio = (a: number, b: number) => `${n(a)}/${n(b)}`;
  return <section className="benchmark" aria-labelledby="benchmark-title">
    <div className="benchmark-heading">
      <div><p className="eyebrow"><FlaskConical size={14}/>{text('LIVE LLM EVALUATION', 'লাইভ LLM মূল্যায়ন')}</p>
        <h2 id="benchmark-title">{text('Better advice starts with evidence.', 'পরামর্শের মান বুঝুন ফলাফল দেখে।')}</h2>
        <p>{text('How reliably did our advisory models follow the research protocol?', 'আমাদের পরামর্শদাতা মডেলগুলো কতটা নির্ভরযোগ্যভাবে গবেষণার প্রোটোকল মেনেছে?')}</p>
      </div><span className="pill">{text('13 Nov 2026 · One completed run', '১৩ নভেম্বর ২০২৬ · একটি সম্পূর্ণ রান')}</span>
    </div>
    <div className="benchmark-stats">
      <div><strong>{n(metrics.total_responses)}</strong><span>{text('Advisory results', 'পরামর্শের ফলাফল')}</span><small>{text('17 cases × 5 modes', '১৭টি কেস × ৫টি মোড')}</small></div>
      <div><strong>{ratio(metrics.consensus_successes, metrics.consensus_cases)}</strong><span>{text('Consensus succeeded', 'Consensus সফল')}</span><small>{text('Non-Healthy cases · no rules fallback', 'অসুস্থ পাতার কেস · rules fallback লাগেনি')}</small></div>
      <div><strong>{n(metrics.consensus_median_seconds, 3)}<em> s</em></strong><span>{text('Consensus median time', 'Consensus-এর মধ্যম সময়')}</span><small>{text('End-to-end response latency', 'সম্পূর্ণ response পেতে সময়')}</small></div>
    </div>
    <div className="benchmark-table table-scroll" role="region" aria-label={text('LLM comparison', 'LLM তুলনা')} tabIndex={0}>
      <table><caption>{text('Single-provider results · 13 non-Healthy cases each', 'একক provider-এর ফলাফল · প্রতিটিতে ১৩টি অসুস্থ পাতার কেস')}</caption>
        <thead><tr>{[text('Model / provider', 'মডেল / provider'), text('Usable advice', 'ব্যবহারযোগ্য পরামর্শ'),text('Median time', 'মধ্যম সময়'),text('P95 time', 'P95 সময়'),text('Valid JSON / attempts', 'সঠিক JSON / প্রচেষ্টা'),text('Retries', 'পুনঃচেষ্টা')].map(label => <th scope="col" key={label}>{label}</th>)}</tr></thead>
        <tbody>{metrics.providers.map(p => <tr key={p.provider}>
          <th scope="row"><strong>{models[p.provider]}</strong><span>{p.provider}</span><code>{p.model}</code></th>
          <td><span className="benchmark-score">{ratio(p.success,p.cases)}</span></td><td>{n(p.median_seconds,3)} s</td><td>{n(p.p95_seconds,3)} s</td><td>{ratio(p.valid_json_attempts,p.attempts)}</td><td>{n(p.retries)}</td>
        </tr>)}</tbody>
      </table>
    </div>
    <p className="benchmark-footnote">{text('Times include retries and fallback overhead; static Healthy responses are excluded. P95 is the 95th percentile. JSON counts include attempts in both single-provider and consensus modes.', 'সময়ের হিসাবে পুনঃচেষ্টা ও fallback অন্তর্ভুক্ত; স্থির Healthy response বাদ দেওয়া হয়েছে। P95 হলো ৯৫তম percentile। JSON হিসাবে single-provider ও consensus—দুই মোডের প্রচেষ্টা আছে।')}</p>
    <div className="benchmark-findings">
      <article><CheckCircle2 size={20}/><div><h3>{text('Guardrails held in this run', 'এই রানে সুরক্ষার নিয়ম মানা হয়েছে')}</h3><p>{text('55/55 low-confidence, ambiguous or Healthy results omitted chemical-category advice. All 5 ambiguous results displayed a mixed-infection warning and failed confidence validation.', 'কম confidence, দ্ব্যর্থক বা Healthy—৫৫/৫৫টি ফলে রাসায়নিক বিভাগের পরামর্শ বাদ ছিল। ৫টি দ্ব্যর্থক ফলেই mixed-infection সতর্কতা ছিল এবং confidence validation পাস করেনি।')}</p></div></article>
      <article><FlaskConical size={20}/><div><h3>{text('What we observed', 'যা দেখা গেছে')}</h3><p>{text('Groq had the lowest median time in this run. Gemini used rules fallback once after HTTP 503. Consensus remained successful across all 13 non-Healthy cases.', 'এই রানে Groq-এর মধ্যম সময় সবচেয়ে কম ছিল। HTTP 503-এর পর Gemini একবার rules fallback ব্যবহার করেছে। অসুস্থ পাতার ১৩টি কেসেই consensus সফল ছিল।')}</p></div></article>
    </div>
    <details className="benchmark-method"><summary>{text('Method, limitations & relation to the paper', 'পদ্ধতি, সীমাবদ্ধতা ও পেপারের সঙ্গে সম্পর্ক')}</summary>
      <p>{text('17 fixed cases covered four classes, English/Bengali display, high/low confidence and one ambiguous disease pair. Five modes: rules, three single providers and consensus requiring at least two models. One repetition, cleared caches, 83 provider attempts including 5 retries. The interrupted diagnostic run is excluded.', '১৭টি নির্দিষ্ট কেসে চারটি class, বাংলা/English প্রদর্শন, বেশি/কম confidence এবং একটি দ্ব্যর্থক রোগের জোড়া ছিল। পাঁচটি মোড: rules, তিনটি single provider ও অন্তত দুই মডেলের consensus। একবার পরীক্ষা, cache খালি; ৫টি পুনঃচেষ্টাসহ ৮৩টি provider প্রচেষ্টা। অসম্পূর্ণ diagnostic রান বাদ দেওয়া হয়েছে।')}</p>
      <p>{text('This measures protocol reliability, not treatment correctness, field performance or classification accuracy. It is not a definitive model ranking. Expert review is still unscored. Bengali text uses stored translations; LLM translation quality was not evaluated.', 'এটি প্রোটোকল অনুসরণের নির্ভরযোগ্যতা মাপে; চিকিৎসার সঠিকতা, মাঠের কার্যকারিতা বা classification accuracy নয়। এটি চূড়ান্ত model ranking নয়। বিশেষজ্ঞ মূল্যায়নে এখনো স্কোর দেওয়া হয়নি। বাংলা লেখায় সংরক্ষিত অনুবাদ ব্যবহৃত হয়; LLM-এর অনুবাদের মান পরীক্ষা হয়নি।')}</p>
      <p>{text('The current LLM selects from an allowed protocol. These results are not directly comparable to the paper’s free-form correct-cure and dosage rubric. The original image-classification results are shown separately below.', 'বর্তমান LLM অনুমোদিত প্রোটোকল থেকে পরামর্শ বেছে নেয়। তাই এই ফলাফল পেপারের free-form correct-cure ও dosage rubric-এর সঙ্গে সরাসরি তুলনাযোগ্য নয়। মূল image-classification-এর ফলাফল নিচে আলাদাভাবে আছে।')}</p>
    </details>
    <div className="benchmark-evidence"><span>{text('Inspect the evidence', 'মূল প্রমাণ দেখুন')}</span>{[
      ['reports/live-2026-11-13.md', text('Full report', 'পূর্ণ রিপোর্ট')],
      ['results/live-2026-11-13/verified-metrics.json', text('Verified metrics', 'যাচাইকৃত metrics')],
      ['results/live-2026-11-13/responses.jsonl', text('85 responses', '৮৫টি response')],
      ['results/live-2026-11-13/audit.jsonl', text('Audit log', 'Audit log')],
      ['results/live-2026-11-13/expert-review.csv', text('Expert review · unscored', 'বিশেষজ্ঞ review · স্কোর বাকি')]
    ].map(([path,label]) => <a href={evidence+path} target="_blank" rel="noreferrer" key={path}>{label}<ArrowUpRight size={14}/></a>)}</div>
  </section>;
}
