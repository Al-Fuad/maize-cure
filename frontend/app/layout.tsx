import type { Metadata } from 'next';
import './globals.css';
export const metadata: Metadata = { title: 'Maize Cure — A closer look at leaf health', description: 'Explore four trained maize leaf classifiers, test your images, and inspect the original evaluation results.' };
export default function RootLayout({children}: {children: React.ReactNode}) {return <html lang="en"><body>{children}</body></html>}
